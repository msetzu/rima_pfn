from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy
import pandas
import sdm
import torch
from sdm import TableTensor
from sdm.models import TabICLv2
from sdm.evaluation import to_class_indices
from tabpfn import TabPFNClassifier
import torch.nn.functional as F
from tqdm import tqdm

import rima_pfn
from rima_pfn.modeling.tasks import Task


def task_loss(task: str, predictions, labels, y_scale: float):
    if task == "classification":
        scores, idx = to_class_indices(predictions, labels, missing_score=0.0)
        loss = F.nll_loss(scores.float().clamp_min(1e-6).log(), idx)
        metric = (scores.argmax(-1) == idx).float().mean()  # accuracy

    else:
        # garbage
        q = predictions.numerical.float()  # [R, n_quantiles]
        t = labels.numerical.float()  # [R, 1]
        levels = torch.arange(1, q.size(-1) + 1, device=q.device, dtype=q.dtype) / (q.size(-1) + 1)
        diff = t - q
        loss = torch.maximum(levels * diff, (levels - 1) * diff).mean() / y_scale  # pinball
        metric = ((q.mean(-1, keepdims=True) - t) ** 2).mean().sqrt()  # RMSE of the mean

    return loss, metric


@torch.no_grad()
def validate_model(
    model,
    table,
    label,
    context_index,
    validation_index,
    task,
    y_scale,
    num_estimators,
    batch_size,
    amp_contextmanager
):
    model.eval()
    context = table[context_index]
    with amp_contextmanager():
        model.fit(
            context.drop_columns(label),
            context.select_columns(label),
            num_estimators=num_estimators,
        )

    total_loss = total_metric = 0.0
    for chunk in validation_index.split(batch_size):
        query = table[chunk]
        with amp_contextmanager():
            predictions = model.predict(query.drop_columns(label))

        loss, metric = task_loss(task, predictions, query.select_columns(label), y_scale)
        total_loss += loss.item() * len(chunk)
        total_metric += metric.item() * len(chunk)
    model.clear()
    n = len(validation_index)

    return total_loss / n, total_metric / n


class TabularModel:
    def __init__(self):
        self.model = None
        self.is_fit = False

        # mapping from and to torch in finetuning
        self.class_indices = None
        self.index_classes = None

    def overrides(self, label: str, task: Task) -> dict:
        return {label: "categorical"} if task == Task.CLASSIFICATION else {label: "numerical"}

    def fit_sdm(self, data: pandas.DataFrame, label: str, task: Task, **kwargs) -> TabularModel:
        """Implementation w/ the SDM library: https://github.com/NVIDIA/structured-data-models"""
        overrides = self.overrides(label, task)
        table = TableTensor.from_pandas(
            df=data,
            stypes=sdm.infer_stypes(data, overrides=overrides),
            device="cpu",
        )

        self.model = TabICLv2(device="cpu")
        self.model.fit(
            x=table.drop_columns(label),
            y=table[:, label],
            num_estimators=32,
            **kwargs,
        )
        self.is_fit = True

        return self

    def predict(self, data: pandas.DataFrame, label: str, task: Task, **kwargs) -> torch.Tensor:
        """Implementation w/ the SDM library: https://github.com/NVIDIA/structured-data-models"""
        overrides = self.overrides(label, task)
        inference_data = data.drop(columns=[label], errors="ignore")
        table = TableTensor.from_pandas(
            df=inference_data,
            stypes=sdm.infer_stypes(inference_data, overrides=overrides),
            device="cpu",
        )

        # also predictions are tabletensors: convert to numpy
        table_predictions = self.model.predict(table, **kwargs).to_pandas()
        predictions = table_predictions.idxmax(axis=1).to_numpy()

        return predictions


    def predict_with_fit(
        self,
        context_df: pandas.DataFrame,
        query_df: pandas.DataFrame,
        task: dict,
        as_objects: bool = False
    ) -> numpy.ndarray:
        """Predict for fine-tuned models, which require an additional fit step for KV caching."""
        data = pandas.concat([context_df, query_df], axis="rows", ignore_index=True)
        overrides = self.overrides(task["label"], task["task"])
        types = sdm.infer_stypes(data, overrides=overrides)
        table = sdm.TableTensor.from_pandas(
            data,
            stypes=types,
            device="cpu"
        )
        context_table, query_table = table[:len(context_df)], table[len(context_df):]

        self.model.fit(context_table.drop_columns(task["label"]), context_table.select_columns(task["label"]), num_estimators=1)
        table_predictions = self.model.predict(query_table.drop_columns(task["label"])).to_pandas()

        predictions = table_predictions.idxmax(axis=1).to_numpy()
        if self.class_indices is not None:
            # classes have been mapped from categorical to integer for torch finetuning, which can results in integers
            #
            # predictions may come out as strings for some reason
            if numpy.issubdtype(predictions.dtype, numpy.str_) or numpy.issubdtype(predictions.dtype, numpy.object_):
                predictions = numpy.array([int(float(c)) for c in predictions])
            elif numpy.issubdtype(predictions.dtype, numpy.floating) and task["task"] == Task.CLASSIFICATION:
                predictions = numpy.array([int(c) for c in predictions])

            if as_objects:
                predictions = numpy.array([self.index_classes[c] for c in predictions])

        return predictions


    def classes_mapping(self, data: pandas.DataFrame, task: dict):
        """Computes a class mapping for the given task, mapping categorical classes to integer for torch finetuning."""
        unique_classes = sorted(data[task["label"]].unique())
        self.class_indices = {class_name: i for i, class_name in enumerate(unique_classes)}
        self.index_classes = {i: class_name for i, class_name in enumerate(unique_classes)}

        return self.class_indices, self.index_classes

    @staticmethod
    def train_val_split(index_size: int, fraction: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
        perm = torch.randperm(index_size, generator=rima_pfn.TORCH_GENERATOR)
        train_size = max(1, int(index_size * fraction))

        return perm[:train_size].to(rima_pfn.TORCH_DEVICE), perm[train_size:].to(rima_pfn.TORCH_DEVICE)


    @staticmethod
    def cosine_scheduler(step: int, warmup: int, total_steps: int) -> float:
        """Cosine learning rate scheduler with warmup."""
        if step < warmup:
            return (step + 1) / warmup

        progress = (step - warmup) / max(1, total_steps - warmup)

        return 0.5 * (1 + math.cos(math.pi * progress))


    @staticmethod
    def amp_contextmanager():
        return torch.autocast(
            device_type=rima_pfn.TORCH_DEVICE.type,
            dtype=torch.bfloat16,
            enabled=rima_pfn.TORCH_DEVICE.type == "cuda"
        )


    def model_snapshot(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


    def fine_tune(self, train_df: pandas.DataFrame, task: dict, **kwargs) -> TabularModel:
        device = rima_pfn.TORCH_DEVICE
        torch.manual_seed(rima_pfn.random_state())

        if self.class_indices is None:
            self.classes_mapping(train_df, task)

        self.model = TabICLv2(
            task="classification" if task["task"] == Task.CLASSIFICATION else "regression",
            pretrained=True,
            device="cpu"
        )

        full = train_df.reset_index(drop=True)
        validation_index, train_index = TabularModel.train_val_split(train_df.shape[0])
        stypes = sdm.infer_stypes(full, overrides=self.overrides(task["label"], task["task"]))

        table = sdm.TableTensor.from_pandas(full, stypes=stypes, device=device)
        train_size = len(train_index)
        context_size = min(kwargs["context_size"], int(0.75 * train_size))
        query_size = min(kwargs["context_size"], train_size - context_size)
        context_evaluation_index = train_index[
            torch.randperm(
                train_size,
                generator=rima_pfn.TORCH_GENERATOR)[:kwargs["eval_context_size"]
            ].to(device)
        ]
        eval_kwargs = dict(
            table=table,
            label=task["label"],
            context_index=context_evaluation_index,
            validation_index=validation_index,
            task=task["type"],
            y_scale=float(table.select_columns(task["label"]).numerical[train_index].float().std().clamp_min(1e-8)) if task["type"] == "regression" else 1.0,
            num_estimators=kwargs["num_estimators_eval"],
            batch_size=kwargs["eval_batch_size"],
            amp_contextmanager=TabularModel.amp_contextmanager,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=kwargs["learning_rate"],
            weight_decay=kwargs["weight_decay"]
        )
        validation_loss, validation_metric = validate_model(self.model, **eval_kwargs)
        best_loss, best_state, bad_epochs = validation_loss, self.model_snapshot(), 0
        total_steps = kwargs["epochs"] * kwargs["steps_per_epoch"]
        warmup = max(1, int(kwargs["warmup_fraction"] * total_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            partial(TabularModel.cosine_scheduler, warmup=warmup, total_steps=total_steps)
        )

        # train
        for epoch in tqdm(range(1, kwargs["epochs"] + 1), total=kwargs["epochs"]):
            self.model.train()  # forward() only tracks gradients in train mode
            running = 0.0
            for _ in range(kwargs["steps_per_epoch"]):
                perm = train_index[torch.randperm(train_size, generator=rima_pfn.TORCH_GENERATOR).to(device)]
                context, query = table[perm[:context_size]], table[perm[context_size: context_size + query_size]]

                with TabularModel.amp_contextmanager():
                    predictions = self.model(
                        x_context=context.drop_columns(task["label"]),
                        y_context=context.select_columns(task["label"]),
                        x_query=query.drop_columns(task["label"]),
                        num_estimators=kwargs["num_estimators_train"],
                    )
                loss = task_loss(
                    task["type"],
                    predictions,
                    query.select_columns(task["label"]),
                    float(table.select_columns(task["label"]).numerical[train_index].float().std().clamp_min(1e-8)) if task["type"] == "regression" else 1.0
                )[0]

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), kwargs["grad_clip"])
                optimizer.step()
                scheduler.step()
                running += loss.item()

            validation_loss, validation_metric = validate_model(self.model, **eval_kwargs)

            has_improved = validation_loss < best_loss
            if has_improved:
                best_loss, best_state, bad_epochs = validation_loss, self.model_snapshot(), 0
            else:
                bad_epochs += 1
            if bad_epochs >= kwargs["patience"]:
                break

        self.model.load_state_dict(best_state)
        self.model.eval()


    def fit_pfn(self, data: pandas.DataFrame, label: str) -> TabularModel:
        """Implementation w/ the Priorlabs library: https://github.com/PriorLabs/TabPFN"""
        model = TabPFNClassifier(device="cpu")
        model.fit(
            torch.Tensor(data.drop(columns=[label]).values),
            torch.Tensor(data[label])
        )

        self.model = model
        self.is_fit = True

        return self


class ICLTabularModel(TabularModel):
    def __init__(self):
        super().__init__()
        self.model = TabICLv2(device="cpu")

    def query(
        self,
        data: pandas.DataFrame,
        label: str,
        task: Task,
        context: pandas.DataFrame | Sequence[pandas.DataFrame] | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Predict, ICL-style.

        Args:
            data:
            label:
            task: One of Task
            context: The context for the ICL inference. One of:
                - a dataframe: the same context is used for every row in `data`.
                - a sequence of dataframes: One for each row in `data`. To each row, its respective context.
                - a sequence of indexes (accessed with context.iloc): Indexes for the context of each row in `data`.
                It is **always assumed** that context share the same header

        Returns:
            The predictions.
        """
        data_types = sdm.infer_stypes(data, overrides=self.overrides(label, task))
        data_table = TableTensor.from_pandas(df=data, stypes=data_types)

        if isinstance(context, pandas.DataFrame):
            # Use full context for everything
            context_types = sdm.infer_stypes(context, overrides=self.overrides(label, task))
            context_table = TableTensor.from_pandas(df=context, stypes=context_types)

            query = data_table.drop_columns(label)
            context = context_table.drop_columns(label)
            context_labels = context_table[:, label]
            predictions = self.model(
                x_context=context,
                x_query=query,
                y_context=context_labels,
            )

        elif isinstance(context, list):
            # Context for each dataframe is already given
            is_index = isinstance(context[0], list)

            predictions = list()
            for i, row in data.iterrows():
                if is_index:
                    # index
                    context_table = TableTensor.from_pandas(
                        df=context.iloc[i],
                        stypes=sdm.infer_stypes(context.iloc[i], overrides=self.overrides(label, task)),
                        device="cpu",
                    )
                else:
                    # dataframe
                    context_table = TableTensor.from_pandas(
                        df=context[i],
                        stypes=sdm.infer_stypes(context[i], overrides=self.overrides(label, task)),
                        device="cpu",
                    )

                query = data_table[i : i + 1].drop_columns(label)
                context = context_table.drop_columns(label)
                labels = data_table[i : i + 1, label]
                predictions.append(
                    self.model(
                        x_context=context,
                        x_query=query,
                        y_context=labels,
                    )
                )

            predictions = torch.Tensor(predictions)

        else:
            raise ValueError(f"Unsupported context type: {type(context)}")

        return predictions
