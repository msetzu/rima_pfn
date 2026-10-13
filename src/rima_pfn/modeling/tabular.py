from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import pandas
import sdm
import torch
from sdm import TableTensor
from sdm.models import TabICLv2
from sdm.evaluation import to_class_indices
from tabpfn import TabPFNClassifier
import torch.nn.functional as F

from rima_pfn.modeling.tasks import Task


def _task_loss(task: str, out, y, y_scale: float):
    """Loss and a human-readable metric for one batch of query predictions."""
    if task == "classification":
        # out: per-class probabilities, one column per class. to_class_indices
        # aligns the columns to the target's categories.
        scores, idx = to_class_indices(out, y, missing_score=0.0)
        loss = F.nll_loss(scores.float().clamp_min(1e-6).log(), idx)
        metric = (scores.argmax(-1) == idx).float().mean()  # accuracy
    else:
        q = out.numerical.float()  # [R, n_quantiles]
        t = y.numerical.float()  # [R, 1]
        levels = torch.arange(1, q.size(-1) + 1, device=q.device, dtype=q.dtype) / (q.size(-1) + 1)
        diff = t - q
        loss = torch.maximum(levels * diff, (levels - 1) * diff).mean() / y_scale  # pinball
        metric = ((q.mean(-1, keepdims=True) - t) ** 2).mean().sqrt()  # RMSE of the mean
    return loss, metric


@torch.no_grad()
def _evaluate(model, table, target, ctx_idx, val_idx, task, y_scale, recipe, num_estimators, batch_size, amp_ctx):
    """Validation via the cached fit()/predict() path (model must be in eval mode)."""
    model.eval()
    ctx = table[ctx_idx]
    with amp_ctx():
        model.fit(
            ctx.drop_columns(target),
            ctx.select_columns(target),
            recipe=recipe,
            num_estimators=num_estimators,
        )
    total_loss = total_metric = 0.0
    for chunk in val_idx.split(batch_size):
        q = table[chunk]
        with amp_ctx():
            out = model.predict(q.drop_columns(target))
        loss, metric = _task_loss(task, out, q.select_columns(target), y_scale)
        total_loss += loss.item() * len(chunk)
        total_metric += metric.item() * len(chunk)
    model.clear()
    n = len(val_idx)
    return total_loss / n, total_metric / n


def finetune_tabiclv2(
        df: pandas.DataFrame,
        target: str,
        task: str = "classification",  # "classification" or "regression"
        val_df: pandas.DataFrame | None = None,
        val_fraction: float = 0.2,
        device: str | None = None,
        # optimisation
        epochs: int = 30,
        steps_per_epoch: int = 20,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        warmup_fraction: float = 0.1,
        grad_clip: float = 1.0,
        patience: int = 5,
        # episode shape
        context_size: int = 1024,
        query_size: int = 512,
        num_estimators_train: int = 2,
        num_estimators_eval: int = 4,
        eval_context_size: int = 10_000,
        eval_batch_size: int = 2048,
        # misc
        recipe: sdm.processing.Recipe | None = None,
        amp: bool = True,
        output_path: str | Path | None = None,
        seed: int = 0,
        verbose: bool = True,
):
    """Fine-tune the pretrained TabICLv2 checkpoint on a single table.

    Args:
        df: training data including the `target` column. Features should be
            numerical/categorical (the default recipe does not handle
            datetime/text columns; pass a custom `recipe` for those).
        target: name of the target column.
        task: "classification" (categorical target) or "regression"
            (numerical target).
        val_df: optional held-out data for early stopping. If None,
            `val_fraction` of `df` is held out.
        epochs, steps_per_epoch: an "epoch" is `steps_per_epoch` episodes,
            followed by one validation pass.
        learning_rate: AdamW LR. Keep it small to preserve the pretrained prior.
        patience: epochs without validation improvement before stopping.
        context_size / query_size: rows per episode used as in-context
            examples / as supervised query rows.
        num_estimators_train / _eval: ensemble members (different recipe
            randomisations) per forward pass.
        recipe: custom pre/post-processing; None uses TabICLv2's default.
        amp: use bfloat16 autocast on CUDA.
        output_path: if given, the best state_dict is saved there.

    Returns:
        (model, history): the model in eval mode with the best-validation
        weights loaded, and a list of per-epoch dicts. Epoch 0 is the
        zero-shot baseline, so fine-tuning is only kept if it beats it.
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    def amp_ctx():
        return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda")

    # ---- data: one table so categorical codes are shared by train and val ----
    if val_df is not None:
        full = pandas.concat([df, val_df], ignore_index=True)
        n_train = len(df)
        train_idx = torch.arange(n_train)
        val_idx = torch.arange(n_train, len(full))
    else:
        full = df.reset_index(drop=True)
        perm = torch.randperm(len(full), generator=g)
        n_val = max(1, int(len(full) * val_fraction))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

    print(f"on a {task} task...")
    stype = "categorical" if task == "classification" else "numerical"
    stypes = sdm.infer_stypes(full, overrides={target: stype})
    print(f"types {stypes}")
    table = sdm.TableTensor.from_pandas(full, stypes=stypes, device=dev)
    train_idx, val_idx = train_idx.to(dev), val_idx.to(dev)
    n_train = len(train_idx)

    y_scale = 1.0
    if task == "regression":
        y_scale = float(table.select_columns(target).numerical[train_idx].float().std().clamp_min(1e-8))

    # ---- model ----
    model = TabICLv2(task=task, pretrained=True, device=dev)

    n_ctx = min(context_size, int(0.75 * n_train))
    n_q = min(query_size, n_train - n_ctx)
    eval_ctx_idx = train_idx[torch.randperm(n_train, generator=g)[:eval_context_size].to(dev)]
    eval_kwargs = dict(
        table=table, target=target, ctx_idx=eval_ctx_idx, val_idx=val_idx, task=task, y_scale=y_scale,
        recipe=recipe, num_estimators=num_estimators_eval, batch_size=eval_batch_size, amp_ctx=amp_ctx,
    )

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    val_loss, val_metric = _evaluate(model, **eval_kwargs)
    best_loss, best_state, bad_epochs = val_loss, snapshot(), 0
    if verbose:
        print(
            f"epoch  0 | zero-shot val_loss={val_loss:.4f} val_{'acc' if task == 'classification' else 'rmse'}={val_metric:.4f}")

    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = epochs * steps_per_epoch
    warmup = max(1, int(warmup_fraction * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    for epoch in range(1, epochs + 1):
        model.train()  # forward() only tracks gradients in train mode
        running = 0.0
        for _ in range(steps_per_epoch):
            perm = train_idx[torch.randperm(n_train, generator=g).to(dev)]
            ctx, qry = table[perm[:n_ctx]], table[perm[n_ctx: n_ctx + n_q]]

            with amp_ctx():
                out = model(
                    x_context=ctx.drop_columns(target),
                    y_context=ctx.select_columns(target),
                    x_query=qry.drop_columns(target),
                    recipe=recipe,
                    num_estimators=num_estimators_train,
                )
            loss, _ = _task_loss(task, out, qry.select_columns(target), y_scale)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            sched.step()
            running += loss.item()

        val_loss, val_metric = _evaluate(model, **eval_kwargs)

        improved = val_loss < best_loss
        if improved:
            best_loss, best_state, bad_epochs = val_loss, snapshot(), 0
        else:
            bad_epochs += 1
        if verbose:
            print(f"epoch {epoch:2d} | train_loss={running / steps_per_epoch:.4f} val_loss={val_loss:.4f} "
                  f"val_{'acc' if task == 'classification' else 'rmse'}={val_metric:.4f}{' *' if improved else ''}")
        if bad_epochs >= patience:
            if verbose:
                print(f"early stopping (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    model.eval()

    return model


def load_finetuned_tabiclv2(path: str | Path, task: str = "classification", device: str | None = None):
    """Rebuild a TabICLv2 and load fine-tuned weights saved by `finetune_tabiclv2`."""
    model = TabICLv2(task=task, pretrained=False, device=device)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


class TabularModel:
    def __init__(self):
        self.model = None
        self.is_fit = False

        # mapping from and to torch in finetuning
        self._class_indices = None
        self.index_classes = None

    def overrides(self, label: str, task: Task) -> dict:
        return {label: "categorical"} if task == Task.CLASSIFICATION else {}

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


    def predict_torch(self, context_df: pandas.DataFrame, query_df: pandas.DataFrame, task: dict):
        overrides = self.overrides(task["label"], task["task"])
        types = sdm.infer_stypes(context_df, overrides=overrides)
        # table = sdm.TableTensor.from_pandas(
        #     pandas.concat([context_df, query_df], axis="rows", ignore_index=True),
        #     stypes=types,
        #     device="cpu"
        # )
        query_table = sdm.TableTensor.from_pandas(
            query_df,
            stypes=sdm.infer_stypes(query_df, overrides=overrides),
            device="cpu"
        )
        # KV caching
        self.model.fit(
            query_table.drop_columns(task["label"]),
            query_table.select_columns(task["label"]),
            num_estimators=16
        )
        table_predictions = self.model.predict(query_table).to_pandas()
        predictions = table_predictions.idxmax(axis=1).to_numpy()

        return predictions
        # if self.index_classes is None and self.class_indices is None:
        #     unique_classes = sorted(context_df[task["label"]].unique())
        #     self.class_indices = {class_name: i for i, class_name in enumerate(unique_classes)}
        #     self.index_classes = {i: class_name for i, class_name in enumerate(unique_classes)}
        #
        # context, query = context_df.copy(), query_df.copy()
        # context[task["label"]] = context[task["label"]].map(self.class_indices)
        # query[task["label"]] = query[task["label"]].map(self.class_indices)
        #
        # overrides = {task["label"]: "categorical"} if task == Task.CLASSIFICATION else {}
        # types = sdm.infer_stypes(context, overrides=overrides)
        # context_table = TableTensor.from_pandas(
        #     df=context,
        #     stypes=types,
        #     device="cpu",
        # )
        # query_table = TableTensor.from_pandas(
        #     df=query.drop(columns=[task["label"]]),
        #     stypes=types,
        #     device="cpu",
        # )
        #
        # logits = self.model(
        #     x_context=context_table.drop_columns([task["label"]]),
        #     y_context=context_table[:, task["label"]],
        #     x_query=query_table,
        #     num_estimators=16,
        # )
        #
        # probabilities = torch.softmax(logits, dim=-1)
        # predictions_index = torch.argmax(probabilities, dim=-1).cpu().numpy()
        # predictions = numpy.array([self.index_classes[i] for i in predictions_index])
        #
        # return predictions

    def fine_tune(self, train_df: pandas.DataFrame, task: dict, **kwargs) -> TabularModel:
        unique_classes = sorted(train_df[task["label"]].unique())
        self.class_indices = {class_name: i for i, class_name in enumerate(unique_classes)}
        self.index_classes = {i: class_name for i, class_name in enumerate(unique_classes)}
        train_df[task["label"]] = train_df[task["label"]].map(self.class_indices)

        self.model = finetune_tabiclv2(
            train_df, target=task["label"], task="classification" if task["task"] == Task.CLASSIFICATION else "regression",
            epochs=1, steps_per_epoch=1,
        )

        # if self.model is None:
        #     self.model = TabICLv2(device="cpu")
        #
        # self.model.train()
        #
        # unique_classes = sorted(train_df[task["label"]].unique())
        # self.class_indices = {class_name: i for i, class_name in enumerate(unique_classes)}
        # self.index_classes = {i: class_name for i, class_name in enumerate(unique_classes)}
        #
        # train_df[task["label"]] = train_df[task["label"]].map(self.class_indices)
        #
        # overrides = self.overrides(label, task)
        # train_table = TableTensor.from_pandas(
        #     df=train_df,
        #     stypes=sdm.infer_stypes(train_df, overrides=overrides),
        #     device="cpu",
        # )
        #
        # epochs = kwargs.get("epochs", 5)
        # episodes_per_epoch = kwargs.get("episodes_per_epoch", 100)
        # support_size = kwargs.get("support_size", 64)
        # query_size = kwargs.get("query_size", 32)
        # n_train = len(train_df)
        #
        # y_train_tensor = torch.tensor(
        #     train_df[label].to_numpy(), device="cpu", dtype=torch.long
        # )
        # optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        # criterion = nn.CrossEntropyLoss()
        #
        # print("Starting Episodic Fine-Tuning on TabICLv2...")
        # for epoch in range(epochs):
        #     epoch_loss = 0.0
        #     for step in range(episodes_per_epoch):
        #         optimizer.zero_grad()
        #
        #         # Randomly partition training data into support context and query targets
        #         perm = torch.randperm(n_train, device="cpu")
        #         support_idx = perm[:support_size]
        #         query_idx = perm[support_size: support_size + query_size]
        #
        #         x_support = train_table[support_idx].drop_columns(label)
        #         y_support = train_table[support_idx, label]
        #
        #         x_query = train_table[query_idx].drop_columns(label)
        #
        #         # Native 1D torch.Tensor target (bypasses SDM wrappers)
        #         y_query = y_train_tensor[query_idx]
        #
        #         # Forward pass (returns SDM ColumnarTensor/TableTensor)
        #         logits = self.model(
        #             x_context=x_support,
        #             y_context=y_support,
        #             x_query=x_query,
        #             num_estimators=1,
        #         )
        #
        #         if hasattr(logits, "values"):
        #             logits = logits.values
        #         if hasattr(logits, "tensor"):
        #             logits = logits.tensor
        #
        #         # 4. Disable SDM's __torch_function__ to cast to base torch.Tensor
        #         with torch._C.DisableTorchFunction():
        #             raw_logits = logits.as_subclass(torch.Tensor)
        #
        #         loss = criterion(raw_logits, y_query)
        #         loss.backward()
        #
        #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        #         optimizer.step()
        #
        #         epoch_loss += loss.item()
        #
        #     avg_loss = epoch_loss / episodes_per_epoch
        #     print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")
        #
        #     # Save model weights
        #     save_filename = f"{task['label']}_tabicl_epoch_{epoch + 1}.pt"
        #     torch.save(self.model.state_dict(), save_filename)
        #     print(f"\tModel saved to {save_filename}")
        #
        # return self

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
