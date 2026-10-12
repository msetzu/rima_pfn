from __future__ import annotations

from typing import Sequence

import numpy
import pandas
import sdm
import torch
from sdm import TableTensor
from sdm.models import TabICLv2
from tabpfn import TabPFNClassifier
from torch import nn
from torch.optim import AdamW

from rima_pfn.modeling.tasks import Task


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
        inference_data = data.drop(columns=[label])
        table = TableTensor.from_pandas(
            df=inference_data,
            stypes=sdm.infer_stypes(inference_data, overrides=overrides),
            device="cpu",
        )

        # also predictions are tabletensors: convert to numpy
        table_predictions = self.model.predict(table, **kwargs).to_pandas()
        predictions = table_predictions.idxmax(axis=1).to_numpy()

        return predictions


    def predict_torch(self, context_df, query_df, task):
        unique_classes = sorted(context_df[task["label"]].unique())
        self.class_indices = {class_name: i for i, class_name in enumerate(unique_classes)}
        self.index_classes = {i: class_name for i, class_name in enumerate(unique_classes)}

        context, query = context_df.copy(), query_df.copy()
        context[task["label"]] = context[task["label"]].map(self.class_indices)
        query[task["label"]] = query[task["label"]].map(self.class_indices)

        overrides = {task["label"]: "categorical"} if task == Task.CLASSIFICATION else {}
        types = sdm.infer_stypes(context, overrides=overrides)
        context_table = TableTensor.from_pandas(
            df=context,
            stypes=types,
            device="cpu",
        )
        query_table = TableTensor.from_pandas(
            df=query.drop(columns=[task["label"]]),
            stypes=types,
            device="cpu",
        )

        logits = self.model(
            x_context=context_table.drop_columns([task["label"]]),
            y_context=context_table[:, task["label"]],
            x_query=query_table,
            num_estimators=16,
        )

        probabilities = torch.softmax(logits, dim=-1)
        predictions_index = torch.argmax(probabilities, dim=-1).cpu().numpy()
        predictions = numpy.array([self.index_classes[i] for i in predictions_index])

        return predictions

    def fine_tune(self, train_df: pandas.DataFrame, label: str, task: Task, **kwargs) -> TabularModel:
        self.model.train()

        overrides = self.overrides(label, task)
        train_table = TableTensor.from_pandas(
            df=train_df,
            stypes=sdm.infer_stypes(train_df, overrides=overrides),
            device="cpu",
        )

        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        epochs = kwargs.get("epochs", 5)
        episodes_per_epoch = kwargs.get("episodes_per_epoch", 100)
        support_size = kwargs.get("support_size", 64)
        query_size = kwargs.get("query_size", 32)
        n_train = len(train_df)

        print("Starting Episodic Fine-Tuning on TabICLv2...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            for step in range(episodes_per_epoch):
                optimizer.zero_grad()

                perm = torch.randperm(n_train, device="cpu")
                support_idx = perm[:support_size]
                query_idx = perm[support_size: support_size + query_size]

                x_support = train_table[support_idx].drop_columns(label)
                y_support = train_table[support_idx, label]

                x_query = train_table[query_idx].drop_columns(label)
                y_query = train_table[query_idx, label]

                logits = self.model(
                    x_context=x_support,
                    y_context=y_support,
                    x_query=x_query,
                    num_estimators=1,
                )

                loss = criterion(logits, y_query)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / episodes_per_epoch
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

            # Save model weights at the end of each epoch
            save_filename = f"{task['label']}_tabicl_epoch_{epoch + 1}.pt"
            torch.save(self.model.state_dict(), save_filename)
            print(f"\tModel saved to {save_filename}")

        return self

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
