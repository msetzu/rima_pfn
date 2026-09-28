from __future__ import annotations

from typing import Callable, Optional, Sequence

import pandas
import sdm
import torch
from sdm import TableTensor
from sdm.models import TabICLv2
from tabpfn import TabPFNClassifier
from transformers.testing_utils import set_model_for_less_flaky_test

from rima_pfn.modeling.tasks import Task


class TabularModel:
    def __init__(self):
        self.model = None
        self.is_fit = False

    def _overrides(self, label: str, task: Task) -> dict:
        return {label: "categorical"} if task == Task.CLASSIFICATION else {}

    def fit_sdm(self, data: pandas.DataFrame, label: str, task: Task) -> TabularModel:
        """Implementation w/ the SDM library: https://github.com/NVIDIA/structured-data-models"""
        overrides = self._overrides(label, task)
        table = TableTensor.from_pandas(
            df=data,
            stypes=sdm.infer_stypes(data, overrides=overrides),
            device="cuda",
        )

        return self


    def fit_pfn(self, data: torch.Tensor, labels: torch.Tensor) -> TabularModel:
        """Implementation w/ the Priorlabs library: https://github.com/PriorLabs/TabPFN"""
        model = TabPFNClassifier(device="auto",)
        model.fit(data, labels)

        self.model = model
        self.is_fit = True

        return self


class ICLTabularModel(TabularModel):
    def __init__(self):
        super().__init__()
        self.model = TabICLv2(device="auto")

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
        data_types = sdm.infer_stypes(data, overrides=self._overrides(label, task))
        data_table = TableTensor.from_pandas(df=data, stypes=data_types)

        if isinstance(context, pandas.DataFrame):
            # Use full context for everything
            context_types = sdm.infer_stypes(context, overrides=self._overrides(label, task))
            context_table = TableTensor.from_pandas(df=context, stypes=context_types)

            query = data_table.drop_columns(label)
            context = context_table.drop_columns(label)
            labels = data_table[:, label]
            predictions = self.model(
                x_context=context,
                x_query=query,
                y_context=labels,
            )

        elif isinstance(context, list):
            # Context for each dataframe is already given
            # todo: assumes same header for every context, maybe adjust later?
            is_index = isinstance(context[0], list)

            predictions = list()
            for i, row in data.iterrows():
                if is_index:
                    # index
                    context_table = TableTensor.from_pandas(
                        df=context.iloc[i],
                        stypes=sdm.infer_stypes(context.iloc[i], overrides=self._overrides(label, task)),
                        device="auto",
                    )
                else:
                    # dataframe
                    context_table = TableTensor.from_pandas(
                        df=context[i],
                        stypes=sdm.infer_stypes(context[i], overrides=self._overrides(label, task)),
                        device="auto",
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
