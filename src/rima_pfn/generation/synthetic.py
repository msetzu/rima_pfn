from typing import Callable, Sequence, Optional

import pandas
import torch
from tabpfn.constants import ModelVersion
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised.experiments import GenerateSyntheticDataExperiment

from rima_pfn.data import RIMADataset


class RIMAGenerator:
    pass


class TabularGenerator:
    """Generates tabular data for RIMA. Focused on categorical, boolean and short text data."""
    features = [
    ]

    def __init__(self, dataset: RIMADataset):
        self.d = dataset.d[TabularGenerator.features]
        # default version V2 to avoid pricey PFNv3 license
        # support me on my patreon and subscribe for more poor bastard protips
        self.model = TabPFNUnsupervisedModel(
            tabpfn_clf=TabPFNClassifier.create_default_for_version(ModelVersion.V2),
            tabpfn_reg=TabPFNRegressor().create_default_for_version(ModelVersion.V2)
        )
        self.generator = GenerateSyntheticDataExperiment(task_type="unsupervised")



    def generate(self, n: int, **generation_parameters) -> pandas.DataFrame:
        """Generate `n` samples.

        Args:
            n: How many samples to generate
            **generation_parameters: Keyword parameters for the generator:
                - temp: Generation temperature

        Returns:
            The generated data, as a DataFrame.
        """
        generated = self.generator.run(
            tabpfn=self.model,
            n_samples=n,
            X=torch.Tensor(self.d.values),
            indices=torch.arange(len(TabularGenerator.features))  # generate all features
            **generation_parameters,
        ).synthetic_X.numpy()

        generated = pandas.DataFrame(generated, columns=TabularGenerator.features)

        return generated
