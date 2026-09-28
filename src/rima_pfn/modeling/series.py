from __future__ import annotations

from typing import Callable, Optional

import numpy
import torch
from tabpfn import TabPFNClassifier


class SeriesModel:
    def __init__(self):
        self.model = None
        self.is_fit = False


    def encode(data: torch.Tensor) -> torch.Tensor:
        features = []
        for sample in data:
            sample_features = [
                numpy.mean(sample),
                numpy.std(sample),
                numpy.min(sample),
                numpy.max(sample),
                numpy.median(sample),
                numpy.percentile(sample, 25),
                numpy.percentile(sample, 75),
                # Trend / slope feature
                numpy.polyfit(numpy.arange(len(sample)), sample, 1)[0],
            ]
            features.append(sample_features)

        return torch.Tensor(features)


    def fit(self, data: torch.Tensor, labels: torch.Tensor, encoder: Optional[Callable]) -> SeriesModel:
        if encoder is None:
            data = self.encode(data)
        else:
            data = encoder(data)

        model = TabPFNClassifier(device="auto",)
        model.fit(data, labels)

        self.model = model
        self.is_fit = True

        return self
