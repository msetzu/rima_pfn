from typing import Optional

import pandas
import numpy
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from rima_pfn.data import RIMADataset
from rima_pfn.generation.synthetic import TabularGenerator


def _maximum_mean_discrepancy(original_data: pandas.DataFrame, synthetic_data: pandas.DataFrame) -> float:
  numeric_features = original_data.select_dtypes(include="number").columns

  original_values = original_data[numeric_features].values
  synthetic_values = synthetic_data[numeric_features].values
  scaler = StandardScaler().fit(original_values)
  scaled_original_data = scaler.transform(original_values)
  scaled_synthetic_data = scaler.transform(synthetic_values)

  gamma = 1.0 / scaled_original_data.shape[1]
  kernel_original_to_original = rbf_kernel(scaled_original_data, scaled_original_data, gamma=gamma)
  kernel_synthetic_to_synthetic = rbf_kernel(scaled_synthetic_data, scaled_synthetic_data, gamma=gamma)
  kernel_original_to_synthetic = rbf_kernel(scaled_original_data, scaled_synthetic_data, gamma=gamma)

  squared_mean_discrepancy = numpy.mean(kernel_original_to_original)\
                            + numpy.mean(kernel_synthetic_to_synthetic)\
                            - 2 * numpy.mean(kernel_original_to_synthetic)
  maximum_mean_discrepancy = numpy.sqrt(numpy.maximum(squared_mean_discrepancy, 0)).item()

  return maximum_mean_discrepancy


def _wasserstein(original_data: pandas.DataFrame, synthetic_data: pandas.DataFrame) -> numpy.ndarray:
    numeric_features = original_data.select_dtypes(include="number").columns

    return numpy.array([
        wasserstein_distance(
            original_data[feature].values,
            synthetic_data[feature].values
        )
        for feature in numeric_features
    ])


def _kl_divergence(original_data: pandas.DataFrame, synthetic_data: pandas.DataFrame) -> numpy.ndarray:
    numeric_features = original_data.select_dtypes(include="number").columns

    return numpy.array([
        numpy.sum(
            original_data[feature].values *\
            numpy.log2(original_data[feature].values / synthetic_data[feature].values)
        ).statistic
        for feature in numeric_features
    ])


def _kolmogorov_smirnov_test(original_data: pandas.DataFrame, synthetic_data: pandas.DataFrame) -> numpy.ndarray:
    numeric_features = original_data.select_dtypes(include="number").columns

    return numpy.array([
        ks_2samp(
            original_data[feature].values,
            synthetic_data[feature].values
        ).statistic
        for feature in numeric_features
    ])


def generate(n: Optional[int] = None) -> tuple[pandas.DataFrame, dict]:
    d = RIMADataset.build()
    generator = TabularGenerator(d)

    if n is None:
        n = d.d.shape[0]

    generated = generator.generate(n)
    evaluation = evaluate_generation(d.d, generated)

    return generated, evaluation

def evaluate_generation(original_data: pandas.DataFrame, synthetic_data: pandas.DataFrame) -> dict:
    metrics = {
        "wasserstein": _wasserstein(original_data, synthetic_data),
        "kl_divergence": _kl_divergence(original_data, synthetic_data),
        "kolmogorov_smirnov": _kolmogorov_smirnov_test(original_data, synthetic_data),
        "maximum_mean_discrepancy": _maximum_mean_discrepancy(original_data, synthetic_data),
    }
    metrics["min_wasserstein"] = min(metrics["wasserstein"])
    metrics["max_wasserstein"] = max(metrics["wasserstein"])
    metrics["mean_wasserstein"] = metrics["wasserstein"].mean()

    metrics["min_kl_divergence"] = min(metrics["kl_divergence"])
    metrics["max_kl_divergence"] = max(metrics["kl_divergence"])
    metrics["mean_kl_divergence"] = metrics["kl_divergence"].mean()

    metrics["min_kolmogorov_smirnov"] = min(metrics["kolmogorov_smirnov"])
    metrics["max_kolmogorov_smirnov"] = max(metrics["kolmogorov_smirnov"])
    metrics["mean_kolmogorov_smirnov"] = metrics["kolmogorov_smirnov"].mean()

    return metrics
