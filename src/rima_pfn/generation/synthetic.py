from typing import Optional, Sequence

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
        "total_tracks",
        "album_popularity",
        "birth_lat",
        "birth_lon",
        "release_year",
        "release_month",
        "release_day",
        "stats_pageviews",
        "bpm",
        "beats_count",
        "beat_loudness",
        "danceability",
        "dynamic_complexity",
        "average_loudness",
        "silence_rate_20db",
        "loudness",
        "centroid_mean",
        "centroid_std",
        "centroid_min",
        "centroid_max",
        "rolloff_mean",
        "rolloff_std",
        "rolloff_min",
        "rolloff_max",
        "flux_mean",
        "flux_std",
        "flux_min",
        "flux_max",
        "onset_novelty_mean",
        "onset_novelty_std",
        "onset_novelty_min",
        "onset_novelty_max",
        "rms_mean",
        "rms_std",
        "rms_min",
        "rms_max",
        "zcr_mean",
        "zcr_std",
        "zcr_min",
        "zcr_max",
        "flatness_mean",
        "flatness_std",
        "flatness_min",
        "flatness_max",
        "spectral_complexity_mean",
        "spectral_complexity_std",
        "spectral_complexity_min",
        "spectral_complexity_max",
        "spectral_energy_mean",
        "spectral_energy_std",
        "spectral_energy_min",
        "spectral_energy_max",
        "pitch_mean",
        "pitch_std",
        "pitch_min",
        "pitch_max",
        "spectral_contrast_band0_mean",
        "spectral_contrast_band0_std",
        "spectral_contrast_band0_min",
        "spectral_contrast_band0_max",
        "spectral_contrast_band1_mean",
        "spectral_contrast_band1_std",
        "spectral_contrast_band1_min",
        "spectral_contrast_band1_max",
        "spectral_contrast_band2_mean",
        "spectral_contrast_band2_std",
        "spectral_contrast_band2_min",
        "spectral_contrast_band2_max",
        "spectral_contrast_band3_mean",
        "spectral_contrast_band3_std",
        "spectral_contrast_band3_min",
        "spectral_contrast_band3_max",
        "spectral_contrast_band4_mean",
        "spectral_contrast_band4_std",
        "spectral_contrast_band4_min",
        "spectral_contrast_band4_max",
        "spectral_contrast_band5_mean",
        "spectral_contrast_band5_std",
        "spectral_contrast_band5_min",
        "spectral_contrast_band5_max",
        "mfcc_0_mean",
        "mfcc_0_std",
        "mfcc_1_mean",
        "mfcc_1_std",
        "mfcc_2_mean",
        "mfcc_2_std",
        "mfcc_3_mean",
        "mfcc_3_std",
        "mfcc_4_mean",
        "mfcc_4_std",
        "mfcc_5_mean",
        "mfcc_5_std",
        "mfcc_6_mean",
        "mfcc_6_std",
        "mfcc_7_mean",
        "mfcc_7_std",
        "mfcc_8_mean",
        "mfcc_8_std",
        "mfcc_9_mean",
        "mfcc_9_std",
        "mfcc_10_mean",
        "mfcc_10_std",
        "mfcc_11_mean",
        "mfcc_11_std",
        "mfcc_12_mean",
        "mfcc_12_std",
        "hpcp_0_mean",
        "hpcp_1_mean",
        "hpcp_2_mean",
        "hpcp_3_mean",
        "hpcp_4_mean",
        "hpcp_5_mean",
        "hpcp_6_mean",
        "hpcp_7_mean",
        "hpcp_8_mean",
        "hpcp_9_mean",
        "hpcp_10_mean",
        "hpcp_11_mean",
        "is_male",
        "is_female",
        "is_group",
        "is_album",
        "is_single",
        "is_compilation"
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


    def generate(self, n: int, features: Optional[Sequence[str]] = None, **generation_parameters) -> pandas.DataFrame:
        """Generate `n` samples.

        Args:
            n: How many samples to generate
            **generation_parameters: Keyword parameters for the generator:
                - temp: Generation temperature

        Returns:
            The generated data, as a DataFrame.
        """
        if features is not None:
            features = features
        else:
            features = TabularGenerator.features

        generated = self.generator.run(
            tabpfn=self.model,
            n_samples=n,
            X=torch.Tensor(self.d[features].values),
            # todo: do we want labels as well? why keep them separate?
            indices=torch.arange(len(features))  # generate all features
            **generation_parameters,
        ).synthetic_X.numpy()

        generated = pandas.DataFrame(generated, columns=features)

        return generated
