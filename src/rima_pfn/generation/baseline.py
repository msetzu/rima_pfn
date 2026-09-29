import numpy
import pandas
from imblearn.over_sampling import SMOTE

from rima_pfn import RANDOM_STATE


def undersample(dataset: pandas.DataFrame, label: str) -> pandas.DataFrame:
    undersample_size = dataset[label].value_counts().min()
    dataset = dataset.groupby(label, group_keys=False)\
        .apply(lambda x: x.sample(n=undersample_size, random_state=RANDOM_STATE))\
        .reset_index(drop=True)

    return dataset


def oversample(dataset: pandas.DataFrame, label: str) -> pandas.DataFrame:
    oversample_size = dataset[label].value_counts().max()
    dataset = dataset.groupby(label, group_keys=False) \
        .apply(lambda x: x.sample(n=oversample_size, replace=True, random_state=RANDOM_STATE)) \
        .reset_index(drop=True)

    return dataset


def smote(dataset: pandas.DataFrame, label: str) -> pandas.DataFrame:
    smote = SMOTE(random_state=RANDOM_STATE)
    data, labels = smote.fit_resample(
        dataset.drop_columns(label).select_dtypes(include="number").values,
        dataset[label].values
    )

    return pandas.DataFrame(
        numpy.hstack(data, labels),
        columns=dataset.drop_columns(label).select_dtypes(include="number"),
    )
