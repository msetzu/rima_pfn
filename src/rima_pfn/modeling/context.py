from typing import Optional

import numpy
import pandas
import rima_pfn

from rima_pfn.data import RIMADataset


def same(
    dataset: RIMADataset | pandas.DataFrame,
    row: pandas.Series,
    selection_columns: list[str],
    samples_nr: Optional[int] = None,
) -> pandas.DataFrame:
    """Select a subset of `dataset` with the same values of `row` in columns `selection_columns`"""
    if isinstance(dataset, RIMADataset):
        dataset = dataset.d.to_pandas()

    selection = dataset[
        dataset[selection_columns] == row[selection_columns]
    ]
    if samples_nr is None:
        return selection
    else:
        return selection.sample(samples_nr, random_state=rima_pfn.random_state())


def almost_same(
    dataset: RIMADataset | pandas.DataFrame,
    row: pandas.Series,
    selection_columns: dict[str, float],
    samples_nr: Optional[int] = None,
) -> pandas.DataFrame:
    """Select a subset of `dataset` with the same values of `row` in columns `selection_columns`"""
    if isinstance(dataset, RIMADataset):
        dataset = dataset.d.to_pandas()

    index_filter = [numpy.array([True] * dataset.shape[0])]
    for feature, tolerance  in selection_columns:
        index_filter.append(
            dataset[feature].between(row[feature] - tolerance, row[feature] + tolerance).to_numpy()
        )
    index_filter = numpy.array(index_filter).all(axis=0)

    if samples_nr is None:
        return dataset.loc[index_filter]
    else:
        return dataset.loc[index_filter].sample(samples_nr, random_state=rima_pfn.random_state())
