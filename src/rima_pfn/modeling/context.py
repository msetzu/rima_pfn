import pandas

from rima_pfn.data import RIMADataset


def same(dataset: RIMADataset | pandas.DataFrame, object_id: str, same_id: str) -> pandas.DataFrame:
    """Subset of `dataset` with the same `same_id` but different `object_id`, e.g.,
    same artist_id, but different track_id"""
    if isinstance(dataset, RIMADataset):
        dataset = dataset.d.to_pandas()

    return dataset[
        (dataset[object_id] != object_id) &
        (dataset[same_id] == same_id)
    ]


def almost_same(dataset: RIMADataset | pandas.DataFrame, object_value: int | float, object_id: str, same_id: str, tol: float) -> pandas.DataFrame:
    """Subset of `dataset` with the same `same_id` (plus or minus `tol`) but different `object_id`, e.g.,
    same artist_id, but bpm +- `tol`."""
    if isinstance(dataset, RIMADataset):
        dataset = dataset.d.to_pandas()

    return dataset[
        (dataset[object_id].between(object_value - tol, object_value + tol)) &
        (dataset[same_id] == same_id)
    ]
