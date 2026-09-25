import pathlib
from typing import Optional

import librosa
import numpy
import pandas
from datasets import load_dataset


CONFIGS = ("albums", "artists", "audio", "edges", "lyrics", "nodes")
SR = 16_000

class RIMADataset:
    def __init__(self, rima_dict: dict, audio_path: Optional[pathlib.Path] = None):
        merge = pandas.merge(
            rima_dict["albums"],
            rima_dict["artists"],
            left_on="artist_ids",
            right_on="id_artist",
            suffixes=("_album", "_artist"),
        )
        merge = pandas.merge(
            merge,
            rima_dict["audio"],
            left_on="name",
            right_on="name_artist",
            suffixes=("_merge1", "_audio"),
        )

        merge["is_male"] = merge["gender"] == "M"
        merge["is_female"] = merge["gender"] == "F"
        merge["is_group"] = merge["gender"] == "G"
        merge["is_album"] = merge["album_type"] == "album"
        merge["is_single"] = merge["album_type"] == "single"
        merge["is_compilation"] = merge["album_type"] == "compilation"

        merge.drop([
            "id_album_spotify",
            "artist_ids",
            "artist_names",
            "upc",
            "mbid_release",
            "release_country",
            "barcode",
            "qid",
            "image_url",
            "mbid",
            "mb_type",
            "mb_area",
            "mb_begin_area",
            "mb_aliases",
            "id_track_genius",
            "id_artist_genius",
            "name_artist_genius",
            "featured_artists",
            "primary_artist",
            "language",
            "lyrics",
            "url",
            "lang_langdetect",
            "lang_langdetect_prob",
            "lang_langid",
            "lang_langid_score",
            "lang_lingua",
            "lang_lingua_conf",
            "id_artist_audio",
            "__index_level_0__",
            "album_image",
            "active_end",
            ],
            axis="columns",
            inplace=True,
        )

        merge.rename({
            "id_artist_merge1": "artist_id",
            "id_album": "album_id",
            "name": "artist_name",
            "id_track": "track_id",
            },
            axis="columns",
            inplace=True,
        )

        if audio_path is not None:
            artists, tracks = merge.id_artist, merge.id_track
            merge["audio"] = [audio_path / f"{a}_{t}'.mp3" for a, t in zip(artists, tracks)]
            merge["audio"] = merge.audio.apply(
                lambda p: librosa.load(p, sr=SR)[0] if p.exists() else numpy.array([])
            )

        self.d = merge


def load_data() -> RIMADataset:
    rima_dict = {
        config: load_dataset("mstz/rima", config)["train"].to_pandas()
        for config in CONFIGS
    }

    return RIMADataset(rima_dict)

if __name__ == "__main__":
    res = load_data()
    pass