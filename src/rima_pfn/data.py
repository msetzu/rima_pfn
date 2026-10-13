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
            rima_dict["audio"],
            rima_dict["lyrics"],
            left_on="id_track",
            right_on="id_track",
            how="inner",
            suffixes=("_audio", "_lyrics"),
        )
        merge = pandas.merge(
            merge,
            rima_dict["artists"],
            left_on="id_artist_audio",
            right_on="id_artist",
            how="inner",
            suffixes=("_audio", "_artist"),
        )

        merge.drop(columns=["id_artist", "id_artist_audio"], inplace=True)
        merge["is_male"] = merge["gender"] == "M"
        merge["is_female"] = merge["gender"] == "F"
        merge["is_group"] = merge["gender"] == "G"
        # preclean for downstream tasks
        merge = merge[(~merge["birth_lon"].isna()) & (~merge["birth_lat"].isna())]

        merge.drop([
            "id_album_spotify",
            "featured_artists_audio",
            "featured_artists_lyrics",
            "artist_ids",
            "artist_names",
            "upc",
            "mbid_release",
            "release_country",
            "release_type",
            "release_year",
            "release_month",
            "release_day",
            "album_audio",
            "album_lyrics",
            "release_year_lyrics",
            "release_month_lyrics",
            "release_day_lyrics",
            "release_year_audio",
            "release_month_audio",
            "release_day_audio",
            "real_name",
            "pseudonyms",
            "birth_date",
            "birth_place",
            "birth_country",
            "nationality",
            "active_start",
            "music_genre",
            "record_label",
            "description",
            "album",
            "lyrics_clean",
            "stats_pageviews",
            "stats_pageviews_audio",
            "stats_pageviews_lyrics",
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
            errors="ignore",
        )

        merge.rename(
            {
                "id_track": "track_id",
                "id_artist_lyrics": "artist_id",
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

    @staticmethod
    def build():
        rima_dict = {
            config: load_dataset("mstz/rima", config)["train"].to_pandas()
            for config in CONFIGS
        }

        return RIMADataset(rima_dict)


if __name__ == "__main__":
    res = RIMADataset.build()
    pass