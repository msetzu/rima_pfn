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
        merge = pandas.merge(
            merge,
            rima_dict["albums"][["album_name", "album_type"]],
            left_on="album_audio",
            right_on="album_name",
            how="left",
            # suffixes=("_audio", "_artist"),
        )

        merge.drop(columns=["id_artist", "id_artist_audio"], inplace=True)
        merge["is_male"] = merge["gender"] == "M"
        merge["is_female"] = merge["gender"] == "F"
        merge["is_group"] = merge["gender"] == "G"
        # preclean for downstream tasks
        merge = merge[(~merge["birth_lon"].isna()) & (~merge["birth_lat"].isna())]

        merge["era"] = merge["release_year_audio"].apply(
            lambda x: "early_rap" if x < 2005 else "golden_era" if x < 2021 else "trap"
        )

        # has featuring?
        merge["has_featuring"] = merge["featured_artists_audio"].apply(
            lambda x: not (x is None)
        )

        merge.drop([
            "__index_level_0__",
            "__index_level_0___audio",
            "__index_level_0___lyrics",
            "active_end",
            "active_start",
            "album",
            "album_audio",
            "album_image",
            "album_lyrics",
            "artist_ids",
            "artist_names",
            "barcode",
            "birth_country",
            "birth_date",
            "birth_place",
            "description",
            "featured_artists",
            "featured_artists_audio",
            "featured_artists_lyrics",
            "id_album_spotify",
            "id_artist_audio",
            "id_artist_genius",
            "id_artist_genius_audio",
            "id_track_genius",
            "id_track_genius_audio",
            "image_url",
            "lang_langdetect",
            "lang_langdetect_prob",
            "lang_langdetect_prob_audio",
            "lang_langdetect_prob_lyrics",
            "lang_langid",
            "lang_langid_score",
            "lang_langid_score_audio",
            "lang_langid_score_lyrics",
            "lang_lingua",
            "lang_lingua_conf",
            "lang_lingua_conf_audio",
            "lang_lingua_conf_lyrics",
            "language",
            "lyrics",
            "lyrics_clean",
            "mb_aliases",
            "mb_area",
            "mb_begin_area",
            "mb_type",
            "mbid",
            "mbid_release",
            "music_genre",
            "name_artist_genius",
            "nationality",
            "primary_artist",
            "pseudonyms",
            "qid",
            "real_name",
            "record_label",
            "release_country",
            "release_day",
            "release_day_audio",
            "release_day_lyrics",
            "release_month",
            "release_month_audio",
            "release_month_lyrics",
            "release_type",
            "release_year",
            "release_year_audio",
            "release_year_lyrics",
            "stats_pageviews",
            "stats_pageviews_audio",
            "stats_pageviews_lyrics",
            "upc",
            "url",
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