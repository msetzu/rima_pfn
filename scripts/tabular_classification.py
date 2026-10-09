import json
import logging
from functools import partial

import pandas
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from rima_pfn.data import RIMADataset
from rima_pfn.modeling import context
from rima_pfn.modeling.tabular import TabularModel, ICLTabularModel
from rima_pfn.modeling.tasks import Task


with open("configs.json", "r") as f:
    CONFIGS = json.load(f)


def _preprocess(d: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    df = d.select_dtypes(include="number").drop(columns=[
        "total_tracks",
        "album_popularity",
        "birth_lat",
        # "birth_long",
    ])
    df = pandas.concat([df, d[["birth_region"]]], axis="columns")
    df = df[~df["birth_region"].isna()]

    # 10-only
    if CONFIGS["school"]["10_only"]:
        df = df[~df["birth_region"].isin(("Puglia", "Emilia-Romagna", "Calabria"))]

    train_df, test_df = train_test_split(
        df,
        stratify=df["birth_region"],
        test_size=0.2,
    )

    return train_df, test_df


def fit_school(d: pandas.DataFrame) -> dict:
    train_df, test_df = _preprocess(d)

    print("\tConstruct model...")
    model = TabularModel()
    print("\tFit...")
    model = model.fit_sdm(train_df, "birth_region", task=Task.CLASSIFICATION).model
    print("\tInference...")
    predictions = model.predict(test_df)
    print("\tEvaluation...")
    report = classification_report(
        y_true=test_df["birth_region"],
        y_pred=predictions,
        output_dict=True,
    )["macro_avg"]
    report["model"] = "TabICLv2"
    report["task"] = "fit_classification"
    report["task_name"] = "region"
    report["modality"] = "tabular"

    return report


def zero_shot_school(d: pandas.DataFrame) -> dict:
    train_df, test_df = _preprocess(d)

    model = TabularModel().fit_sdm(train_df, "birth_region", task=Task.CLASSIFICATION).model
    predictions = model.predict(test_df)
    report = classification_report(
        y_true=test_df["birth_region"],
        y_pred=predictions,
        output_dict=True,
    )["macro avg"]
    report["model"] = "TabICLv2"
    report["task"] = "zeroshot_classification"
    report["task_name"] = "region"
    report["modality"] = "tabular"

    return report


def icl(d: pandas.DataFrame) -> list[dict]:
    train_df, test_df = _preprocess(d)
    icl_model = ICLTabularModel()

    reports = list()
    for context_selection in CONFIGS["icl"]:
        if isinstance(context_selection, str):
            query = test_df
            context_set = train_df
            predictions = icl_model.query(
                data=query,
                label="region",
                task=Task.CLASSIFICATION,
                context=context_set,
            )

        else:
            query = test_df
            context_set = train_df

            context_selection_policy = context_selection["selection_policy"]
            context_selection_function = getattr(context, context_selection_policy)
            context_selection_function = partial(context_selection_function, data=context_set,
                                                 **context_selection["kwargs"])
            contexts_per_row = query[context_selection["object_id"]].apply(context_selection_function)

            predictions = icl_model.query(
                data=query,
                label="region",
                task=Task.CLASSIFICATION,
                context=contexts_per_row,
            )

        report = classification_report(
            y_true=test_df["birth_region"],
            y_pred=predictions,
            output_dict=True,
        )["macro_avg"]
        report["model"] = "TabICLv2"
        report["task"] = "icl_classification"
        report["task_name"] = "region"
        report["modality"] = "tabular"
        report["selection_policy"] = context_selection_policy
        report["object_id"] = context_selection["object_id"]
        report.update(context_selection["kwargs"])

        reports.append(report)

    return reports


def run():
    d = RIMADataset.build().d

    task_reports = list()
    ###########################################################
    # region/school classification ############################
    ###########################################################

    #############################
    # Fit #######################
    #############################
    print("School classification")
    task_reports.append(fit_school(d))

    #############################
    # zero-shot #################
    #############################
    task_reports.append(zero_shot_school(d))

    #############################
    # ICL #######################
    #############################
    # task_reports.append(icl(d))

    with open("../data/reports/classification.json", "w") as f:
        json.dump(task_reports, f)


if __name__ == "__main__":
    run()
