import json
from functools import partial

import numpy
import pandas
import sdm
import torch
from sdm import TableTensor
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import rima_pfn
from rima_pfn.data import RIMADataset
from rima_pfn.modeling import context
from rima_pfn.modeling.tabular import TabularModel, ICLTabularModel
from rima_pfn.modeling.tasks import Task


def preprocess(d: pandas.DataFrame, config: dict, by_artist: bool = True) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    df = d.copy()
    df = df.dropna(axis=0, how="any")
    if "exclude_classes" in config:
        df = df[~df[config["label"]].isin(config["exclude_classes"])]
    if "exclude_features" in config:
        df = df.drop(columns=config["exclude_features"], errors="ignore")

    df = df.drop(columns=[config["label"]], errors="ignore")
    df = pandas.concat([df, d.iloc[df.index][[config["label"]]]], axis="columns")

    if by_artist:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(df, groups=df["artist_id"]))
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_df = train_df.drop(columns=["artist_id"], errors="ignore").select_dtypes(include="number")
        test_df = test_df.drop(columns=["artist_id"], errors="ignore").select_dtypes(include="number")

    else:
        df = df.select_dtypes(include="number")

        train_df, test_df = train_test_split(
            df,
            stratify=df[config["label"]],
            test_size=0.2,
            random_state=rima_pfn.random_state(),
        )

    train_df = pandas.concat([train_df, d.iloc[train_df.index][[config["label"]]]], axis="columns")
    test_df = pandas.concat([test_df, d.iloc[test_df.index][[config["label"]]]], axis="columns")

    return train_df, test_df


def evaluate(model, data, labels: numpy.ndarray, task: dict, as_torch: bool = False) -> dict:
    if as_torch:
        predictions = model.predict_torch(data, task["label"], task=task["task"])
        labels = numpy.array([model.class_indices[c] for c in labels])

    print("\tEvaluation...")
    report = classification_report(
        y_true=labels,
        y_pred=predictions,
        output_dict=True,
    )["macro avg"]
    if task["task"] == Task.CLASSIFICATION:
        report["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)

    return report


def fit_task(d: pandas.DataFrame, task) -> dict:
    train_df, test_df = preprocess(d, task)

    model = TabularModel()
    print("\tFit...")
    model.fit_sdm(train_df, task["label"], task=task["task"])
    print("\tInference...")
    report = evaluate(
        model,
        test_df.drop(columns=[task["label"]]),
        test_df[task["label"]],
        task,
        as_torch=False
    )
    report["model"] = "TabICLv2"
    report["task"] = "fit"
    report["task_name"] = task["label"]
    report["modality"] = "tabular"

    return report


def finetune_task(d: pandas.DataFrame, task, finetune_args: dict) -> dict:
    train_df, test_df = preprocess(d, task)

    model = TabularModel()
    print("\tFit...")
    model.fine_tune(train_df, task["label"], task=task["task"], **finetune_args)
    print("\tInference...")
    report = evaluate(
        model,
        test_df.drop(columns=[task["label"]]),
        test_df[task["label"]],
        task,
        as_torch=True
    )
    report["model"] = "TabICLv2"
    report["task"] = "fit"
    report["task_name"] = task["label"]
    report["modality"] = "tabular"

    return report


def icl(d: pandas.DataFrame, config, icl_config) -> list[dict]:
    train_df, test_df = preprocess(d, config)
    icl_model = ICLTabularModel()

    reports = list()
    for context_selection in icl_config:
        if isinstance(context_selection, str):
            query = test_df
            context_set = train_df
            predictions = icl_model.query(
                data=query,
                label=config["label"],
                task=config["task"],
                context=context_set,
            )

        else:
            query = test_df
            context_set = train_df

            context_selection_policy = context_selection["selection_policy"]
            # selection function indicated as string, retrieved w/ reflection
            context_selection_function = getattr(context, context_selection_policy)
            context_selection_function = partial(
                context_selection_function,
                data=context_set,
                **context_selection["kwargs"]
            )
            contexts_per_row = query[context_selection["object_id"]].apply(context_selection_function)

            predictions = icl_model.query(
                data=query,
                label=config["label"],
                task=config["task"],
                context=contexts_per_row,
            )

        report = classification_report(
            y_true=test_df[config["label"]],
            y_pred=predictions,
            output_dict=True,
        )["macro avg"]
        report["model"] = "TabICLv2"
        report["mode"] = "icl"
        report["task"] = "icl_classification"
        report["task_name"] = config["label"]
        report["modality"] = "tabular"
        report["selection_policy"] = context_selection_policy
        report["object_id"] = context_selection["object_id"]
        report.update(context_selection["kwargs"])

        reports.append(report)

    return reports


def run():
    with open("../scripts/configs.json", "r") as f:
        config = json.load(f)

    d = RIMADataset.build().d

    task_reports = list()

    for task_name, task in config["tasks"].items():
        print(task_name)
        config["task"] = Task.CLASSIFICATION if config["label"] == "classification" else Task.REGRESSION

        if "fit" in config["modes"]:
            task_reports.append(fit_task(d, config))
            pandas.DataFrame(task_reports).to_csv("classification_report.csv")

        if "fine_tune" in config:
            finetuning_config = config["fine_tune"]
            task_reports.append(finetune_task(d, task, finetuning_config))

        if "icl" in config["modes"]:
            for icl_config in config["modes"]["icl"]:
                task_reports.append(icl(d, config, icl_config))
                pandas.DataFrame(task_reports).to_csv("classification_report.csv")



# if __name__ == "__main__":
#     run()
