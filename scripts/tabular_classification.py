import os
import json
from functools import partial
from typing import Optional

import numpy
import pandas
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from xgboost import XGBClassifier

import rima_pfn
from rima_pfn.data import RIMADataset
from rima_pfn.modeling import context
from rima_pfn.modeling.tabular import TabularModel, ICLTabularModel
from rima_pfn.modeling.tasks import Task


def preprocess(
        d: pandas.DataFrame,
        task: dict,
        by_artist: bool = True,
        intersect_labels: bool = True,
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    df = d.copy()
    if "exclude_classes" in task:
        df = df[~df[task["label"]].isin(task["exclude_classes"])]
    if "exclude_features" in task:
        df = df.drop(columns=task["exclude_features"], errors="ignore")
    df = df.dropna(axis=0, how="any", subset=df.select_dtypes(include="number").columns)

    if by_artist:
        if "artist_id" not in df.columns:
            df = pandas.concat([ df, d.loc[df.index][["artist_id"]] ], axis="columns")

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
            stratify=df[task["label"]],
            test_size=0.2,
            random_state=rima_pfn.random_state(),
        )

    # we may accidentally drop the label column
    if task["label"] not in train_df.columns:
        train_df = pandas.concat([train_df, d.loc[train_df.index][[task["label"]]]], axis="columns")
        test_df = pandas.concat([test_df, d.loc[test_df.index][[task["label"]]]], axis="columns")

    if intersect_labels:
        legal_labels = numpy.intersect1d(
            train_df[task["label"]].values,
            test_df[task["label"]].values,
        )
        train_df = train_df[train_df[task["label"]].isin(legal_labels)]
        test_df = test_df[test_df[task["label"]].isin(legal_labels)]

    return train_df, test_df


def evaluate(
    model: TabularModel,
    task: dict,
    query: pandas.DataFrame,
    labels: numpy.ndarray,
    context: Optional[pandas.DataFrame] = None,
    is_finetuned: bool = False
) -> dict:
    if is_finetuned:
        if context is None:
            raise ValueError("context required for evaluation on is_finetuned==True")
        
        predictions = model.predict_with_fit(query_df=query, context_df=context, task=task, as_objects=False)
    else:
        predictions = model.predict(query, task["label"], task=task["task"])

    report = classification_report(
        y_true=labels,
        y_pred=predictions,
        output_dict=True,
    )["macro avg"]
    if task["task"] == Task.CLASSIFICATION:
        report["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)

    return report


def evaluate_sklearn(
    model,
    task: dict,
    query: pandas.DataFrame,
    labels: numpy.ndarray,
) -> dict:
    predictions = model.predict(query)

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
    model.zero_shot(train_df, task["label"], task=task["task"])
    report = evaluate(
        model,
        query=test_df.drop(columns=[task["label"]]),
        labels=test_df[task["label"]],
        task=task,
        is_finetuned=False
    )
    report["model"] = "TabICLv2"
    report["task"] = "zeroshot"
    report["task_name"] = task["label"]
    report["modality"] = "tabular"

    return report


def fit_xgboost(d: pandas.DataFrame, task) -> dict:
    train_df, test_df = preprocess(d, task)

    if train_df.dtypes[task["label"]].kind not in "iu":
        encoder = LabelEncoder().fit(pandas.concat([
            train_df[task["label"]],
            test_df[task["label"]]
        ], axis="rows",
        ))
        train_df[task["label"]] = encoder.transform(train_df[task["label"]])
        test_df[task["label"]] = encoder.transform(test_df[task["label"]])

    model = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=1)
    model.fit(train_df.drop(columns=task["label"]), train_df[task["label"]])
    report = evaluate_sklearn(
        model,
        query=test_df.drop(columns=[task["label"]]),
        labels=test_df[task["label"]],
        task=task,
    )
    report["model"] = "xgboost"
    report["task"] = "fit"
    report["task_name"] = task["label"]
    report["modality"] = "tabular"

    return report


def finetune_task(d: pandas.DataFrame, task, finetune_args: dict) -> dict:
    train_df, test_df = preprocess(d, task, by_artist=True, intersect_labels=True)

    model = TabularModel()
    full_df = pandas.concat([train_df, test_df], axis="rows", ignore_index=True)
    class_indices, _ = model.classes_mapping(full_df, task)
    train_df[task["label"]] = train_df[task["label"]].map(class_indices)
    test_df[task["label"]] = test_df[task["label"]].map(class_indices)
    model.fine_tune(train_df, task=task, **finetune_args)
    report = evaluate(
        model,
        query=test_df,
        context=train_df,
        labels=test_df[task["label"]],
        task=task,
        is_finetuned=True
    )
    report["model"] = "TabICLv2"
    report["task"] = "finetune"
    report["task_name"] = task["label"]
    report["modality"] = "tabular"

    return report


def icl_task(d: pandas.DataFrame, task, icl_task) -> tuple[dict, dict]:
    train_df, test_df = preprocess(d, task)

    # sklearn models don't like objects as classes, gotta encode them
    if train_df.dtypes[task["label"]].kind not in "iu":  # categorical label
        encoder = LabelEncoder().fit(pandas.concat([
            train_df[task["label"]],
            test_df[task["label"]]
        ], axis="rows",
        ))

        sklearn_train_df = train_df.copy()
        sklearn_test_df = test_df.copy()
        sklearn_train_df[task["label"]] = encoder.transform(sklearn_train_df[task["label"]])
        sklearn_test_df[task["label"]] = encoder.transform(sklearn_test_df[task["label"]])

        sklearn_context_set = sklearn_train_df
        sklearn_query = sklearn_test_df
    else:
        sklearn_context_set = train_df
        sklearn_query = test_df


    query = test_df
    context_set = train_df

    context_selection_policy = icl_task["selection_policy"]
    # selection function indicated as string, retrieved w/ reflection
    context_selection_function = getattr(context, context_selection_policy)
    context_selection_function = partial(
        context_selection_function,
        **icl_task["kwargs"]
    )

    icl_predictions = list()
    fit_predictions = list()
    sizes = list()
    for index in tqdm(query.index):
        local_context = context_selection_function(row=query.loc[index], dataset=context_set)
        local_query = query.loc[[index]]

        # icl
        model = TabularModel()
        model.zero_shot(local_context, task["label"], task=task["task"])
        icl_predictions.append(
            model.predict(
                local_query,
                task["label"], task=task["task"]
            ).item()
        )

        # fit
        local_context = context_selection_function(row=sklearn_query.loc[index], dataset=sklearn_context_set)
        local_query = sklearn_query.loc[[index]]

        model = XGBClassifier(n_estimators=2, max_depth=2)
        model.fit(local_context.drop(columns=task["label"]), local_context[task["label"]])
        fit_predictions.append(model.predict(local_query.drop(columns=[task["label"]])).item())

        sizes.append(local_query.shape[0])

    icl_predictions = numpy.array(icl_predictions)
    icl_report = classification_report(
        y_true=query[task["label"]].values,
        y_pred=icl_predictions,
        output_dict=True,
    )["macro avg"]
    icl_report["model"] = "TabICLv2"
    icl_report["task"] = "icl"
    icl_report["task_name"] = task["label"]
    icl_report["modality"] = "tabular"
    icl_report["selection_policy"] = icl_task["selection_policy"]
    icl_report.update(icl_task["kwargs"])
    if task["task"] == Task.CLASSIFICATION:
        icl_report["balanced_accuracy"] = balanced_accuracy_score(query[task["label"]].values, icl_predictions)
    icl_report["sizes"] = sizes


    fit_predictions = numpy.array(fit_predictions)
    fit_report = classification_report(
        y_true=sklearn_query[task["label"]].values,
        y_pred=fit_predictions,
        output_dict=True,
    )["macro avg"]
    fit_report["model"] = "xgboost"
    fit_report["task"] = "icl"
    fit_report["task_name"] = task["label"]
    fit_report["modality"] = "tabular"
    fit_report["selection_policy"] = icl_task["selection_policy"]
    fit_report.update(icl_task["kwargs"])
    if task["task"] == Task.CLASSIFICATION:
        fit_report["balanced_accuracy"] = balanced_accuracy_score(sklearn_query[task["label"]].values, fit_predictions)
    icl_report["sizes"] = sizes

    return icl_report, fit_report


def run():
    with open("configs.json", "r") as f:
        config = json.load(f)

    d = RIMADataset.build().d

    if "classification_report.csv" in os.listdir():
        task_reports = pandas.read_csv("classification_report.csv")
    else:
        task_reports = pandas.DataFrame(columns=[
            "precision",
            "recall",
            "f1-score",
            "support",
            "balanced_accuracy",
            "model",
            "task",
            "task_name",
            "icl_config",
            "icl_config_hash"
        ])

    for task_name, task in config["tasks"].items():
        print(f"Task: {task_name}")
        task["task"] = Task.CLASSIFICATION if task["type"] == "classification" else Task.REGRESSION

        if "fit" in task["modes"]:
            print("\tFitting...")
            try:
                if task_reports[
                    (task_reports["model"] == "TabICLv2") &
                    (task_reports["task"] == "icl_full") &
                    (task_reports["task_name"] == task["label"])
                ].shape[0] == 0:
                    task_results = pandas.DataFrame([fit_task(d, task)])
                    task_results["icl_config_hash"] = numpy.nan
                    task_results["icl_config"] = dict()
                    task_reports = pandas.concat((task_reports, task_results), axis="rows")
                
                task_reports.to_csv("classification_report.csv")


                if task_reports[
                    (task_reports["model"] == "xgboost") &
                    (task_reports["task"] == "fit") &
                    (task_reports["task_name"] == task["label"])
                ].shape[0] == 0:
                    task_results = pandas.DataFrame([fit_xgboost(d, task)])
                    task_results["icl_config_hash"] = numpy.nan
                    task_results["icl_config"] = dict()
                    task_reports = pandas.concat((task_reports, task_results), axis="rows")
                

                task_reports.to_csv("classification_report.csv")
            except Exception:
                print(f"\tError on fit")

        if task["fine_tune"]:
            print("\tTuning...")
            finetuning_config = config["fine_tune"]
            try:
                if task_reports[
                    (task_reports["model"] == "TabICLv2") &
                    (task_reports["task"] == "finetune") &
                    (task_reports["task_name"] == task["label"])
                ].shape[0] == 0:
                    task_results = pandas.DataFrame([finetune_task(d, task, finetuning_config)])
                    task_results["icl_config_hash"] = numpy.nan
                    task_results["icl_config"] = dict()
                    task_reports = pandas.concat((task_reports, task_results), axis="rows")

                
                task_reports.to_csv("classification_report.csv")
            except Exception:
                print(f"\tError on tuning")


        print("\tICL...")
        for task_icl in task["modes"].get("icl", []):
            task_icl_hash = hash(str(task_icl))
            if task_reports[
                (task_reports["model"] == "TabICLv2") &
                (task_reports["task"] == "icl") &
                (task_reports["task_name"] == task["label"]) &
                (task_reports["icl_config_hash"] == task_icl_hash)
            ].shape[0] == 0:
                task_results = pandas.DataFrame(icl_task(d, task, task_icl))
                task_results["icl_config"] = task_icl
                task_results["icl_config_hash"] = task_icl_hash
                task_reports = pandas.concat((task_reports, task_results))
            
            task_reports.to_csv("classification_report.csv")


if __name__ == "__main__":
    run()
