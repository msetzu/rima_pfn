import json
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


#######################################################################################################################
# Tasks ###############################################################################################################
#######################################################################################################################

d = RIMADataset.build().d

task_reports = list()
###########################################################
# region/school classification ############################
###########################################################

#############################
# Fit #######################
#############################
print("School classification")
df = d.select_dtypes(include="number").drop(columns=[
    "total_tracks",
    "album_popularity",
    "birth_lat",
    "birth_long",
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

print(f"\tTabICL")
fit_tabiclV2_model = TabularModel().fit_sdm(train_df, "birth_region", task=Task.CLASSIFICATION).model
fit_tabiclV2_predictions = fit_tabiclV2_model.predict(test_df)
report = classification_report(
    y_true=df["birth_region"],
    y_pred=fit_tabiclV2_predictions,
    output_dict=True,
)["macro_avg"]
report["model"] = "TabICLv2"
report["task"] = "fit_classification"
report["task_name"] = "region"
report["modality"] = "tabular"
task_reports.append(report)


#############################
# zero-shot #################
#############################
print(f"\tTabICL")
fit_tabiclV2_model = TabularModel().fit_sdm(train_df, "birth_region", task=Task.CLASSIFICATION).model
fit_tabiclV2_predictions = fit_tabiclV2_model.predict(test_df)
report = classification_report(
    y_true=df["birth_region"],
    y_pred=fit_tabiclV2_predictions,
    output_dict=True,
)["macro avg"]
report["model"] = "TabICLv2"
report["task"] = "oneshot_classification"
report["task_name"] = "region"
report["modality"] = "tabular"
task_reports.append(report)


#############################
# ICL #######################
#############################
icl_model = ICLTabularModel()

for context_selection in CONFIGS["icl"]:
    if isinstance(context_selection, str):
        query = test_df
        context = train_df
        icl_model.query(
            data=query,
            label="region",
            task=Task.CLASSIFICATION,
            context=context,
        )

    else:
        query = test_df
        context_set = train_df

        context_selection_policy = context_selection["selection_policy"]
        context_selection_function = getattr(context, context_selection_policy)
        context_selection_function = partial(context_selection_function, data=context_set, **context_selection["kwargs"])
        contexts_per_row = query[context_selection["object_id"]].apply(context_selection_function)

        icl_model.query(
            data=query,
            label="region",
            task=Task.CLASSIFICATION,
            context=contexts_per_row,
        )

