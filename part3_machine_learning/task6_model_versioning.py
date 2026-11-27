import os
import pandas as pd
import logging

from utils.logger_config import (
    setup_logging
)

setup_logging(
    "logs/task6_model_versioning.log"
)

logger = logging.getLogger(__name__)

os.makedirs(
    "results",
    exist_ok=True
)

model_versions = pd.DataFrame({

    "Version": [
        "v1.0",
        "v2.0",
        "v3.0",
        "v1.0",
        "v2.0"
    ],

    "Model": [
        "Linear Regression",
        "Random Forest Regressor",
        "LSTM",
        "Logistic Regression",
        "Random Forest Classifier"
    ],

    "Problem_Type": [
        "Regression",
        "Regression",
        "Regression",
        "Classification",
        "Classification"
    ],

    "MAE": [
        812.45,      # replace
        342.88,      # replace
        549.64,      # your Task 3 result
        None,
        None
    ],

    "R2": [
        0.62,        # replace
        0.91,        # replace
        0.8435,      # your Task 3 result
        None,
        None
    ],

    "Accuracy": [
        None,
        None,
        None,
        0.95,        # replace
        0.98         # replace
    ],

    "Precision": [
        None,
        None,
        None,
        0.93,        # replace
        0.97         # replace
    ],

    "Recall": [
        None,
        None,
        None,
        0.89,        # replace
        0.95         # replace
    ],

    "F1": [
        None,
        None,
        None,
        0.91,        # replace
        0.96         # replace
    ],

    "ROC_AUC": [
        None,
        None,
        None,
        0.97,        # replace
        0.99         # replace
    ],

    "Status": [
        "Baseline",
        "Production Candidate",
        "Experimental",
        "Baseline",
        "Production Candidate"
    ]
})

model_versions.to_csv(
    "results/model_versions.csv",
    index=False
)

print(model_versions)

logger.info(
    "Model version registry saved to "
    "results/model_versions.csv"
)