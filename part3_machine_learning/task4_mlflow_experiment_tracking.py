import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import logging

from utils.logger_config import (
    setup_logging
)

setup_logging(
    "logs/task4_mlflow_experiment_tracking.log"
)

logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,

    mean_absolute_error,
    r2_score
)

# =====================================================
# LOAD DATA
# =====================================================

logger.info("Loading dataset...")
os.makedirs(
    "results",
    exist_ok=True
)

df = pd.read_csv(
    "data/traffic_ml_ready.csv"
)

# =====================================================
# PREPARE FEATURES
# =====================================================

df["holiday_flag"] = np.where(
    df["holiday"] != "No Holiday",
    1,
    0
)

df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

df["is_low_visibility"] = np.where(
    df["weather_main"].isin(
        ["Fog", "Mist", "Haze", "Smoke"]
    ),
    1,
    0
)

# =====================================================
# CONGESTION CATEGORY
# =====================================================

q1, q2, q3 = (
    df["traffic_volume"]
    .quantile([0.25, 0.50, 0.75])
    .values
)

def bucket(v):

    if v <= q1:
        return "Low"
    elif v <= q2:
        return "Medium"
    elif v <= q3:
        return "High"

    return "Severe"

df["congestion_category"] = (
    df["traffic_volume"]
    .apply(bucket)
)

# =====================================================
# HIGH RISK LABEL
# =====================================================

high_congestion = (
    df["congestion_category"]
    .isin(["High", "Severe"])
)

risky_weather = (
    df["bad_weather"] == 1
) | (
    df["is_low_visibility"] == 1
)

df["high_risk"] = (
    high_congestion &
    risky_weather
).astype(int)

# =====================================================
# WEATHER FEATURES
# =====================================================

weather_columns = [

    col

    for col in df.columns

    if (
        col.startswith("weather_")
        and col not in [
            "weather_main",
            "weather_description"
        ]
    )

]

feature_columns = [

    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    "holiday_flag",

    "hour_sin",
    "hour_cos",

    "day_sin",
    "day_cos",

    "temp_scaled",

    "is_raining",
    "is_snowing",

    "bad_weather",

    "is_low_visibility"

] + weather_columns

# =====================================================
# MLFLOW
# =====================================================

mlflow.set_experiment(
    "Traffic Analytics Capstone"
)

# =====================================================
# EXPERIMENT RESULTS STORAGE
# =====================================================

experiment_results = []
# =====================================================
# LOGISTIC REGRESSION
# =====================================================

X = df[feature_columns]
y = df["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run(
    run_name="Logistic Regression"
):

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(X_test)

    prob = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        pred
    )

    precision = precision_score(
        y_test,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        pred
    )

    roc_auc = roc_auc_score(
        y_test,
        prob
    )

    mlflow.log_param(
        "model",
        "Logistic Regression"
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )
    
    experiment_results.append({

    "Model":
        "Logistic Regression",

    "Problem":
        "Classification",

    "Accuracy":
        accuracy,

    "Precision":
        precision,

    "Recall":
        recall,

    "F1":
        f1,

    "ROC_AUC":
        roc_auc,

    "MAE":
        None,

    "R2":
        None

    })

    try:

        mlflow.sklearn.log_model(
            model,
            "model"
        )

    except Exception as e:

        logger.warning(
            f"Model artifact not logged: {e}"
        )


# =====================================================
# RANDOM FOREST REGRESSOR
# =====================================================

X = df[feature_columns]
y = df["traffic_volume"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run(
    run_name="Random Forest Regressor"
):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    r2 = r2_score(
        y_test,
        pred
    )

    mlflow.log_param(
        "model",
        "Random Forest Regressor"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_metric(
        "mae",
        mae
    )

    mlflow.log_metric(
        "r2",
        r2
    )
    
    experiment_results.append({

    "Model":
        "Random Forest Regressor",

    "Problem":
        "Regression",

    "Accuracy":
        None,

    "Precision":
        None,

    "Recall":
        None,

    "F1":
        None,

    "ROC_AUC":
        None,

    "MAE":
        mae,

    "R2":
        r2

    })

    try:

        mlflow.sklearn.log_model(
            model,
            "model"
        )

    except Exception as e:

        logger.warning(
            f"Model artifact not logged: {e}"
        )


# =====================================================
# SAVE EXPERIMENT SUMMARY
# =====================================================

experiment_df = pd.DataFrame(
    experiment_results
)

experiment_df.to_csv(
    "results/mlflow_experiment_summary.csv",
    index=False
)

logger.info(
    "Experiments logged successfully."
)

logger.info(
    "Experiment summary saved to "
    "results/mlflow_experiment_summary.csv"
)

logger.info(
    "Task 4 completed successfully."
)