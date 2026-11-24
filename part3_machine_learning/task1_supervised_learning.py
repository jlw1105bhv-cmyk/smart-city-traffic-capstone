import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Classification Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Regression Metrics
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

df = pd.read_csv(
    "data/traffic_ml_ready.csv",
    parse_dates=["date_time"]
)

print(
    f"Dataset loaded: "
    f"{df.shape[0]} rows, "
    f"{df.shape[1]} columns"
)

# =====================================================
# FEATURE PREPARATION
# =====================================================

print("\nPreparing features...")

# Holiday flag

df["holiday_flag"] = np.where(
    df["holiday"] != "No Holiday",
    1,
    0
)

# Day of week cyclical encoding

df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

# =====================================================
# LOW VISIBILITY FEATURE
# =====================================================

LOW_VISIBILITY_WEATHER = [
    "Fog",
    "Mist",
    "Haze",
    "Smoke"
]

df["is_low_visibility"] = np.where(
    df["weather_main"].isin(
        LOW_VISIBILITY_WEATHER
    ),
    1,
    0
)

# =====================================================
# CONGESTION CATEGORY
# =====================================================

q1, q2, q3 = df[
    "traffic_volume"
].quantile(
    [0.25, 0.50, 0.75]
).values


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
# PROXY ACCIDENT RISK LABEL
# =====================================================

SEVERE_WEATHER = [
    "Thunderstorm",
    "Rain",
    "Snow",
    "Drizzle"
]

high_congestion = (
    df["congestion_category"]
    .isin(["High", "Severe"])
)

risky_weather = (
    df["weather_main"]
    .isin(SEVERE_WEATHER)
    |
    (df["is_low_visibility"] == 1)
)

df["high_risk"] = (
    high_congestion &
    risky_weather
).astype(int)

# =====================================================
# USE EXISTING WEATHER FEATURES FROM PART 2
# =====================================================

print("\nChecking engineered weather features...")

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

print("\nWeather features detected:")

for col in weather_columns:
    print(f"  {col}")

if len(weather_columns) == 0:

    raise ValueError(
        "No encoded weather columns found."
    )

for col in weather_columns:

    if df[col].dtype == "bool":

        df[col] = df[col].astype(int)

print(
    f"\nUsing {len(weather_columns)} "
    f"weather encoded features."
)

# =====================================================
# DUPLICATE COLUMN CHECK
# =====================================================

duplicate_columns = (
    df.columns[df.columns.duplicated()]
    .tolist()
)

if duplicate_columns:

    raise ValueError(
        f"Duplicate columns detected: "
        f"{duplicate_columns}"
    )

print("No duplicate columns detected.")

# =====================================================
# COMMON FEATURE SET
# =====================================================

feature_columns = [

    # Time Features
    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    # Holiday
    "holiday_flag",

    # Cyclical Encodings
    "hour_sin",
    "hour_cos",

    "day_sin",
    "day_cos",

    # Weather Features
    "is_raining",
    "is_snowing",
    "bad_weather",
    "is_low_visibility",

    # Temperature
    "temp_scaled"

] + weather_columns

print(
    f"{len(feature_columns)} features selected."
)

# =====================================================
# FEATURE VALIDATION
# =====================================================

missing_features = [

    feature

    for feature in feature_columns

    if feature not in df.columns

]

if missing_features:

    raise ValueError(
        f"Missing features: "
        f"{missing_features}"
    )

print(
    f"All {len(feature_columns)} "
    f"features validated."
)

# =====================================================
# CLASSIFICATION SECTION
# =====================================================

print("\n" + "=" * 60)
print("CLASSIFICATION MODELS")
print("=" * 60)

X = df[feature_columns]
y = df["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------------------------
# Logistic Regression
# -----------------------------------------------------

lr = LogisticRegression(
    max_iter=1000
)

lr.fit(
    X_train,
    y_train
)

lr_pred = lr.predict(X_test)

lr_prob = lr.predict_proba(
    X_test
)[:, 1]

# -----------------------------------------------------
# Random Forest Classifier
# -----------------------------------------------------

rfc = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rfc.fit(
    X_train,
    y_train
)

rfc_pred = rfc.predict(X_test)

rfc_prob = rfc.predict_proba(
    X_test
)[:, 1]

# =====================================================
# CLASSIFICATION COMPARISON
# =====================================================

classification_results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest Classifier"
    ],

    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, rfc_pred)
    ],

    "Precision": [
        precision_score(y_test, lr_pred),
        precision_score(y_test, rfc_pred)
    ],

    "Recall": [
        recall_score(y_test, lr_pred),
        recall_score(y_test, rfc_pred)
    ],

    "F1 Score": [
        f1_score(y_test, lr_pred),
        f1_score(y_test, rfc_pred)
    ],

    "ROC AUC": [
        roc_auc_score(y_test, lr_prob),
        roc_auc_score(y_test, rfc_prob)
    ]
})

print("\nCLASSIFICATION COMPARISON")
print(classification_results.round(4))

best_classifier = classification_results.loc[
    classification_results["ROC AUC"].idxmax()
]

print("\nBEST CLASSIFICATION MODEL")
print(
    f"{best_classifier['Model']} "
    f"(ROC AUC = "
    f"{best_classifier['ROC AUC']:.4f})"
)

# =====================================================
# REGRESSION SECTION
# =====================================================

print("\n" + "=" * 60)
print("REGRESSION MODELS")
print("=" * 60)

X_reg = df[feature_columns]

y_reg = df["traffic_volume"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# -----------------------------------------------------
# Linear Regression
# -----------------------------------------------------

linreg = LinearRegression()

linreg.fit(
    X_train_reg,
    y_train_reg
)

lin_pred = linreg.predict(
    X_test_reg
)

# -----------------------------------------------------
# Random Forest Regressor
# -----------------------------------------------------

rfr = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rfr.fit(
    X_train_reg,
    y_train_reg
)

rfr_pred = rfr.predict(
    X_test_reg
)

# =====================================================
# REGRESSION COMPARISON
# =====================================================

regression_results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest Regressor"
    ],

    "MAE": [
        mean_absolute_error(
            y_test_reg,
            lin_pred
        ),

        mean_absolute_error(
            y_test_reg,
            rfr_pred
        )
    ],

    "R Squared": [
        r2_score(
            y_test_reg,
            lin_pred
        ),

        r2_score(
            y_test_reg,
            rfr_pred
        )
    ]
})

print("\nREGRESSION COMPARISON")
print(regression_results.round(4))

best_regressor = regression_results.loc[
    regression_results["R Squared"].idxmax()
]

print("\nBEST REGRESSION MODEL")
print(
    f"{best_regressor['Model']} "
    f"(R² = "
    f"{best_regressor['R Squared']:.4f})"
)

# =====================================================
# SAVE RESULTS
# =====================================================

os.makedirs(
    "results",
    exist_ok=True
)

classification_results.to_csv(
    "results/classification_model_comparison.csv",
    index=False
)

regression_results.to_csv(
    "results/regression_model_comparison.csv",
    index=False
)

print("\nComparison results saved:")

print(
    "- results/classification_model_comparison.csv"
)

print(
    "- results/regression_model_comparison.csv"
)

print("\nTask 1 completed successfully.")