import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

import shap
import matplotlib.pyplot as plt

# =====================================================
# OUTPUT FOLDER
# =====================================================

os.makedirs(
    "results",
    exist_ok=True
)

os.makedirs(
    "figures",
    exist_ok=True
)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

df = pd.read_csv(
    "data/traffic_ml_ready.csv"
)

print(
    f"Dataset: "
    f"{df.shape[0]} rows, "
    f"{df.shape[1]} columns"
)

# =====================================================
# FEATURES
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

    "hour_sin",
    "hour_cos",

    "temp_scaled",

    "is_raining",
    "is_snowing",
    "bad_weather"

] + weather_columns

target = "traffic_volume"

# =====================================================
# SCALE DATA
# =====================================================

X = df[feature_columns]

y = df[target]

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# CREATE LSTM SEQUENCES
# =====================================================

def create_sequences(
    X,
    y,
    sequence_length=24
):

    X_seq = []
    y_seq = []

    for i in range(
        sequence_length,
        len(X)
    ):

        X_seq.append(
            X[
                i-sequence_length:i
            ]
        )

        y_seq.append(
            y.iloc[i]
        )

    return (
        np.array(X_seq),
        np.array(y_seq)
    )

X_seq, y_seq = create_sequences(
    X_scaled,
    y
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

split_index = int(
    len(X_seq) * 0.8
)

X_train = X_seq[:split_index]
X_test = X_seq[split_index:]

y_train = y_seq[:split_index]
y_test = y_seq[split_index:]

# =====================================================
# BUILD LSTM
# =====================================================

print("\nTraining LSTM...")

model = Sequential([

    LSTM(
        64,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1)

])

model.compile(

    optimizer="adam",

    loss="mse",

    metrics=["mae"]

)

history = model.fit(

    X_train,
    y_train,

    epochs=10,

    batch_size=64,

    validation_split=0.2,

    verbose=1
)

# =====================================================
# EVALUATE LSTM
# =====================================================

y_pred = model.predict(
    X_test
).flatten()

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nLSTM RESULTS")

print(
    f"MAE: {mae:.2f}"
)

print(
    f"R²: {r2:.4f}"
)

# =====================================================
# SAVE RESULTS
# =====================================================

lstm_results = pd.DataFrame({

    "Metric": [
        "MAE",
        "R2"
    ],

    "Value": [
        mae,
        r2
    ]

})

lstm_results.to_csv(
    "results/lstm_results.csv",
    index=False
)

# =====================================================
# SHAP EXPLAINABILITY
# =====================================================

print(
    "\nTraining Random Forest "
    "for SHAP..."
)

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(

    X_scaled,

    y,

    test_size=0.2,

    random_state=42
)

rf = RandomForestRegressor(

    n_estimators=100,

    random_state=42
)

rf.fit(
    X_train_rf,
    y_train_rf
)

# =====================================================
# SHAP
# =====================================================

explainer = shap.TreeExplainer(
    rf
)

sample = X_test_rf[:200]

shap_values = explainer.shap_values(
    sample
)

plt.figure()

shap.summary_plot(

    shap_values,

    sample,

    feature_names=feature_columns,

    show=False
)

plt.tight_layout()

plt.savefig(
    "figures/shap_summary_plot.png"
)

plt.close()

# =====================================================
# FEATURE IMPORTANCE TABLE
# =====================================================

importance = np.abs(
    shap_values
).mean(axis=0)

importance_df = pd.DataFrame({

    "Feature": feature_columns,

    "SHAP Importance":
        importance

})

importance_df = (

    importance_df

    .sort_values(
        "SHAP Importance",
        ascending=False
    )
)

importance_df.to_csv(

    "results/shap_feature_importance.csv",

    index=False
)

print(
    "\nTop SHAP Features"
)

print(
    importance_df.head(10)
)

print(
    "\nOutputs Saved:"
)

print(
    "results/lstm_results.csv"
)

print(
    "results/shap_feature_importance.csv"
)

print(
    "figures/shap_summary_plot.png"
)

print(
    "\nTask 3 completed successfully."
)