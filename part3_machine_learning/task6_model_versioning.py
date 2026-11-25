import os
import pandas as pd

os.makedirs(
    "results",
    exist_ok=True
)

model_versions = pd.DataFrame({

    "Version": [
        "v1.0",
        "v2.0",
        "v3.0"
    ],

    "Model": [
        "Linear Regression",
        "Random Forest Regressor",
        "LSTM"
    ],

    "MAE": [
        812.45,
        342.88,
        549.64
    ],

    "R2": [
        0.62,
        0.91,
        0.84
    ],

    "Status": [
        "Baseline",
        "Production Candidate",
        "Experimental"
    ]
})

model_versions.to_csv(
    "results/model_versions.csv",
    index=False
)

print(model_versions)