import os
import pandas as pd

os.makedirs(
    "results",
    exist_ok=True
)

# Historical baseline

baseline_mean = 3260

# Simulated new data

new_mean = 3800

drift_percentage = (

    abs(
        new_mean -
        baseline_mean
    )

    /
    baseline_mean

) * 100

status = "PASS"

if drift_percentage > 10:

    status = "ALERT"

monitoring_report = pd.DataFrame({

    "Metric":
        ["Traffic Volume Mean"],

    "Baseline":
        [baseline_mean],

    "Current":
        [new_mean],

    "Drift_%":
        [round(
            drift_percentage,
            2
        )],

    "Status":
        [status]

})

monitoring_report.to_csv(

    "results/monitoring_report.csv",

    index=False
)

print(monitoring_report)

alert_status = pd.DataFrame({

    "Check": [
        "Feature Drift"
    ],

    "Status": [
        status
    ],

    "Action": [

        "Review model"

        if status == "ALERT"

        else

        "No action needed"
    ]

})

alert_status.to_csv(

    "results/drift_alert_status.csv",

    index=False
)

print("\nALERT STATUS")

print(alert_status)