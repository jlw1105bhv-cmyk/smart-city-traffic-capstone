import os
import pandas as pd
import logging

from utils.logger_config import (
    setup_logging
)

setup_logging(
    "logs/task6_monitoring.log"
)

logger = logging.getLogger(__name__)

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

logger.info(
    f"Drift percentage: "
    f"{drift_percentage:.2f}%"
)

if status == "ALERT":

    logger.warning(
        "Monitoring alert triggered."
    )

else:

    logger.info(
        "Monitoring check passed."
    )

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

logger.info(
    "Monitoring report saved."
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

logger.info(
    "Alert status report saved."
)

print("\nALERT STATUS")

print(alert_status)