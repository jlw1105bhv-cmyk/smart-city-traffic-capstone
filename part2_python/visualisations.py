import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import sys

# =====================================================
# LOGGING CONFIGURATION
# =====================================================

from utils.logger_config import setup_logging
os.makedirs("figures", exist_ok=True)


logger = logging.getLogger(__name__)


# =====================================================
# LOAD DATA
# =====================================================

def load_data(filepath):

    try:

        df = pd.read_csv(filepath)

        logger.info(
            f"Dataset loaded successfully. "
            f"Rows={df.shape[0]}, Columns={df.shape[1]}"
        )

        return df

    except FileNotFoundError:

        logger.error(
            f"File not found: {filepath}",
            exc_info=True
        )

        sys.exit(1)

    except Exception:

        logger.error(
            "Failed to load dataset.",
            exc_info=True
        )

        sys.exit(1)


# =====================================================
# VISUALISATION 1
# TRAFFIC DEMAND BY HOUR
# =====================================================

def traffic_by_hour(df):

    hourly_traffic = (
        df.groupby("hour")["traffic_volume"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        hourly_traffic.index,
        hourly_traffic.values,
        marker="o",
        linewidth=2,
        color="steelblue"
    )

    plt.title("Average Traffic Volume by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Traffic Volume")
    plt.grid(True)

    filepath = "figures/traffic_by_hour.png"

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    logger.info(
        f"Figure saved successfully: {filepath}"
    )


# =====================================================
# VISUALISATION 2
# WEEKDAY VS WEEKEND TRAFFIC
# =====================================================

def weekday_vs_weekend(df):

    traffic_data = (
        df.groupby("is_weekend")["traffic_volume"]
        .mean()
    )

    labels = ["Weekday", "Weekend"]

    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        traffic_data.values,
        color=["steelblue", "orange"]
    )

    plt.title(
        "Weekday vs Weekend Traffic Volume"
    )

    plt.ylabel(
        "Average Traffic Volume"
    )

    filepath = "figures/weekday_vs_weekend.png"

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    logger.info(
        f"Figure saved successfully: {filepath}"
    )


# =====================================================
# VISUALISATION 3
# MONTHLY TRAFFIC PATTERN
# =====================================================

def monthly_traffic_pattern(df):

    monthly_traffic = (
        df.groupby("month")["traffic_volume"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        monthly_traffic.index,
        monthly_traffic.values,
        marker="o",
        linewidth=2,
        color="green"
    )

    plt.title(
        "Average Monthly Traffic Volume"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Average Traffic Volume"
    )

    plt.xticks(range(1, 13))
    plt.grid(True)

    filepath = (
        "figures/monthly_traffic_pattern.png"
    )

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    logger.info(
        f"Figure saved successfully: {filepath}"
    )


# =====================================================
# MAIN
# =====================================================

def main():

    setup_logging(
        "logs/visualisations.log"
    )

    try:

        logger.info(
            "Visualisation process started."
        )

        df = load_data(
            "data/traffic_ml_ready.csv"
        )

        traffic_by_hour(df)

        weekday_vs_weekend(df)

        monthly_traffic_pattern(df)

        logger.info(
            "All visualisations completed successfully."
        )

    except Exception:

        logger.error(
            "Visualisation process failed.",
            exc_info=True
        )

        sys.exit(1)


if __name__ == "__main__":
    main()