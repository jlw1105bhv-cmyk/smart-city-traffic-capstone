import pandas as pd
import logging
import argparse
import os
import sys

# =====================================================
# LOGGING CONFIGURATION
# =====================================================
from utils.logger_config import setup_logging

logger = logging.getLogger(__name__)


# =====================================================
# REQUIRED COLUMNS
# =====================================================

REQUIRED_COLUMNS = [
    "date_time",
    "traffic_volume",
    "weather_main",
    "temp",
    "hour",
    "is_weekend",
    "congestion_level"
]

# =====================================================
# LOAD DATA
# =====================================================

def load_data():

    try:

        df = pd.read_csv(
            "data/traffic_ml_ready.csv",
            parse_dates=["date_time"]
        )

        logger.info(
            f"Dataset loaded successfully. "
            f"Rows={df.shape[0]}, Columns={df.shape[1]}"
        )

        return df

    except FileNotFoundError:

        logger.error(
            "traffic_ml_ready.csv not found."
        )

        print(
            "ERROR: Processed dataset not found."
        )

        sys.exit(1)

    except Exception:

        logger.error(
            "Dataset loading failed.",
            exc_info=True
        )

        print(
            "ERROR: Unable to load dataset."
        )

        sys.exit(1)

# =====================================================
# VALIDATE SCHEMA
# =====================================================

def validate_schema(df):

    missing_columns = []

    for column in REQUIRED_COLUMNS:

        if column not in df.columns:

            missing_columns.append(column)

    if missing_columns:

        logger.error(
            f"Missing required columns: "
            f"{missing_columns}"
        )

        print(
            "ERROR: Dataset schema validation failed."
        )

        print(
            f"Missing columns: {missing_columns}"
        )

        sys.exit(1)

    logger.info(
        "Dataset schema validation passed."
    )

# =====================================================
# COMMAND 1
# QUERY SPECIFIC DATETIME
# =====================================================

def query_datetime(df, user_datetime):

    logger.info(
        f"Command=query_datetime "
        f"Argument={user_datetime}"
    )

    try:

        target_datetime = pd.to_datetime(
            user_datetime
        )

    except ValueError:

        logger.error(
            f"Invalid datetime supplied: "
            f"{user_datetime}"
        )

        print(
            "ERROR: Invalid datetime format."
        )

        print(
            "Example:"
        )

        print(
            "2017-01-15 08:00:00"
        )

        return

    result = df[
        df["date_time"] == target_datetime
    ]

    if result.empty:

        print(
            "No matching records found."
        )

        return

    print(
        result[
            [
                "date_time",
                "traffic_volume",
                "weather_main",
                "temp"
            ]
        ]
    )

# =====================================================
# COMMAND 2
# HIGH TRAFFIC PERIODS
# =====================================================

def high_traffic_periods(df):

    logger.info(
        "Command=high_traffic"
    )

    top_hours = (

        df[df["congestion_level"] == "Very High"]

        .groupby("hour")["traffic_volume"]

        .mean()

        .sort_values(ascending=False)

        .head(10)
    )

    print("\nTop High-Traffic Hours\n")

    print(top_hours)

# =====================================================
# COMMAND 3
# WEEKDAY VS WEEKEND
# =====================================================

def compare_weekday_weekend(df):

    logger.info(
        "Command=compare"
    )

    comparison = (
        df.groupby("is_weekend")
        ["traffic_volume"]
        .mean()
    )

    weekday_avg = comparison.get(0, 0)
    weekend_avg = comparison.get(1, 0)

    print("\nAverage Traffic Volume\n")

    print(
        f"Weekday : {weekday_avg:.2f}"
    )

    print(
        f"Weekend : {weekend_avg:.2f}"
    )

# =====================================================
# COMMAND 4
# RECOMMENDED TRAVEL PERIODS
# =====================================================

def recommended_travel_periods(df):

    logger.info(
        "Command=travel"
    )

    best_hours = (
        df.groupby("hour")
        ["traffic_volume"]
        .mean()
        .sort_values()
        .head(5)
    )

    print(
        "\nRecommended Travel Hours "
        "(Lowest Average Traffic)\n"
    )

    print(best_hours)

# =====================================================
# COMMAND 5
# WEATHER + TRAFFIC INFO
# =====================================================

def weather_traffic(df, condition):

    logger.info(
        f"Command=weather "
        f"Argument={condition}"
    )

    result = df[
        df["weather_main"]
        .str.lower()
        ==
        condition.lower()
    ]

    if result.empty:

        logger.error(
            f"No data found for weather "
            f"condition: {condition}"
        )

        print(
            f"No records found for "
            f"'{condition}'."
        )

        return

    avg_traffic = (
        result["traffic_volume"]
        .mean()
    )

    avg_temp = (
        result["temp"]
        .mean()
    )

    print(
        f"\nWeather Condition: "
        f"{condition}"
    )

    print(
        f"Average Traffic Volume: "
        f"{avg_traffic:.2f}"
    )

    print(
        f"Average Temperature: "
        f"{avg_temp:.2f} K"
    )

# =====================================================
# MAIN
# =====================================================

def main():

    setup_logging(
        "logs/traffic_app.log"
    )
    
    parser = argparse.ArgumentParser(
        description=
        "Traffic Analytics Application"
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    # Query Datetime

    p1 = subparsers.add_parser(
        "query_datetime"
    )

    p1.add_argument(
        "--datetime",
        required=True,
        help="YYYY-MM-DD HH:MM:SS"
    )

    # High Traffic

    subparsers.add_parser(
        "high_traffic"
    )

    # Compare

    subparsers.add_parser(
        "compare"
    )

    # Travel

    subparsers.add_parser(
        "travel"
    )

    # Weather

    p5 = subparsers.add_parser(
        "weather"
    )

    p5.add_argument(
        "--condition",
        required=True
    )

    args = parser.parse_args()

    df = load_data()

    validate_schema(df)

    if args.command == "query_datetime":

        query_datetime(
            df,
            args.datetime
        )

    elif args.command == "high_traffic":

        high_traffic_periods(df)

    elif args.command == "compare":

        compare_weekday_weekend(df)

    elif args.command == "travel":

        recommended_travel_periods(df)

    elif args.command == "weather":

        weather_traffic(
            df,
            args.condition
        )

    else:

        parser.print_help()

if __name__ == "__main__":
    main()
