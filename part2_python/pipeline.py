import pandas as pd
import numpy as np
import logging
import os
import sys

# =====================================================
# LOGGING CONFIGURATION
# =====================================================

from utils.logger_config import setup_logging

logger = logging.getLogger(__name__)

# =====================================================
# EXPECTED SCHEMA
# =====================================================

EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume"
]

# =====================================================
# LOAD DATA
# =====================================================

def load_data(filepath):
    """
    Load CSV file safely.
    """

    try:
        df = pd.read_csv(filepath)

        logger.info(
            f"File loaded successfully. "
            f"Rows={df.shape[0]}, Columns={df.shape[1]}"
        )

        return df

    except FileNotFoundError:
        logger.error(
            f"File not found: {filepath}",
            exc_info=True
        )
        sys.exit()

    except pd.errors.ParserError:
        logger.error(
            "CSV parsing failure",
            exc_info=True
        )
        sys.exit()

    except Exception:
        logger.error(
            "Unexpected error loading file",
            exc_info=True
        )
        sys.exit()

# =====================================================
# VALIDATE SCHEMA
# =====================================================

def validate_schema(df):

    missing_columns = []

    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:

        logger.error(
            f"Schema validation failed. "
            f"Missing columns: {missing_columns}"
        )

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    logger.info("Schema validation successful.")

# =====================================================
# STANDARDISE CATEGORICAL VALUES
# =====================================================

def standardise_categories(df):

    before = df["holiday"].nunique()

    holiday_mapping = {
        "None": "No Holiday",
        np.nan: "No Holiday"
    }

    df["holiday"] = (
        df["holiday"]
        .replace(holiday_mapping)
        .fillna("No Holiday")
    )

    after = df["holiday"].nunique()

    logger.warning(
        f"Standardised holiday categories. "
        f"Unique values before={before}, after={after}"
    )

    return df

# =====================================================
# VALIDATE DATETIME
# =====================================================

def parse_datetime(df):

    invalid_before = df["date_time"].isna().sum()

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    invalid_after = df["date_time"].isna().sum()

    affected = invalid_after - invalid_before

    if affected > 0:

        logger.warning(
            f"{affected} invalid date rows detected."
        )

    df = df.dropna(subset=["date_time"])

    return df

# =====================================================
# REMOVE DUPLICATES
# =====================================================

def remove_duplicates(df):

    duplicates = df.duplicated().sum()

    if duplicates > 0:

        logger.warning(
            f"Removed {duplicates} duplicate rows."
        )

        df = df.drop_duplicates()

    return df

# =====================================================
# HANDLE OUTLIERS
# =====================================================

def handle_temperature_outliers(df):

    # Impossible Kelvin conversion example

    invalid_mask = df["temp"] <= 0

    count = invalid_mask.sum()

    if count > 0:

        median_temp = df.loc[
            ~invalid_mask,
            "temp"
        ].median()

        df.loc[
            invalid_mask,
            "temp"
        ] = median_temp

        logger.warning(
            f"Imputed {count} temperature outliers "
            f"using median value {median_temp:.2f}"
        )

    return df


def handle_rainfall_outliers(df):

    invalid_mask = df["rain_1h"] > 9000

    count = invalid_mask.sum()

    if count > 0:

        median_rain = df.loc[
            ~invalid_mask,
            "rain_1h"
        ].median()

        df.loc[
            invalid_mask,
            "rain_1h"
        ] = median_rain

        logger.warning(
            f"Imputed {count} rainfall outliers "
            f"(>9000 mm) using median={median_rain}"
        )

    return df

# =====================================================
# MONTHLY MEDIAN IMPUTATION
# LOOP + CONDITIONAL REQUIREMENT
# =====================================================

def monthly_imputation(df):

    df["month"] = df["date_time"].dt.month

    for month in range(1, 13):

        month_mask = df["month"] == month

        month_median = df.loc[
            month_mask,
            "traffic_volume"
        ].median()

        missing_rows = (
            month_mask &
            df["traffic_volume"].isna()
        )

        count = missing_rows.sum()

        if count > 0:

            df.loc[
                missing_rows,
                "traffic_volume"
            ] = month_median

            logger.warning(
                f"Imputed {count} traffic_volume values "
                f"for month {month} "
                f"using median={month_median:.2f}"
            )

    return df

# =====================================================
# SAVE OUTPUT
# =====================================================

def save_data(df, filepath):

    df.to_csv(filepath, index=False)

    logger.info(
        f"Cleaned data saved to {filepath}"
    )

# =====================================================
# MAIN PIPELINE
# =====================================================

def main():

    setup_logging("logs/pipeline.log")

    try:

        logger.info("Pipeline started.")

        df = load_data(
            "data/Metro_Interstate_Traffic_Volume.csv"
        )

        validate_schema(df)

        df = standardise_categories(df)

        df = parse_datetime(df)

        df = remove_duplicates(df)

        df = handle_temperature_outliers(df)

        df = handle_rainfall_outliers(df)

        df = monthly_imputation(df)

        save_data(
            df,
            "data/traffic_cleaned.csv"
        )

        logger.info(
            "Pipeline completed successfully."
        )

    except Exception:

        logger.error(
            "Pipeline failed.",
            exc_info=True
        )

        sys.exit(1)


if __name__ == "__main__":
    main()