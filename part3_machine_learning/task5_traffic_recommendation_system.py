import os
import pandas as pd
import numpy as np

# =====================================================
# SETUP
# =====================================================

os.makedirs(
    "results",
    exist_ok=True
)

print("Loading dataset...")

df = pd.read_csv(
    "data/traffic_ml_ready.csv"
)

print(
    f"Dataset loaded: "
    f"{df.shape[0]} rows, "
    f"{df.shape[1]} columns"
)

# =====================================================
# CREATE CONGESTION CATEGORY
# =====================================================

q1, q2, q3 = (
    df["traffic_volume"]
    .quantile([0.25, 0.50, 0.75])
    .values
)

def congestion_bucket(v):

    if v <= q1:
        return "Low"

    elif v <= q2:
        return "Medium"

    elif v <= q3:
        return "High"

    return "Severe"

df["congestion_category"] = (
    df["traffic_volume"]
    .apply(congestion_bucket)
)

# =====================================================
# CREATE LOW VISIBILITY FEATURE
# =====================================================

LOW_VISIBILITY = [
    "Fog",
    "Mist",
    "Haze",
    "Smoke"
]

df["is_low_visibility"] = np.where(
    df["weather_main"].isin(
        LOW_VISIBILITY
    ),
    1,
    0
)

# =====================================================
# CREATE PROXY RISK LABEL
# =====================================================

SEVERE_WEATHER = [
    "Rain",
    "Snow",
    "Thunderstorm",
    "Drizzle"
]

high_congestion = (
    df["congestion_category"]
    .isin(["High", "Severe"])
)

risky_weather = (
    df["weather_main"].isin(
        SEVERE_WEATHER
    )
    |
    (df["is_low_visibility"] == 1)
)

df["high_risk"] = (
    high_congestion &
    risky_weather
).astype(int)

# =====================================================
# DAY TYPE
# =====================================================

df["day_type"] = np.where(
    df["is_weekend"] == 1,
    "Weekend",
    "Weekday"
)

# =====================================================
# RECOMMENDATION ENGINE
# =====================================================

recommendations = []

for day_type in ["Weekday", "Weekend"]:

    subset = df[

        (df["day_type"] == day_type)

        &

        (df["high_risk"] == 0)

    ]

    hourly = (

        subset

        .groupby("hour")

        .agg({

            "traffic_volume": "mean"

        })

        .sort_values(
            "traffic_volume"
        )

    )

    top3 = hourly.head(3)

    print("\n" + "=" * 60)
    print(f"{day_type.upper()} RECOMMENDATIONS")
    print("=" * 60)

    for hour in top3.index:

        end_hour = (
            hour + 1
        ) % 24

        avg_traffic = (
            top3.loc[
                hour,
                "traffic_volume"
            ]
        )

        recommendation = (

            f"For a {day_type.lower()} journey, "

            f"consider travelling between "

            f"{hour:02d}:00 and "

            f"{end_hour:02d}:00, "

            f"when historical traffic volumes "

            f"average approximately "

            f"{avg_traffic:.0f} vehicles "

            f"and risk conditions are low."

        )

        print("\n" + recommendation)

        recommendations.append({

            "Day_Type":
                day_type,

            "Recommended_Hour":
                f"{hour:02d}:00",

            "Average_Traffic":
                round(
                    avg_traffic,
                    2
                ),

            "Recommendation":
                recommendation

        })

# =====================================================
# LOWEST TRAFFIC WEATHER
# =====================================================

weather_summary = (

    df.groupby(
        "weather_main"
    )["traffic_volume"]

    .mean()

    .sort_values()

)

best_weather = (
    weather_summary.index[0]
)

best_weather_volume = (
    weather_summary.iloc[0]
)

print("\n" + "=" * 60)
print("WEATHER INSIGHT")
print("=" * 60)

print(

    f"\nThe lowest average traffic volume "

    f"occurs during "

    f"{best_weather} conditions "

    f"({best_weather_volume:.0f} vehicles)."

)

# =====================================================
# PEAK HOURS TO AVOID
# =====================================================

peak_hours = (

    df.groupby("hour")
    ["traffic_volume"]

    .mean()

    .sort_values(
        ascending=False
    )

    .head(5)
)

print("\nHours to Avoid:")

for hour, volume in peak_hours.items():

    print(
        f"{hour:02d}:00 "
        f"(Average Traffic "
        f"{volume:.0f})"
    )

# =====================================================
# SAVE RESULTS
# =====================================================

recommendation_df = pd.DataFrame(
    recommendations
)

recommendation_df.to_csv(

    "results/traffic_recommendations.csv",

    index=False
)

print(
    "\nSaved:"
)

print(
    "results/traffic_recommendations.csv"
)

print(
    "\nTask 5 completed successfully."
)