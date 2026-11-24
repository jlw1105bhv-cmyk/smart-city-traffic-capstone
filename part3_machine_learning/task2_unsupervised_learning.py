import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)

# =====================================================
# CREATE RESULTS FOLDER
# =====================================================

os.makedirs(
    "results",
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
    f"Dataset loaded: "
    f"{df.shape[0]} rows, "
    f"{df.shape[1]} columns"
)

# =====================================================
# CREATE CONGESTION CATEGORY
# =====================================================

q1, q2, q3 = (
    df["traffic_volume"]
    .quantile([0.25, 0.5, 0.75])
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
# K-MEANS CLUSTERING
# =====================================================

print("\n" + "=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)

# Weather severity score

severity_map = {
    "Clear": 1,
    "Clouds": 2,
    "Mist": 3,
    "Fog": 4,
    "Haze": 4,
    "Smoke": 4,
    "Drizzle": 5,
    "Rain": 6,
    "Snow": 7,
    "Thunderstorm": 8,
    "Squall": 8
}

df["weather_severity"] = (
    df["weather_main"]
    .map(severity_map)
    .fillna(2)
)

cluster_features = [

    "hour",
    "weather_severity",
    "traffic_volume"

]

X_cluster = df[cluster_features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_cluster
)

# 4 clusters works well with quartile traffic data

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(
    X_scaled
)

cluster_summary = (

    df.groupby("cluster")[
        [
            "hour",
            "weather_severity",
            "traffic_volume"
        ]
    ]

    .mean()

    .round(2)

)

print("\nCluster Summary")

print(cluster_summary)

cluster_summary.to_csv(
    "results/kmeans_cluster_summary.csv"
)

# =====================================================
# INTERPRET CLUSTERS
# =====================================================

print("\nCluster Interpretation")

for cluster in cluster_summary.index:

    avg_hour = cluster_summary.loc[
        cluster,
        "hour"
    ]

    avg_weather = cluster_summary.loc[
        cluster,
        "weather_severity"
    ]

    avg_traffic = cluster_summary.loc[
        cluster,
        "traffic_volume"
    ]

    print(
        f"\nCluster {cluster}"
    )

    print(
        f"Average Hour: {avg_hour:.1f}"
    )

    print(
        f"Weather Severity: "
        f"{avg_weather:.1f}"
    )

    print(
        f"Traffic Volume: "
        f"{avg_traffic:.1f}"
    )

# =====================================================
# ASSOCIATION RULE MINING
# =====================================================

print("\n" + "=" * 60)
print("ASSOCIATION RULE MINING")
print("=" * 60)

# -----------------------------------------------------
# TIME OF DAY
# -----------------------------------------------------

def time_bucket(hour):

    if 0 <= hour < 6:
        return "Night"

    elif 6 <= hour < 12:
        return "Morning"

    elif 12 <= hour < 18:
        return "Afternoon"

    return "Evening"

df["time_of_day"] = (
    df["hour"]
    .apply(time_bucket)
)

# -----------------------------------------------------
# WEEKDAY TYPE
# -----------------------------------------------------

df["weekday_type"] = np.where(
    df["is_weekend"] == 1,
    "Weekend",
    "Weekday"
)

# -----------------------------------------------------
# WEATHER CATEGORY
# -----------------------------------------------------

df["weather_condition"] = np.where(

    df["weather_main"].isin(
        [
            "Rain",
            "Snow",
            "Thunderstorm",
            "Drizzle"
        ]
    ),

    "Bad Weather",

    "Normal Weather"
)

# -----------------------------------------------------
# TRANSACTION FORMAT
# -----------------------------------------------------

transactions = pd.DataFrame({

    "time_item":
        "Time_" +
        df["time_of_day"],

    "day_item":
        "Day_" +
        df["weekday_type"],

    "weather_item":
        "Weather_" +
        df["weather_condition"],

    "congestion_item":
        "Congestion_" +
        df["congestion_category"]

})

# Convert to basket format

basket = pd.get_dummies(
    transactions.stack()
)

transactions = (
    basket
    .groupby(level=0)
    .max()
    .astype(bool)
)

# -----------------------------------------------------
# APRIORI
# -----------------------------------------------------

frequent_itemsets = apriori(
    transactions,
    min_support=0.03,
    use_colnames=True
)

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

rules = rules.sort_values(
    by="lift",
    ascending=False
)

# Keep strongest rules

top_rules = rules[
    [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]
].head(10)

print("\nTop Rules")

print(top_rules)

top_rules.to_csv(
    "results/association_rules.csv",
    index=False
)

# =====================================================
# PLAIN LANGUAGE RULE INTERPRETATION
# =====================================================

print(
    "\nTop Association Rules "
    "(Plain Language)"
)

for i, row in top_rules.head(5).iterrows():

    print("\nRule")

    print(
        f"If {set(row['antecedents'])}"
    )

    print(
        f"Then {set(row['consequents'])}"
    )

    print(
        f"Confidence = "
        f"{row['confidence']:.2f}"
    )

    print(
        f"Lift = "
        f"{row['lift']:.2f}"
    )

# =====================================================
# SAVE OUTPUTS
# =====================================================

print("\nResults saved:")

print(
    "results/kmeans_cluster_summary.csv"
)

print(
    "results/association_rules.csv"
)

print(
    "\nTask 2 completed successfully."
)