# smart-city-traffic-capstone
AI, ML and Data Science Capstone Project

# Traffic Analytics Pipeline

## Project Overview

This project develops a reproducible traffic analytics workflow using the Metro Interstate Traffic Volume dataset.

The solution consists of:

1. Data Pipeline Construction
2. Feature Engineering
3. Data Visualisation
4. Logging and Reproducibility

Python packages used:

- Pandas
- NumPy
- Matplotlib
- Logging

---

## Project Structure

traffic-analytics/

├── data/

│ ├── Metro_Interstate_Traffic_Volume.csv

│ ├── traffic_cleaned.csv

│ └── traffic_ml_ready.csv

│

├── figures/

│ ├── traffic_by_hour.png

│ ├── weekday_vs_weekend.png

│ └── congestion_distribution.png

│

├── logs/

│ ├── pipeline.log

│ ├── feature_engineering.log

│ └── visualisations.log

│

├── pipeline.py

├── feature_engineering.py

├── visualisations.py

└── README.md

---

## Task 1: Data Pipeline

The pipeline performs:

- CSV loading with exception handling
- Schema validation
- Datetime validation
- Duplicate removal
- Standardisation of categorical values
- Outlier detection
- Median-based imputation
- Logging of every processing step

Output:

traffic_cleaned.csv

---

## Task 2: Feature Engineering

The following machine learning features were created.

### Time Features

- hour
- day_of_week
- month
- is_weekend

### Cyclical Encoding

- hour_sin
- hour_cos

### Weather Features

- is_raining
- is_snowing
- bad_weather

### Encoded Weather Categories

One-hot encoding applied to:

- weather_main

### Scaled Numerical Features

- temp_scaled
- traffic_volume_scaled

### Congestion Target

A congestion category was created using quartile-based thresholds.

| Category | Definition |
|----------|------------|
| Low | <= 25th percentile |
| Medium | 25th - 50th percentile |
| High | 50th - 75th percentile |
| Very High | > 75th percentile |

This approach creates a balanced, data-driven classification target suitable for machine learning.

Output:

traffic_ml_ready.csv

---

## Task 3: Traffic Pattern Visualisation

Three Matplotlib visualisations were produced.

### Figure 1: Average Traffic Volume by Hour

File:

figures/traffic_by_hour.png

Interpretation:

Traffic volume increases significantly during morning commuting hours and remains relatively high throughout the working day. A second peak occurs during evening commute periods before traffic decreases overnight. These patterns reflect typical daily commuting behaviour.

---

### Figure 2: Weekday vs Weekend Traffic Volume

File:

figures/weekday_vs_weekend.png

Interpretation:

Average traffic volume is higher on weekdays than weekends. This suggests that commuting, work activities and school travel are major contributors to traffic demand across the interstate corridor.

---

### Figure 3: Distribution of Congestion Levels

File:

figures/congestion_distribution.png

Interpretation:

The congestion categories appear relatively balanced because they were created using quartile-based thresholds derived from traffic volume. This confirms that the feature engineering process successfully generated a data-driven target suitable for classification modelling.

---

## Logging

Logging was implemented throughout the project using Python's logging module.

Log files include:

- logs/pipeline.log
- logs/feature_engineering.log
- logs/visualisations.log

Logging provides:

- Auditability
- Traceability
- Error reporting
- Reproducibility

The project uses:

- INFO messages for successful operations
- WARNING messages for cleaning and imputation actions
- DEBUG messages for intermediate calculations
- ERROR messages for failures and exceptions

---

## Running the Project

Step 1:
python pipeline.py

Step 2:
python feature_engineering.py

Step 3:
python visualisations.py


Generated outputs include:
traffic_cleaned.csv
traffic_ml_ready.csv
visualisation figures
log files


