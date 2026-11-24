# smart-city-traffic-capstone
AI, ML and Data Science Capstone Project

# Traffic Analytics Pipeline

## Project Overview

This project implements a reproducible traffic analytics workflow using the Metro Interstate Traffic Volume dataset. The workflow demonstrates data engineering, feature engineering, data visualization, logging, command-line application development, and Git-based reproducibility practices.

The project is divided into five tasks:

1. Data Pipeline Construction
2. Feature Engineering
3. Traffic Pattern Visualisation
4. Mini Traffic Analytics Application
5. GitHub and Reproducibility

---

# Project Structure

```text
traffic-analytics/
│
├── data/
│   ├── Metro_Interstate_Traffic_Volume.csv
│   ├── traffic_cleaned.csv
│   └── traffic_ml_ready.csv
│
├── figures/
│   ├── traffic_by_hour.png
│   ├── weekday_vs_weekend.png
│   └── monthly_traffic_pattern.png
│
├── logs/
│   ├── pipeline.log
│   ├── feature_engineering.log
│   ├── visualisations.log
│   └── traffic_app.log
│
├── utils/
│   ├── __init__.py
│   └── logger_config.py
│
├── pipeline.py
├── feature_engineering.py
├── visualisations.py
├── traffic_app.py
│
├── README.md
└── requirements.txt
```

---

# Task 1: Data Pipeline Construction

## Objective

Develop a robust and reproducible data pipeline that loads, validates, cleans and prepares the raw traffic dataset for analysis.

## Data Cleaning Activities

The pipeline performs the following operations:

### Data Loading

- Loads the raw CSV file using Pandas.
- Uses exception handling to manage file and parsing errors.
- Logs dataset dimensions upon successful loading.

### Schema Validation

The dataset schema is validated before processing begins.

Expected columns:

- holiday
- temp
- rain_1h
- snow_1h
- clouds_all
- weather_main
- weather_description
- date_time
- traffic_volume

### Data Cleaning

The pipeline:

- Standardises categorical values.
- Parses and validates datetime records.
- Removes duplicate rows.
- Detects invalid temperature values (≤ 0 Kelvin).
- Detects rainfall measurements exceeding 9000 mm.
- Imputes outliers using median values.
- Performs monthly median imputation for missing traffic volumes.

### Processing Logic

The pipeline explicitly uses:

- Functions
- Loops
- Conditional statements
- Exception handling

to satisfy the assignment requirements.

## Output

```text
data/traffic_cleaned.csv
```

---

# Task 2: Feature Engineering

## Objective

Create machine-learning-ready features from the cleaned dataset using Pandas and NumPy.

## Time Features

The following temporal features were created:

| Feature | Description |
|----------|------------|
| hour | Hour of day (0-23) |
| day_of_week | Day of week (0-6) |
| month | Month of year |
| is_weekend | Weekend indicator |

---

## Cyclical Features

Hour values were cyclically encoded to preserve their circular nature.

Features:

- hour_sin
- hour_cos

This allows machine learning models to recognise that 23:00 and 00:00 are temporally adjacent.

---

## Weather Features

The following weather-related features were engineered:

| Feature | Description |
|----------|------------|
| is_raining | Rain indicator |
| is_snowing | Snow indicator |
| bad_weather | Rain, drizzle, snow or thunderstorm indicator |

---

## Encoded Weather Variables

One-hot encoding was applied to:

```text
weather_main
```

This converts categorical weather conditions into numerical features suitable for machine learning.

---

## Scaled Features

Two continuous variables were normalized using Min-Max scaling:

- temp_scaled
- traffic_volume_scaled

Scaled values range between:

```text
0.0 – 1.0
```

---

## Congestion Category Target

A new target variable called:

```text
congestion_level
```

was created from traffic volume.

### Classification Logic

Traffic volumes were divided using quartile-based thresholds:

| Category | Threshold |
|----------|-----------|
| Low | ≤ 25th percentile |
| Medium | 25th–50th percentile |
| High | 50th–75th percentile |
| Very High | > 75th percentile |

### Rationale

Using quartiles creates a balanced and data-driven classification target based on the actual distribution of traffic observations rather than arbitrary thresholds.

## Output

```text
data/traffic_ml_ready.csv
```

---

# Task 3: Visualise Traffic Patterns

## Objective

Use Matplotlib to explore and communicate traffic patterns through meaningful visualisations.

---

## Figure 1 – Average Traffic Volume by Hour

**File**

```text
figures/traffic_by_hour.png
```

### Interpretation

Traffic volume increases during the morning commuting period and remains elevated throughout working hours. A second peak typically occurs during the evening commute before traffic decreases overnight. This pattern reflects typical daily commuting behaviour.

---

## Figure 2 – Weekday vs Weekend Traffic Volume

**File**

```text
figures/weekday_vs_weekend.png
```

### Interpretation

Average traffic volume is higher on weekdays than weekends. This suggests that commuting, business activity and school travel significantly contribute to overall traffic demand.

---

## Figure 3 – Average Monthly Traffic Volume

**File**

```text
figures/monthly_traffic_pattern.png
```

### Interpretation

Traffic volume varies throughout the year, indicating potential seasonal influences such as weather conditions, holidays and vacation travel patterns. Monthly trends provide additional insight beyond daily and weekly traffic cycles.

---

# Task 4: Mini Traffic Analytics Application

## Objective

Develop a command-line application that enables users to query the processed dataset and retrieve traffic insights.

The application uses:

- argparse
- Pandas
- Logging
- Exception handling
- Dataset validation

---

## Available Commands

### 1. Query Traffic Information by Date and Time

Retrieve traffic and weather information for a specific timestamp.

#### Command

```bash
python traffic_app.py query_datetime --datetime "2017-01-15 08:00:00"
```

#### Returns

- Timestamp
- Traffic volume
- Weather condition
- Temperature

---

### 2. Identify High-Traffic Periods

Identify hours commonly associated with severe congestion.

#### Command

```bash
python traffic_app.py high_traffic
```

#### Returns

- Top high-traffic hours
- Average traffic volume

---

### 3. Compare Weekday and Weekend Traffic

Compare average traffic demand between weekdays and weekends.

#### Command

```bash
python traffic_app.py compare
```

#### Returns

- Average weekday traffic volume
- Average weekend traffic volume

---

### 4. Recommend Travel Periods

Identify low-congestion travel periods.

#### Command

```bash
python traffic_app.py travel
```

#### Returns

- Lowest traffic hours
- Recommended travel periods

---

### 5. Weather-Based Traffic Analysis

Analyse traffic conditions under a chosen weather category.

#### Command

```bash
python traffic_app.py weather --condition Rain
```

#### Returns

- Average traffic volume
- Average temperature

---

## Input Validation

The application validates:

- Dataset existence
- Dataset schema
- Date format correctness
- Weather queries

Invalid inputs are handled gracefully and recorded in the application log.

---

# Task 5: GitHub and Reproducibility

## Version Control

Git was used throughout the project to maintain source control and project history.

Incremental commits were created for:

- Task 1: Data Pipeline
- Task 2: Feature Engineering
- Task 3: Visualisation
- Task 4: CLI Application
- Task 5: Documentation and Reproducibility

---

## Logging Configuration

The project implements structured logging using Python's logging module.

Each module creates a logger using:

```python
logger = logging.getLogger(__name__)
```

A shared logging configuration is provided by:

```text
utils/logger_config.py
```

---

## Logging Output

Logs are written to:

### Console Output

Real-time execution monitoring.

### Log Files

| File | Purpose |
|--------|----------|
| logs/pipeline.log | Data loading and cleaning |
| logs/feature_engineering.log | Feature engineering |
| logs/visualisations.log | Figure generation |
| logs/traffic_app.log | Command-line application |

---

## Log Format

All log entries include:

- Timestamp
- Log Level
- Module Name
- Message

Example:

```text
2026-11-21 14:05:18 - INFO - pipeline - Pipeline started.
```

---

## Logging Levels

### DEBUG

Used for internal calculations and troubleshooting.

Examples:

- Quartile thresholds
- Intermediate calculations

### INFO

Used for normal processing milestones.

Examples:

- Data successfully loaded
- Schema validation completed
- File saved successfully
- Visualisation generated

### WARNING

Used for recoverable data-quality issues.

Examples:

- Duplicate rows removed
- Outliers detected
- Missing values imputed

### ERROR

Used for failures that prevent successful execution.

Examples:

- Missing files
- Invalid schema
- Processing failures
- Invalid command parameters

---

# Running the Project

Execute the project in the following order:

## Step 1 – Build Clean Dataset

```bash
python pipeline.py
```

Creates:

```text
data/traffic_cleaned.csv
```

---

## Step 2 – Feature Engineering

```bash
python feature_engineering.py
```

Creates:

```text
data/traffic_ml_ready.csv
```

---

## Step 3 – Generate Visualisations

```bash
python visualisations.py
```

Creates:

```text
figures/traffic_by_hour.png

figures/weekday_vs_weekend.png

figures/monthly_traffic_pattern.png
```

---

## Step 4 – Run the Analytics Application

Example:

```bash
python traffic_app.py compare
```

---

# Reproducibility

This project follows reproducible analytics principles by:

- Separating data processing stages into individual scripts.
- Maintaining structured project folders.
- Logging all major processing events.
- Using schema validation before transformations.
- Saving intermediate and final outputs.
- Applying version control using Git.
- Documenting execution steps in the README.

The workflow can be executed repeatedly on the raw dataset to generate consistent outputs and analysis results.



# Tools and Technologies

The project was developed using the following technologies:

## Programming Language

- Python 3.x

## Python Libraries

- Pandas
- NumPy
- Matplotlib
- Logging
- Argparse

## Development Tools

- Git
- GitHub
- Visual Studio Code

---

# Machine Learning Features

Although a predictive machine learning model was not trained as part of this project, the dataset was prepared for future machine learning applications through feature engineering.

The machine-learning-ready dataset contains:

## Time Features

- hour
- day_of_week
- month
- is_weekend

## Cyclical Features

- hour_sin
- hour_cos

## Weather Features

- is_raining
- is_snowing
- bad_weather

## Encoded Variables

One-hot encoded weather categories derived from:

- weather_main

## Scaled Features

- temp_scaled
- traffic_volume_scaled

## Target Variable

A congestion classification target was created:

```text
congestion_level
```

Categories:

- Low
- Medium
- High
- Very High

These categories were derived using quartile-based thresholds and provide a balanced target suitable for future classification models.

---

# Logging Configuration

The project uses Python's logging module to provide traceability, auditing and reproducibility.

Each module creates a logger using:

```python
logger = logging.getLogger(__name__)
```

A shared logging configuration is implemented in:

```text
utils/logger_config.py
```

The logging system writes messages to:

- Console output
- Dedicated log files

## Log Format

Each log record includes:

- Timestamp
- Log Level
- Module Name
- Message

Example:

```text
2026-11-21 14:05:18 - INFO - pipeline - Pipeline started.
```

## Logging Levels

### DEBUG

Used for detailed internal calculations.

Examples:

- Quartile thresholds
- Intermediate calculations

### INFO

Used for normal processing events.

Examples:

- Dataset loaded
- Features created
- Figure saved

### WARNING

Used for recoverable issues.

Examples:

- Duplicate records removed
- Values imputed
- Outliers detected

### ERROR

Used when processing cannot continue.

Examples:

- Missing files
- Invalid schema
- Invalid user input

## Log File Locations

| Log File | Purpose |
|-----------|----------|
| logs/pipeline.log | Pipeline execution |
| logs/feature_engineering.log | Feature engineering |
| logs/visualisations.log | Figure generation |
| logs/traffic_app.log | Application commands and errors |

---

# Running the Project

Execute the scripts in the following sequence.

## Step 1 - Data Cleaning Pipeline

```bash
python pipeline.py
```

Output:

```text
data/traffic_cleaned.csv
```

---

## Step 2 - Feature Engineering

```bash
python feature_engineering.py
```

Output:

```text
data/traffic_ml_ready.csv
```

---

## Step 3 - Generate Visualisations

```bash
python visualisations.py
```

Outputs:

```text
figures/traffic_by_hour.png

figures/weekday_vs_weekend.png

figures/monthly_traffic_pattern.png
```

---

## Step 4 - Run the Mini Analytics Application

Examples:

```bash
python traffic_app.py compare
```

```bash
python traffic_app.py travel
```

```bash
python traffic_app.py weather --condition Rain
```

```bash
python traffic_app.py query_datetime --datetime "2017-01-15 08:00:00"
```

---

# Reproducing Results

To reproduce the complete analysis:

1. Download the Metro Interstate Traffic Volume dataset.
2. Place the CSV file inside the `data/` folder.
3. Run:

```bash
python pipeline.py
```

4. Run:

```bash
python feature_engineering.py
```

5. Run:

```bash
python visualisations.py
```

6. Review:

```text
traffic_cleaned.csv

traffic_ml_ready.csv

generated figures

log files
```

7. Use the command-line application for interactive analysis.

Following these steps will reproduce the complete workflow and outputs.

---

# Assumptions and Limitations

## Assumptions

- The input dataset contains the expected schema.
- Timestamps are valid after cleaning.
- Outliers can be reasonably corrected using median imputation.
- Quartile-based congestion categories are representative of traffic conditions.

## Limitations

- No predictive machine learning model was trained.
- Results are based on a single traffic dataset.
- Weather effects on traffic were explored descriptively rather than through statistical modelling.
- The command-line application supports a limited set of predefined queries.
- Traffic patterns may not generalize beyond the period and location covered by the dataset.

---

# Future Enhancements

Potential improvements include:

- Traffic volume forecasting using machine learning.
- Congestion prediction models.
- Interactive dashboard development using Power BI or Streamlit.
- Automated scheduling of data refresh and analytics pipelines.
- Additional weather and seasonal feature engineering.