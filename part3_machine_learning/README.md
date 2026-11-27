# Capstone Part 3 – Intelligent Mobility Solution

## Project Overview

This project extends the traffic analytics workflow developed in Parts 1 and 2 by applying Machine Learning, Deep Learning, Explainable AI, MLOps, and Responsible AI practices to create an intelligent mobility solution.

Using the Metro Interstate Traffic Volume dataset, the project develops predictive models, discovers hidden traffic patterns, implements a deep learning traffic forecasting solution, applies explainability techniques, tracks experiments using MLflow, simulates deployment and monitoring workflows, and generates practical travel recommendations.

---

## Important Note About Accident Data

No accident dataset was provided for this capstone.

To complete the classification task, a documented proxy accident-risk label was created.

### Proxy Accident-Risk Definition

A record is classified as high risk when:

- Traffic congestion is classified as **High** or **Severe**
- AND
- Weather conditions are severe or visibility is low

Implementation:

```python
high_congestion = df["congestion_category"].isin(
    ["High", "Severe"]
)

risky_weather = (
    df["weather_main"].isin(SEVERE_WEATHER)
    |
    (df["is_low_visibility"] == 1)
)

df["high_risk"] = (
    high_congestion &
    risky_weather
).astype(int)
```

### Disclaimer

The proxy label should **not** be interpreted as an actual accident prediction.

It is only used to demonstrate a supervised classification workflow in the absence of a real accident dataset.

---

# Repository Structure

```text
capstone_part3/
│
├── data/
│   └── traffic_ml_ready.csv
│
├── figures/
│   └── shap_summary_plot.png
│
├── logs/
│   ├── task1_supervised_learning.log
│   ├── task2_unsupervised_learning.log
│   ├── task3_deep_learning.log
│   ├── task4_mlflow_experiment_tracking.log
│   ├── task5_traffic_recommendation_system.log
│   ├── task6_model_versioning.log
│   ├── task6_monitoring.log
│   └── task6_flask_api.log
│
├── mlruns/
│
├── results/
│   ├── classification_model_comparison.csv
│   ├── regression_model_comparison.csv
│   ├── kmeans_cluster_summary.csv
│   ├── association_rules.csv
│   ├── lstm_results.csv
│   ├── shap_feature_importance.csv
│   ├── mlflow_experiment_summary.csv
│   ├── traffic_recommendations.csv
│   ├── model_versions.csv
│   ├── monitoring_report.csv
│   └── drift_alert_status.csv
│
├── utils/
│   └── logger_config.py
│
├── task1_supervised_learning.py
├── task2_unsupervised_learning.py
├── task3_deep_learning_explainability.py
├── task4_mlflow_experiment_tracking.py
├── task5_traffic_recommendation_system.py
├── task6_model_versioning.py
├── task6_monitoring.py
├── task6_flask_api.py
│
├── Bias_and_Fairness_Report.pdf
├── Final_Capstone_Report.pdf
│
└── README.md
```

---

# Technologies Used

## Programming Language

- Python 3.x

## Machine Learning

- Scikit-learn
- TensorFlow / Keras

## Data Processing

- Pandas
- NumPy

## Explainable AI

- SHAP

## Unsupervised Learning

- K-Means Clustering
- Association Rule Mining
- MLXtend

## MLOps

- MLflow

## Deployment

- Flask

## Visualisation

- Matplotlib

## Logging

- Python logging module

---

# Task 1 – Supervised Machine Learning

Two predictive problems were addressed:

## Classification

Target:

```text
high_risk
```

Models:

- Logistic Regression
- Random Forest Classifier

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC

Outputs:

```text
results/classification_model_comparison.csv
```

---

## Regression

Target:

```text
traffic_volume
```

Models:

- Linear Regression
- Random Forest Regressor

Evaluation Metrics:

- MAE
- R²

Outputs:

```text
results/regression_model_comparison.csv
```

---

# Task 2 – Unsupervised Machine Learning

## K-Means Clustering

Traffic conditions were clustered using:

- Hour
- Weather Severity
- Traffic Volume

Output:

```text
results/kmeans_cluster_summary.csv
```

---

## Association Rule Mining

Traffic patterns were analysed using:

- Time of Day
- Day Type
- Weather Category
- Congestion Category

Output:

```text
results/association_rules.csv
```

---

# Task 3 – Deep Learning with Explainability

## LSTM Traffic Forecasting

An LSTM neural network was implemented to predict traffic volume.

Target:

```text
traffic_volume
```

Evaluation Metrics:

- MAE
- R²

Output:

```text
results/lstm_results.csv
```

---

## SHAP Explainability

SHAP was applied using a Random Forest surrogate model trained on the same prediction objective.

Outputs:

```text
results/shap_feature_importance.csv

figures/shap_summary_plot.png
```

---

# Task 4 – Advanced AI Technique

## Selected Technique

MLflow Experiment Tracking

MLflow was used to:

- Track experiments
- Record parameters
- Store evaluation metrics
- Compare model runs

Artifacts:

```text
mlruns/

mlflow.db

results/mlflow_experiment_summary.csv
```

---

# Task 5 – Traffic Recommendation System

The recommendation engine converts model outputs and traffic insights into practical travel recommendations.

Features:

- Identifies low-traffic periods
- Avoids high-risk conditions
- Considers day type
- Generates plain-language travel advice

Example:

```text
For a weekday journey, consider travelling
between 03:00 and 04:00, when historical
traffic volumes are typically lower and risk
conditions are minimal.
```

Output:

```text
results/traffic_recommendations.csv
```

---

# Task 6 – MLOps and Deployment Simulation

## Model Versioning

Models were documented and compared.

Output:

```text
results/model_versions.csv
```

---

## Experiment Tracking

MLflow tracks:

- Parameters
- Metrics
- Model versions
- Experiment runs

---

## Deployment Simulation

A Flask API was developed.

Run:

```bash
python task6_flask_api.py
```

Example Request:

```text
http://127.0.0.1:5001/predict?hour=17&weather=Rain
```

Example Response:

```json
{
  "hour": 17,
  "weather": "Rain",
  "predicted_congestion": "High"
}
```

---

## Monitoring

Feature drift monitoring was simulated.

Output:

```text
results/monitoring_report.csv
```

---

## Alerting

System status:

```text
PASS
```

or

```text
ALERT
```

Output:

```text
results/drift_alert_status.csv
```

---

# Task 7 – Responsible and Sustainable AI

A dedicated bias and fairness report is included.

Topics covered:

## Bias and Fairness

- Dataset coverage limitations
- Proxy accident-risk labels
- Uneven prediction performance
- Weather and temporal bias

## Governance

- Human oversight requirements
- Model validation processes
- Monitoring and accountability

## Sustainability

- Computational cost of models
- Energy consumption considerations
- Accuracy versus resource trade-offs

Deliverable:

```text
Bias_and_Fairness_Report.pdf
```

---

# Logging

All scripts implement Python's logging framework.

Each module uses:

```python
logger = logging.getLogger(__name__)
```

Logging Levels:

- INFO
- WARNING
- ERROR

Log Files:

```text
logs/
```

Each task generates its own dedicated log file.

---

# How to Run the Project

## Install Dependencies

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install tensorflow
pip install shap
pip install mlxtend
pip install mlflow
pip install flask
pip install matplotlib
```

---

## Task 1

```bash
python task1_supervised_learning.py
```

---

## Task 2

```bash
python task2_unsupervised_learning.py
```

---

## Task 3

```bash
python task3_deep_learning_explainability.py
```

---

## Task 4

```bash
python task4_mlflow_experiment_tracking.py
```

Optional:

```bash
python -m mlflow ui
```

---

## Task 5

```bash
python task5_traffic_recommendation_system.py
```

---

## Task 6

Model Versioning:

```bash
python task6_model_versioning.py
```

Monitoring:

```bash
python task6_monitoring.py
```

Deployment:

```bash
python task6_flask_api.py
```

---

# Key Findings

- Random Forest models outperformed baseline linear models.
- Traffic volume follows strong daily and weekly patterns.
- Adverse weather conditions contribute to higher traffic risk.
- LSTM models successfully captured sequential traffic behaviour.
- SHAP identified the most influential features affecting traffic predictions.
- MLflow supported reproducibility and experiment management.
- Practical travel recommendations can be generated using historical traffic patterns.

---

# Limitations

- No real accident dataset was available.
- Proxy labels were required for classification.
- The dataset represents a single traffic corridor.
- Models are based on historical data rather than real-time traffic conditions.
- Recommendations do not account for accidents, roadworks, or special events occurring after data collection.

---

# Author

Capstone Part 3 – Intelligent Mobility Solution

Machine Learning, Deep Learning, MLOps, Recommendation Systems, and Responsible AI for Smart-City Traffic Analytics.