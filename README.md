# Smart City Traffic Capstone

## Project Overview

This repository contains the complete Smart City Traffic Capstone project, covering:

1. Data Analytics and Business Intelligence
2. Python Data Engineering and Automation
3. Machine Learning, Deep Learning, MLOps, and Responsible AI

The project uses the Metro Interstate Traffic Volume dataset to analyse traffic patterns, engineer reusable analytics pipelines, develop predictive machine learning models, implement explainable AI techniques, simulate deployment workflows, and generate intelligent travel recommendations.

---

# Repository Structure

```text
smart-city-traffic-capstone/
│
├── part1_data_analytics/
│
├── part2_python/
│
├── part3_machine_learning/
│
└── README.md
```

---

# Part 1 – Data Analytics

## Objective

Explore and analyse traffic behaviour using SQL, Power BI and statistical techniques.

## Activities

- Traffic trend analysis
- Holiday traffic analysis
- Weather impact analysis
- Correlation analysis
- Descriptive statistics
- Power BI dashboard development
- Smart-city mobility insights

## Key Outputs

- SQL analysis queries
- Power BI dashboards
- Traffic trend visualisations
- Data Analytics Insights Report

---

# Part 2 – Python Data Engineering

## Objective

Develop a reusable traffic analytics pipeline using Python.

## Activities

- Data cleaning
- Exploratory data analysis
- Feature engineering
- Visualisation generation
- Command-line analytics application
- Logging implementation

## Key Outputs

- Cleaned traffic dataset
- Machine-learning-ready dataset
- Visualisations
- Python analytics application

---

# Part 3 – Machine Learning and AI

## Objective

Transform traffic analytics into an intelligent mobility solution using machine learning, deep learning, explainable AI, recommendation systems and MLOps practices.

## Tasks Completed

### Task 1 – Supervised Machine Learning

Classification Models:

- Logistic Regression
- Random Forest Classifier

Regression Models:

- Linear Regression
- Random Forest Regressor

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- MAE
- R²

---

### Task 2 – Unsupervised Machine Learning

- K-Means Clustering
- Association Rule Mining

Outputs:

- Traffic condition clusters
- Congestion pattern rules

---

### Task 3 – Deep Learning with Explainability

Deep Learning Model:

- LSTM Neural Network

Explainability:

- SHAP

Outputs:

- Traffic demand predictions
- SHAP feature importance analysis

---

### Task 4 – Advanced AI Technique

Selected Technique:

- MLflow Experiment Tracking

Capabilities:

- Parameter tracking
- Metric tracking
- Experiment tracking
- Model comparison

---

### Task 5 – Traffic Recommendation System

Features:

- Low-traffic period identification
- Risk-aware recommendations
- Weather-aware recommendations
- Plain-language travel advice

Example:

```text
For a weekday journey, consider travelling
between 03:00 and 04:00 when historical
traffic volumes are typically lower and
risk conditions are minimal.
```

---

### Task 6 – MLOps and Deployment Simulation

Components:

- Model versioning
- Experiment tracking
- Flask API deployment simulation
- Monitoring simulation
- Drift detection
- Alerting

---

### Task 7 – Responsible and Sustainable AI

Topics:

- Bias analysis
- Fairness considerations
- Governance requirements
- Sustainability considerations
- Environmental trade-offs

---

# Important Note About Accident Data

No accident dataset was provided as part of the capstone.

To demonstrate the classification workflow in Part 3, a documented proxy accident-risk label was created using:

- Severe congestion
- Severe weather conditions
- Low-visibility conditions

This proxy label is intended only for educational demonstration purposes and should not be interpreted as a prediction of actual accidents.

---

# Technologies Used

## Data Analytics

- SQL
- Power BI

## Programming

- Python 3.x

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn

## Deep Learning

- TensorFlow
- Keras

## Explainable AI

- SHAP

## Unsupervised Learning

- MLXtend

## MLOps

- MLflow

## Deployment

- Flask

## Visualisation

- Matplotlib

## Logging

- Python Logging Module

---

# Logging

All Python modules use Python's logging framework.

Features include:

- INFO logging
- WARNING logging
- ERROR logging
- File logging
- Console logging

Log files are stored in:

```text
logs/
```

---

# Running the Project

## Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
pip install tensorflow
pip install shap
pip install mlxtend
pip install mlflow
pip install flask
```

---

## Part 2

Run the data pipeline:

```bash
python pipeline.py
```

Run feature engineering:

```bash
python feature_engineering.py
```

---

## Part 3

### Task 1

```bash
python task1_supervised_learning.py
```

### Task 2

```bash
python task2_unsupervised_learning.py
```

### Task 3

```bash
python task3_deep_learning_explainability.py
```

### Task 4

```bash
python task4_mlflow_experiment_tracking.py
```

### Task 5

```bash
python task5_traffic_recommendation_system.py
```

### Task 6

```bash
python task6_model_versioning.py

python task6_monitoring.py

python task6_flask_api.py
```

---

# Key Findings

- Daily and weekly traffic patterns are highly predictable.
- Weather conditions influence traffic volume and congestion.
- Random Forest models outperformed baseline linear models.
- LSTM networks successfully captured sequential traffic behaviour.
- SHAP improved transparency by explaining prediction drivers.
- MLflow supported reproducible machine learning experiments.
- Traffic recommendations can be generated from historical mobility patterns.

---

# Limitations

- Only a single traffic corridor is represented.
- No real accident dataset was available.
- Historical data may not reflect future traffic behaviour.
- Real-time incidents and road closures are not included.
- Recommendation outputs are based on historical patterns rather than live traffic conditions.

---

# Author

Smart City Traffic Capstone

A complete end-to-end analytics, machine learning, MLOps, and responsible AI solution for smart-city traffic analytics and mobility decision support.