# Air_Quality_Forecasting
Air Quality Classification
# Problem Statement

Air pollution (PM2.5, NO₂) harms health, causing respiratory and cardiovascular diseases.

People need clear AQI categories (Good, Moderate, Unhealthy, Hazardous) instead of raw pollutant numbers.

Lack of timely forecasts limits prevention and response.

Goal: Use ML classification to predict AQI categories for early warnings and better public health decisions.

# Overview

This project applies machine learning classification models to predict Air Quality Index (AQI) categories (Good, Moderate, Unhealthy, Hazardous) based on real-world pollutant data.

We use the OpenAQ API to collect air quality measurements (PM2.5, NO₂) and transform them into health-relevant classes. The goal is to provide an early-warning tool for citizens and health authorities.

 # Objectives

Collect real-time air quality data (PM2.5, NO₂) using the OpenAQ API.

Engineer features (time of day, lags, rolling averages).

Train classification models (Logistic Regression, Random Forest, Gradient Boosting).

Evaluate performance with Accuracy, Precision, Recall, F1-score.

Provide insights into which features influence AQI most.

 # Data Sources

 OpenAQ API
 → global open-source air quality data.

Pollutants:

PM2.5 → fine inhalable particles (≤2.5 μm).

NO₂ → Nitrogen Dioxide from vehicles & industry.

(Optional) Weather features: temperature, humidity, wind speed.

# Methods

Data Collection → pull hourly/daily PM2.5 & NO₂ values from OpenAQ.

Preprocessing →

Handle missing values (linear interpolation).

Resample to consistent time steps.

Create features: lag values, rolling averages, time encoding.

Convert pollutant readings into AQI categories.

# Modeling →

Logistic Regression (baseline).

Decision Tree, Random Forest.

Gradient Boosted Trees (optional).

Handle class imbalance (SMOTE / class weights).

Evaluation →

Metrics: Accuracy, Precision, Recall, F1-score.

Confusion Matrix → shows which AQI categories are misclassified.

Feature Importance → interpret model decisions.

 

# Future Work

Add weather forecast data for better predictions.

Expand to more pollutants (O₃, CO, SO₂).

Build a Streamlit dashboard for real-time city-level alerts.

Extend to multiple cities (Nairobi, Kampala, London).

# Tech Stack

Python (pandas, numpy, scikit-learn, matplotlib, seaborn).

OpenAQ API → data source.

ML Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.

Visualization: matplotlib, seaborn.

(Optional) Streamlit for dashboard deployment.
