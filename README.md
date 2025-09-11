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

Air Quality Forecasting with Machine Learning
📝 Problem Statement

Air pollution in Nairobi poses serious health risks due to traffic emissions, industrial activities, and rapid urbanization. Traditional monitoring provides limited foresight, making it hard to act proactively. This project tackles the challenge of forecasting air quality using machine learning, enabling timely interventions and informed decision-making.

📌 Project Overview

Goal: Classify air quality into categories (Good, Moderate, Poor)

Task Type: Supervised Classification

Motivation: Accurate air quality prediction helps policymakers, environmental agencies, and the public reduce health risks and take preventive actions.

📂 Dataset

Source: Kaggle – Air Quality in Nairobi
 
Kaggle

Features:

Numerical: PM (Particulate Matter), Temperature, Humidity

Categorical: Wind Direction, Region, Season

Preprocessing Steps:

Handled missing values

One-hot encoded categorical variables

Standardized numerical features

Applied SMOTE to balance imbalanced classes

Train-test split

⚙️ Installation
# Clone the repo
git clone https://github.com/yourusername/air-quality-forecasting.git
cd air-quality-forecasting

# Create environment (optional)
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

🛠️ Usage
# Load model and preprocessor
import joblib
import pandas as pd

model = joblib.load('gradient_boosting_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Example: Predict air quality
sample_input = pd.DataFrame({
    'PM': [35],
    'Temperature': [25],
    'Humidity': [60],
    'WindDirection_N': [1],
    'WindDirection_E': [0],
    # ... include all other features after preprocessing
})

sample_input_transformed = preprocessor.transform(sample_input)
prediction = model.predict(sample_input_transformed)
print("Predicted Air Quality Class:", prediction[0])

🧰 Modeling

Baseline Models: Logistic Regression, Decision Tree

Advanced Models: Random Forest, Gradient Boosting, XGBoost

Best Model: Gradient Boosting with hyperparameter tuning via RandomizedSearchCV

📊 Evaluation

Metrics: Accuracy, Precision, Recall, F1-score

Visualizations:

Confusion Matrix

Feature Importance

Learning Curves

Insights:

PM, Temperature, and Humidity are the most influential features

Extreme pollution levels are harder to predict due to limited data

⚡ Error Analysis

Misclassified cases analyzed via confusion matrix

Higher errors for rare or extreme pollution levels

Recommendations: collect more data, engineer new features, explore alternative models

🚀 Deployment

Deployed with Streamlit for interactive use

Users can input environmental parameters and receive predicted air quality

Saved artifacts:

Model: gradient_boosting_model.pkl

Preprocessor: preprocessor.pkl

📝 Conclusion

This project delivers a complete ML workflow — from preprocessing to deployment — for air quality forecasting in Nairobi. It provides actionable insights, highlights key pollution drivers, and offers a deployable tool for real-time predictions to support public health and policy decisions.

📌 Requirements

Python ≥ 3.8

Libraries: pandas, numpy, scikit-learn, imblearn, xgboost, matplotlib, seaborn, joblib, streamlit
