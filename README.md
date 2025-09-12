## Air Quality Forecasting using Machine Learning
### Overview

This project builds a machine learning model to predict air quality status (Good vs Poor) using environmental features like temperature, humidity, and location. We applied data preprocessing, class balancing, model training, hyperparameter tuning, and deployed the solution in a simple Streamlit app for end-user interaction.

## 1. Problem Statement

Air pollution poses significant health risks. Forecasting air quality can help citizens, policymakers, and environmental agencies take timely action.

Goal: Predict whether air quality will be Good or Poor.

Type of Task: Classification.

Beneficiaries: Local communities, environmental agencies, healthcare organizations.

## 2. Data Collection & Understanding
### Dataset Overview

Source: Public dataset from Kaggle on air quality monitoring.

Shape: ~719,000 rows × 8 columns.

Granularity: Each row represents a single sensor measurement at a given timestamp and location.

Columns:

sensor_id → Unique ID of the sensor device.

sensor_type → Type of pollutant/measurement (e.g., PM2.5, PM10, temperature, humidity).

location → Numeric identifier for monitoring station (~70+ unique).

lat / lon → Geographical coordinates of the station.

timestamp → Date and time of the measurement.

value_type → Category of measurement (e.g., “P1”, “P2”, “humidity”, “temperature”).

value → Recorded measurement (continuous).
.

## 3. Data Preprocessing

Missing Values: Imputed using median strategy.

Feature Scaling:

StandardScaler for humidity.

RobustScaler for temperature (less sensitive to outliers).

Categorical Encoding: One-Hot Encoding for location.

Class Imbalance: Handled with SMOTE (Synthetic Minority Oversampling Technique).

Train-Test Split: 80/20 stratified split.

## 4. Modeling

We tested multiple algorithms (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting).

Chosen Model: Gradient Boosting Classifier (best F1-score).

Hyperparameter Tuning: Performed with RandomizedSearchCV (learning_rate, n_estimators, max_depth, etc.).

## 5. Evaluation

Metrics Used: Accuracy, Precision, Recall, F1-score.

Confusion Matrix: Shows misclassification distribution.

Learning Curves: Checked for bias/variance tradeoff.

Validation Curves: Analyzed model sensitivity to hyperparameters (n_estimators, max_depth, learning_rate).

Key Results

Training Accuracy: ~99%

Test Accuracy: ~99%

However, recall for the minority class (“Poor” Air Quality) is lower, indicating the class imbalance challenge.

## 6. Error Analysis

The model performs very well on the “Good” class but struggles with rare “Poor” cases.

Errors mainly occur in locations with very few “Poor” readings.

## Possible solutions:

Collect more “Poor” air quality samples.

Apply anomaly detection models.

Use cost-sensitive learning to penalize misclassifying “Poor” cases.

## 7. Model Interpretation

Feature Importance:

Temperature and humidity are the strongest predictors.

Certain locations (urban hotspots) contribute more to predictions.

Business Meaning:

Higher temperature + high particulate readings correlate with poor air quality.

Helps policymakers focus on pollution-heavy locations.

## 8. Deployment

A Streamlit web app was developed to make predictions.
Features:

User inputs temperature and humidity.

Predicts whether air quality is Good or Poor.

Displays probability score (e.g., “There’s a 78% chance of Poor Air Quality”).

CSV upload supported for batch predictions.


## 9. Conclusion & Future Work

✅ Achieved high accuracy but need to improve recall for Poor Air Quality cases.

✅ Built an interpretable Gradient Boosting model.

✅ Deployed using Streamlit for practical use.

### Future Work:

Collect more balanced data (especially Poor cases).

Try ensemble models (XGBoost, LightGBM, CatBoost).

Deploy as an API for integration into dashboards or IoT sensors.


