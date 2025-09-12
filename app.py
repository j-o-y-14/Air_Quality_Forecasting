import streamlit as st
import pandas as pd
import joblib


# Load Model and Preprocessor

model = joblib.load("gradient_boosting_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


# Streamlit UI

st.set_page_config(page_title="Air Quality Predictor", layout="centered")

st.title("🌍 Air Quality Prediction App")
st.markdown("Enter the input values below to predict **Air Quality Status**.")


# Input Form

with st.form("prediction_form"):
    temperature = st.number_input("🌡️ Temperature", value=25.0)
    humidity = st.number_input("💧 Humidity", value=50.0)

    # Example: let user pick location
    location = st.selectbox("📍 Location", [f"location_{i}" for i in range(1, 71)])

    submitted = st.form_submit_button("Predict")


# Make Prediction

if submitted:
    # Build input dataframe
    new_data = pd.DataFrame([[temperature, humidity, location]],
                            columns=["temperature", "humidity", "location"])

    # One-hot encode location column
    new_data_encoded = pd.get_dummies(new_data, columns=["location"])

    # Align with training features
    all_features = preprocessor.feature_names_in_
    new_data_encoded = new_data_encoded.reindex(columns=all_features, fill_value=0)

    # Transform features
    new_data_scaled = preprocessor.transform(new_data_encoded)

    # Predict class and probability
    prediction = model.predict(new_data_scaled)[0]
    prob = model.predict_proba(new_data_scaled)[0]

    # Display results
    st.subheader("🔎 Prediction Result:")
    if prediction == 1:
        st.error(f"❌ Poor Air Quality (Confidence: {prob[1]*100:.2f}%)")
    else:
        st.success(f"✅ Good Air Quality (Confidence: {prob[0]*100:.2f}%)")

    st.write("📊 Probability Scores:")
    st.write({ "Good Air Quality": f"{prob[0]*100:.2f}%", 
               "Poor Air Quality": f"{prob[1]*100:.2f}%" })








