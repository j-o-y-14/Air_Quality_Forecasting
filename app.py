import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk

# =========================
# Load Model & Preprocessor
# =========================
model = joblib.load("gb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# =========================
# Location Data
# =========================
location_df = pd.DataFrame({
    "location_number": [7, 18, 27, 28, 29, 30, 31, 34, 37, 41, 51, 59, 60, 61, 63, 67, 68, 69, 70],
    "location_name": [
        "Downtown", "Airport", "Industrial Area", "Parkside", "Uptown",
        "Harbor", "University", "Suburb East", "Suburb West", "Market",
        "Station", "Mall", "Residential North", "Residential South",
        "Hilltop", "Lakeview", "Riverside", "Old Town", "New Town"
    ],
    "latitude": [-1.290, -1.295, -1.285, -1.292, -1.298, -1.290, -1.287, -1.300, -1.280, -1.275,
                 -1.270, -1.265, -1.262, -1.265, -1.260, -1.250, -1.376, -1.245, -1.240],
    "longitude": [36.777, 36.780, 36.785, 36.790, 36.795, 36.777, 36.780, 36.785,
                  36.790, 36.795, 36.800, 36.805, 36.810, 36.815, 36.820,
                  36.825, 36.929, 36.830, 36.835]
})

# Mapping: Name -> location code
location_options = {row["location_name"]: f"location_{row['location_number']}" for _, row in location_df.iterrows()}

# Store prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []

# =========================
# Streamlit UI
# =========================
st.title("🌍 Air Quality Prediction App")
st.write("Enter environmental conditions to predict if the air quality is Healthy or Unhealthy.")

# Input fields
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50)
temperature = st.slider("Temperature (°C)", min_value=-10, max_value=50, value=25)
hour = st.slider("Hour of the Day", 0, 23, 12)
location_name = st.selectbox("Select Location for Prediction", options=list(location_options.keys()))
location = location_options[location_name]

# =========================
# Prepare Input Data & Predict
# =========================
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "humidity": humidity,
        "temperature": temperature,
        "hour": hour,
        "location": location
    }])

    # Align columns with preprocessor
    input_data = input_data.reindex(columns=preprocessor.feature_names_in_, fill_value=0)

    # Transform
    input_processed = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(input_processed)[0]

    # Output
    if prediction == 1:
        st.success("✅ Air Quality is **Healthy**")
        pred_label = "Healthy"
        color = [0, 255, 0]
    else:
        st.error("⚠️ Air Quality is **Unhealthy**")
        pred_label = "Unhealthy"
        color = [255, 0, 0]

    # Save to history
    st.session_state["history"].append({
        "Humidity": humidity,
        "Temperature": temperature,
        "Hour": hour,
        "Location": location_name,
        "Prediction": pred_label
    })

    # =========================
    # Map Visualization
    # =========================
    st.subheader("📍 Sensor Locations Map")

    # Update map colors
    location_df["color"] = location_df["location_name"].apply(
        lambda x: color if x == location_name else [150, 150, 150]
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=location_df,
        get_position='[longitude, latitude]',
        get_color='color',
        get_radius=200,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=location_df["latitude"].mean(),
        longitude=location_df["longitude"].mean(),
        zoom=11
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{location_name}"}
    )

    st.pydeck_chart(r)

# =========================
# Prediction History
# =========================
if st.session_state["history"]:
    st.subheader("📊 Prediction History")
    hist_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(hist_df)

    # Download CSV button
    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Prediction History as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv")






