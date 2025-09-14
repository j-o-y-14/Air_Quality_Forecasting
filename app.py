import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk

# =========================
# Load Model & Preprocessor
# =========================
model = joblib.load("gradient_boosting_model.pkl")
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
    "latitude": [-1.290, -1.295, -1.285, -1.292, -1.298, -1.290, -1.287, -1.300, -1.280, -1.275, -1.270, -1.265, -1.262, -1.265, -1.260, -1.250, -1.376, -1.245, -1.240],
    "longitude": [36.777, 36.780, 36.785, 36.790, 36.795, 36.777, 36.780, 36.785, 36.790, 36.795, 36.800, 36.805, 36.810, 36.815, 36.820, 36.825, 36.929, 36.830, 36.835]
})

# Mapping: Name -> location code
location_options = {row["location_name"]: f"location_{row['location_number']}" for _, row in location_df.iterrows()}

# =========================
# Streamlit UI
# =========================
st.title("🌍 Air Quality Prediction App")
st.write("Enter environmental conditions to predict if the air quality is Healthy or Unhealthy.")

# Input fields
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
hour = st.slider("Hour of the Day", 0, 23, 12)
location_name = st.selectbox("Select Location", options=list(location_options.keys()))
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

    # Ensure input_data has correct columns and order
    input_data = input_data.reindex(columns=preprocessor.feature_names_in_, fill_value=0)

    # Transform using preprocessor
    input_processed = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(input_processed)[0]

    # Output
    if prediction == 1:
        st.success("✅ Air Quality is **Healthy**")
    else:
        st.error("⚠️ Air Quality is **Unhealthy**")

# =========================
# Location Guide Dropdown (Bottom)
# =========================
st.subheader("📌 Location Guide")
selected_location = st.selectbox(
    "Select a location to see details",
    options=location_df["location_name"]
)

# Show details for the selected location
loc_details = location_df[location_df["location_name"] == selected_location][
    ["location_number", "location_name", "latitude", "longitude"]
].rename(columns={
    "location_number": "Location Code",
    "location_name": "Location Name",
    "latitude": "Latitude",
    "longitude": "Longitude"
})

st.table(loc_details)

# =========================
# Map Visualization
# =========================
st.subheader("📍 Sensor Locations Map")

# Color: selected → green, others → gray
location_df["color"] = location_df["location_name"].apply(
    lambda x: [0, 255, 0] if x == location_name else [150, 150, 150]
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
















