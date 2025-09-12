import pandas as pd
import joblib

# Load model and preprocessor
loaded_model = joblib.load("gradient_boosting_model.pkl")
loaded_preprocessor = joblib.load("preprocessor.pkl")

# Example new input (must match training feature structure!)
new_data = pd.DataFrame([{
    "humidity": 55,
    "temperature": 27,
    "location_A": 1,
    "location_B": 0,
    "location_C": 0
}])

# Transform with preprocessor
new_data_scaled = loaded_preprocessor.transform(new_data)

# Predict
prediction = loaded_model.predict(new_data_scaled)
print("Prediction:", prediction)
