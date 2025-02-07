import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd

# Load available models dynamically
model_files = [f for f in os.listdir() if f.endswith(".pkl")]

st.title("ML Model Predictor")

# Dropdown to select the model
selected_model_file = st.selectbox("Choose a model:", model_files)


# Load selected model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


pipeline = load_model(selected_model_file)

st.write(f"Loaded model: **{selected_model_file}**")

# --- Create input fields dynamically ---
# Example feature names (adjust based on your dataset)
feature_names = [
    "STATE_FIPS",
    "EVENT_TYPE",
    "CZ_FIPS",
    "BEGIN_RANGE",
    "BEGIN_AZIMUTH",
    "SEASONS",
]

features_choices = {
    "EVENT_TYPE": ["Thunderstorm Wind", "Hail"],
    "BEGIN_AZIMUTH": [
        "E",
        "SE",
        "ENE",
        "NW",
        "WSW",
        "N",
        "WNW",
        "SW",
        "W",
        "S",
        "SSE",
        "ESE",
        "SSW",
        "NNW",
        "NE",
        "NNE",
        "WSWNW",
    ],
    "SEASONS": ["Summer", "Spring", "Fall", "Winter"],
}

feature_inputs = {}
for feature in feature_names:
    if feature in features_choices:
        # If the feature has predefined choices, use a dropdown
        feature_inputs[feature] = st.selectbox(f"{feature}:", features_choices[feature])
    elif feature == "BEGIN_RANGE":
        feature_inputs[feature] = st.number_input(f"{feature}:", value=3.0)
        # Otherwise, let the user input a numerical value
    else:
        feature_inputs[feature] = st.number_input(f"{feature}:", value=10)

# Convert to Pandas DataFrame
input_df = pd.DataFrame([feature_inputs])
# Display DataFrame in Streamlit
st.write("Selected Feature Values:")
st.dataframe(input_df)


# --- Prediction ---
if st.button("Predict"):
    prediction = pipeline.predict(input_df)
    st.write(f"Prediction: **{prediction[0]}**")
