import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("🔥 ML Model Prediction App")
st.write("Enter the feature values below and click Predict.")

# Automatically adjust input boxes based on model input size
# Example assumes model was trained on 4 features
num_features = 4
inputs = []
for i in range(num_features):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

if st.button("Predict"):
    # Convert to numpy array
    features = np.array([inputs])

    # Apply scaling if scaler exists
    try:
        features = scaler.transform(features)
    except AttributeError:
        pass   # skip scaling if scaler is just an array

    # Make prediction
    prediction = model.predict(features)

    st.success(f"🔥 Model Prediction: {prediction[0]}")   