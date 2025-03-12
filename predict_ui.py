import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load the trained model, scaler, and feature names
model = joblib.load("model/logistic_regression_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

expected_features = len(feature_names)

st.title("💳 Credit Card Default Prediction")
st.write(f"Enter exactly {expected_features} numerical values as a comma-separated list.")

user_input = st.text_input("Input values:", "")

if st.button("Predict"):
    try:
        # Convert input string to list of floats
        input_list = [float(x.strip()) for x in user_input.split(",")]
        if len(input_list) != expected_features:
            st.error(f"Invalid input! Expected {expected_features} features, but got {len(input_list)}.")
        else:
            # Create DataFrame with the proper feature names
            input_df = pd.DataFrame([input_list], columns=feature_names)
            # Scale the input using the same scaler as during training
            input_scaled = scaler.transform(input_df)
            # Get prediction probabilities
            probas = model.predict_proba(input_scaled)[0]
            # Adjust threshold: here, if probability of default (assumed to be at index 1) > 0.3, predict default.
            threshold = 0.3
            prediction = 1 if probas[1] > threshold else 0
            st.write(f"Probability of Default: {probas[1] * 100:.2f}%")
            result = "⚠️ Default Expected" if prediction == 1 else "✅ No Default"
            st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Invalid input! Please enter numerical values in a comma-separated format.\nError: {e}")
