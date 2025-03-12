import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model/logistic_regression_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

st.title("Credit Card Default Prediction")
st.write("Enter values in the correct order, separated by commas.")

# User input
user_input = st.text_area("Input (comma-separated):", "")

if st.button("Predict"):
    try:
        input_values = np.array([list(map(float, user_input.split(",")))])
        if input_values.shape[1] != len(feature_names):
            st.error(f"Expected {len(feature_names)} values, but got {input_values.shape[1]}.")
        else:
            input_scaled = scaler.transform(input_values)
            prediction = model.predict(input_scaled)[0]
            confidence = model.predict_proba(input_scaled)[0][prediction] * 100

            if prediction == 1:
                st.error(f"⚠️ Likely to Default (Confidence: {confidence:.2f}%)")
            else:
                st.success(f"✅ No Default Expected (Confidence: {confidence:.2f}%)")
    except:
        st.error("Invalid input format. Please enter numeric values separated by commas.")
