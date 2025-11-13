import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load Pretrained Model Pipeline
model_path = "artifacts/best_xgb_model.joblib"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please save it first as 'artifacts/best_xgb_model.joblib'.")
else:
    model = joblib.load(model_path)

#  Streamlit App Configuration

st.set_page_config(page_title="Car Price Prediction", layout="centered", page_icon="🚗")

st.markdown("""
    <style>
        .title {text-align: center; color: #0078ff; font-size: 2.5em; font-weight: bold;}
        .footer {text-align: center; color: gray; margin-top: 40px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🚗 Car Price Prediction App</p>', unsafe_allow_html=True)
st.write("Enter car details below to predict its selling price (in lakhs).")


#  Input Fields (must match the training data)
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, max_value=50.0, step=0.1, value=5.0)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

with col2:
    kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, step=500, value=25000)
    selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])


#  Compute Derived Feature (Car_Age)
from datetime import datetime
current_year = datetime.now().year
car_age = current_year - year


#  Prediction

if st.button("Predict Selling Price"):
    if not os.path.exists(model_path):
        st.error("⚠️ Model not found. Please ensure 'artifacts/best_xgb_model.joblib' exists.")
    else:
        try:
            # Prepare input DataFrame to match pipeline training format
            input_data = pd.DataFrame({
                "Present_Price": [present_price],
                "Driven_kms": [kms_driven],
                "Fuel_Type": [fuel_type],
                "Selling_type": [selling_type],
                "Transmission": [transmission],
                "Car_Age": [car_age]
            })

            # Predict using trained pipeline (preprocessing handled inside)
            prediction = model.predict(input_data)[0]

            st.success(f"💰 **Predicted Selling Price:** ₹{prediction:.2f} lakhs")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

st.markdown('<div class="footer">Developed with ❤️ using Streamlit + RandomForest</div>', unsafe_allow_html=True)
