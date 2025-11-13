
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('sales_model.pkl')

st.title("📊 Sales Prediction using Advertising Data")
st.write("Predict product sales based on advertising budget in different media channels.")

# Input features based on Advertising dataset columns
tv = st.number_input("TV Advertising Budget (in thousands)", min_value=0.0)
radio = st.number_input("Radio Advertising Budget (in thousands)", min_value=0.0)
newspaper = st.number_input("Newspaper Advertising Budget (in thousands)", min_value=0.0)

# Prepare data for prediction
features = np.array([[tv, radio, newspaper]])

if st.button("Predict Sales"):
    try:
        prediction = model.predict(features)[0]
        st.success(f"Predicted Sales: {prediction:.2f} units")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
