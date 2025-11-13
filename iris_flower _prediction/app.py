import streamlit as st
import joblib
import numpy as np


# 1️⃣ Load the trained Iris model

MODEL_PATH = "/content/iris_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = joblib.load(file)
    return model

model = load_model()


# 2️⃣ Streamlit App UI

st.set_page_config(page_title="🌸 Iris Flower Classifier", layout="centered")

st.title("🌼 Iris Flower Species Prediction")
st.write("Enter the measurements below and click **Predict** to classify the Iris flower species.")

# Feature input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Combine features into array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])


#3️ Prediction
if st.button("Predict"):
    try:
        # Assuming the model outputs a numerical label that needs to be mapped back to species name
        # The original notebook encoded species as 0, 1, 2. Let's assume 0: Iris-setosa, 1: Iris-versicolor, 2: Iris-virginica
        species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
        prediction_label = model.predict(input_data)[0]
        predicted_species = species_map.get(prediction_label, "Unknown Species")
        st.success(f"🌺 Predicted Species: **{predicted_species}**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

st.markdown("---")
st.markdown("🚀 *Deployed with Streamlit*")
