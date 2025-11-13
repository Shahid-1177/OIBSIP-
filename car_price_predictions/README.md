# **Used Car Price Prediction**

This project uses a Random Forest Regressor to predict the selling price of used cars based on their features.

## **Project Goal**

To build a regression model that accurately estimates the Selling\_Price of a used car. The final trained model is saved as car\_price\_model.pkl.

## **Workflow**

1. **Data Loading & Inspection:** The car data.csv dataset is loaded and inspected using head, shape, info, and describe to identify data types and check for missing values.  
2. **Feature Engineering:** The Year column is converted into a Car\_Age feature (Current Year \- Year) to provide a more intuitive metric for the model.  
3. **Data Preprocessing:**  
   * **Columns Dropped:** Car\_Name (irrelevant for prediction), Year (replaced by Car\_Age), and Owner   
   * **Categorical Features:** Fuel\_Type, Selling\_type, and Transmission are identified for one-hot encoding.  
   * **Numerical Features:** Present\_Price, Driven\_kms, and Car\_Age are identified.  
4. **Model Pipeline:**  
   * An sklearn.pipeline.Pipeline is created to streamline the entire process.  
   * A ColumnTransformer is used inside the pipeline to apply OneHotEncoder (with drop='first') to the categorical features and allow numerical features to passthrough.  
   * The preprocessed data is then fed into a RandomForestRegressor (n\_estimators=200).  
5. **Training & Evaluation:**  
   * The data is split into an 80% training set and a 20% testing set.  
   * The model pipeline is trained on the training data.  
   * Performance is evaluated on the test set using R-squared (R²) and Root Mean Squared Error (RMSE).  
6. **Analysis & Export:**  
   * A scatter plot is created to compare actual vs. predicted selling prices.  
   * Feature importances are extracted from the trained Random Forest model and visualized.  
   * The final trained model pipeline is saved to disk as car\_price\_model.pkl using joblib.
   * The project includes a basic Streamlit app for interactive prediction demonstration.

## **Final Model & Results**

* **Model:** Random Forest Regressor (n\_estimators=200)  
* **R² Score:** **0.96**  
* **RMSE:** **0.959**

The feature importance plot shows that Present\_Price is by far the most significant predictor of a car's selling price, followed by Car\_Age and Driven\_kms.


## 💾 Model Saving

The trained model was serialized using **Joblib**:

```python
import joblib, os
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/best_rf_model.joblib")
```

This allows reloading without retraining.

---
## 🌐 Deployment using Streamlit & Ngrok

### app.py
```python
import streamlit as st
import joblib
import numpy as np

model = joblib.load('artifacts/best_rf_model.joblib')

st.title("🚗 Car Price Prediction App")
st.write("Enter car details to estimate selling price.")

present_price = st.number_input("Present Price (in lakhs)", 0.0)
kms_driven = st.number_input("Kms Driven", 0)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
car_age = st.slider("Car Age (in years)", 0, 20)

if st.button("Predict Price"):
    input_data = np.array([[present_price, kms_driven, fuel_type, seller_type, transmission, car_age]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Selling Price: ₹{prediction[0]:.2f} lakhs")
```

### Ngrok Tunnel Setup
```python
from pyngrok import ngrok
!ngrok authtoken "YOUR_NGROK_AUTH_TOKEN"

public_url = ngrok.connect(8501)
print("Public URL:", public_url)
!streamlit run app.py --server.port 8501
```

Access your app using the printed **public ngrok URL**.

---


## **How to Run**

1. Ensure you have the required libraries installed:  
   pip install \-r requirements.txt

2. Make sure the car data.csv file is in this folder.  
3. Launch Jupyter and open the car\_price\_prediction.ipynb notebook. The notebook will also generate the car\_price\_model.pkl file.
