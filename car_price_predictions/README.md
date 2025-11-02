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

## **Final Model & Results**

* **Model:** Random Forest Regressor (n\_estimators=200)  
* **R² Score:** **0.96**  
* **RMSE:** **0.959**

The feature importance plot shows that Present\_Price is by far the most significant predictor of a car's selling price, followed by Car\_Age and Driven\_kms.

## **How to Run**

1. Ensure you have the required libraries installed:  
   pip install \-r requirements.txt

2. Make sure the car data.csv file is in this folder.  
3. Launch Jupyter and open the car\_price\_prediction.ipynb notebook. The notebook will also generate the car\_price\_model.pkl file.