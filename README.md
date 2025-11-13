# 📊 OASIS INFOBYTE Internship – Data Science Projects

This repository contains the projects I developed during my **Data Science Internship** at **Oasis Infobyte (OIBSIP)**.  
Each project demonstrates different machine learning concepts, from classification to regression and deployment.



## **Table of Contents**

1. [Project 1: Iris Flower Species Classification](#bookmark=id.hbwaogt1xhkn)  
2. [Project 2: Used Car Price Prediction](#bookmark=id.nf5h0p40wqu6)  
3. [Project 3: Advertising Sales Prediction](#bookmark=id.69a843vepou1)  
4. [How to Use](#bookmark=id.48ydhdrsvk1w)  
5. [Technologies Used](#bookmark=id.j2ek0c7wpiys)

## **Project 1: Iris Flower Species Classification**

* **Notebook:** iris-classification/iris_flower_prediction.ipynb  
* **Dataset:** iris-classification/iris.csv

### **Goal**

To classify Iris flowers into one of three species (Iris-setosa, Iris-versicolor, Iris-virginica) based on their sepal and petal length/width.

### **Workflow**

1. **Data Loading & Inspection:** Loaded the Iris.csv dataset and performed initial analysis (head, describe, info, value\_counts).  
2. **Exploratory Data Analysis (EDA):** Visualized the data using histograms, scatter plots (colored by species), a pair plot, and a correlation heatmap.  
3. **Data Preprocessing:**  
   * Dropped the unnecessary Id column.  
   * Used LabelEncoder to convert the categorical Species target variable into numerical values (0, 1, 2).  
4. **Baseline Model Training:** Split the data (70/30) and trained five baseline classification models.  
5. **Data Cleaning:** Identified and removed four outliers from the SepalWidthCm feature using the Interquartile Range (IQR) method.  
6. **Final Model Training:** Re-split the cleaned data and re-trained the top-performing models (SVC, Logistic Regression, KNN).
7. simple Streamlit application (app.py) for Iris Flower classification

### **Final Model & Results**

After cleaning the data, both the **Support Vector Classifier (SVC)** and **Logistic Regression** models achieved **100% accuracy** on the test set.

## **Project 2: Used Car Price Prediction**

* **Notebook:** car-price-prediction/car_price_prediction.ipynb  
* **Dataset:** car-price-prediction/car data.csv  
* **Model Output:** car-price-prediction/car\_price\_model.pkl

### **Goal**

To predict the selling price of used cars using a regression model based on their features.

### **Workflow**

1. **Data Loading & Inspection:** Loaded the car data.csv dataset and performed initial checks (head, shape, info, describe).  
2. **Feature Engineering:** Converted the Year column into a more useful Car\_Age feature.  
3. **Data Preprocessing:** Created a preprocessing pipeline using ColumnTransformer to OneHotEncode the categorical features.  
4. **Model Training:** Trained a RandomForestRegressor within an sklearn.pipeline.Pipeline.  
5. **Evaluation & Visualization:** Evaluated the model using R² (0.96) and RMSE (0.959). Plotted feature importances, which identified Present\_Price as the strongest predictor.  
6. **Model Saving:** The final trained model was saved to car\_price\_model.pkl using joblib.
7. Streamlit application (app.py) for live prediction of car price by user input.

### **Final Model & Results**

* **Model:** Random Forest Regressor  
* **R² Score:** **0.96**  
* **RMSE:** **0.959**

## **Project 3: Advertising Sales Prediction**

* **Notebook:** sales-prediction/Sales\_prediction.ipynb  
* **Dataset:** sales-prediction/Advertising.csv

### **Goal**

To predict product sales based on advertising spend across three different media channels: TV, Radio, and Newspaper.

### **Workflow**

1. **Data Loading & Inspection:** Loaded the Advertising.csv dataset and performed standard checks (head, info,describe, isnull, duplicated).  
2. **Exploratory Data Analysis (EDA):** Visualized the data using histograms, boxplots, scatter plots, and a correlation heatmap, which showed a strong positive correlation between TV spend and Sales.  
3. **Model Training :**  
   * Trained model with RandomForestRegressor .
  
4. simple Streamlit application (app.py) for live prediction based on user-input marketing budgets.

### **Final Model & Results**

The **Random Forest Regressor** provided a good fit for the data.

| Model | R² Score (Test Set) | MSE (Test Set) |
| :---- | :---- | :---- |
| **Random Forest (Best)** | **0.981** | **0.59** |



## **How to Use**

To run these projects on your local machine:

1. **Clone the repository:**  
   git clone https://github.com/Shahid-1177/OIBSIP-

2. **Navigate to a project folder:**  
   cd car-price-prediction

3. **Install requirements:**  
   pip install \-r requirements.txt
 
4. **Launch Jupyter:**  
   jupyter notebook

5. Open the .ipynb file in that folder to explore the project.

## **Technologies Used**

* Python  
* Jupyter Notebook  
* Pandas  
* NumPy  
* Scikit-learn (sklearn)  
* Matplotlib  
* Seaborn  
* Joblib
* Streamlit for web application development.
