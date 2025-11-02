# OIBSIP-
📈 3 Machine Learning Projects: Iris Species Classification (SVC), Used Car Price Prediction (Random Forest), and Ad Sales Prediction (Regression).
Machine Learning Projects 
Welcome! This repository contains a collection of three distinct machine learning projects. Each project is self-contained in its own folder with its own notebook and dataset.
Table of Contents
Project 1: Iris Flower Species Classification
Project 2: Used Car Price Prediction
Project 3: Advertising Sales Prediction
How to Use
Technologies Used
Project 1: Iris Flower Species Classification
Notebook: iris-classification/task_1 (1).ipynb
Dataset: iris-classification/Iris.csv
Goal
To classify Iris flowers into one of three species (Iris-setosa, Iris-versicolor, Iris-virginica) based on their sepal and petal length/width.
Workflow
Data Loading & Inspection: Loaded the Iris.csv dataset and performed initial analysis (head, describe, info, value_counts).
Exploratory Data Analysis (EDA): Visualized the data using histograms, scatter plots (colored by species), a pair plot, and a correlation heatmap.
Data Preprocessing:
Dropped the unnecessary Id column.
Used LabelEncoder to convert the categorical Species target variable into numerical values (0, 1, 2).
Baseline Model Training: Split the data (70/30) and trained five baseline classification models.
Data Cleaning: Identified and removed four outliers from the SepalWidthCm feature using the Interquartile Range (IQR) method.
Final Model Training: Re-split the cleaned data and re-trained the top-performing models (SVC, Logistic Regression, KNN).
Final Model & Results
After cleaning the data, both the Support Vector Classifier (SVC) and Logistic Regression models achieved 100% accuracy on the test set.
Project 2: Used Car Price Prediction
Notebook: car-price-prediction/car.ipynb
Dataset: car-price-prediction/car data.csv
Model Output: car-price-prediction/car_price_model.pkl
Goal
To predict the selling price of used cars using a regression model based on their features.
Workflow
Data Loading & Inspection: Loaded the car data.csv dataset and performed initial checks (head, shape, info, describe).
Feature Engineering: Converted the Year column into a more useful Car_Age feature.
Data Preprocessing: Created a preprocessing pipeline using ColumnTransformer to OneHotEncode the categorical features.
Model Training: Trained a RandomForestRegressor within an sklearn.pipeline.Pipeline.
Evaluation & Visualization: Evaluated the model using R² (0.96) and RMSE (0.959). Plotted feature importances, which identified Present_Price as the strongest predictor.
Model Saving: The final trained model was saved to car_price_model.pkl using joblib.
Final Model & Results
Model: Random Forest Regressor
R² Score: 0.96
RMSE: 0.959
Project 3: Advertising Sales Prediction
Notebook: sales-prediction/Sales_prediction.ipynb
Dataset: sales-prediction/Advertising.csv
Goal
To predict product sales based on advertising spend across three different media channels: TV, Radio, and Newspaper.
Workflow
Data Loading & Inspection: Loaded the Advertising.csv dataset and performed standard checks (head, info,describe, isnull, duplicated).
Exploratory Data Analysis (EDA): Visualized the data using histograms, boxplots, scatter plots, and a correlation heatmap, which showed a strong positive correlation between TV spend and Sales.
Model Training & Comparison:
Trained a LinearRegression model as a baseline.
Trained a RandomForestRegressor for comparison.
Evaluation: Compared the performance of both models using R² and MSE.
Final Model & Results
The Random Forest Regressor provided a much better fit for the data than the simple linear model.
Model
R² Score (Test Set)
MSE (Test Set)
Random Forest (Best)
0.981
0.59
Linear Regression
0.899
3.17

How to Use
To run these projects on your local machine:
Clone the repository:
git clone <your-repo-url>


Navigate to a project folder:
cd car-price-prediction


Install requirements:
pip install -r requirements.txt

(Each project folder has its own requirements.txt).
Launch Jupyter:
jupyter notebook


Open the .ipynb file in that folder to explore the project.
Technologies Used
Python
Jupyter Notebook
Pandas
NumPy
Scikit-learn (sklearn)
Matplotlib
Seaborn
Joblib
