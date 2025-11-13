# **Advertising Sales Prediction**

This project analyzes the impact of advertising budgets across different media (TV, Radio, Newspaper) on product sales using a Random Forest Regressor.

## **Project Goal**

The goal is to build a model that accurately predicts Sales based on the advertising spend in the TV, Radio, and Newspaper channels.

## **Workflow**

1. **Data Loading & Inspection:** The Advertising.csv dataset is loaded and inspected using head(), info(), describe(), isnull().sum(), and duplicated().sum() to ensure data quality.  
2. **Data Cleaning:** The extraneous Unnamed: 0 index column is dropped from the DataFrame.  
3. **Exploratory Data Analysis (EDA):**  
   * Histograms and Boxplots are used to check feature distributions and identify outliers (notably in the Newspaper feature).  
   * Scatter plots are created for each channel (TV, Radio, Newspaper) against Sales to visually inspect relationships.  
   * A correlation heatmap is generated (sns.heatmap), which clearly shows a very strong positive correlation (0.78) between TV spend and Sales.  
4. **Model Preparation:**  
   * Features (X) are defined as TV, Radio, and Newspaper. The target (y) is Sales.  
   * The data is split into an 80% training set and a 20% testing set using train\_test\_split (with random\_state=42).  
5. **Model Training:**  
   * A RandomForestRegressor (with n\_estimators=100 and random\_state=42) is trained on the training data.  
6. **Evaluation:**  
   * The trained model is used to make predictions on the test set.  
   * Performance is evaluated using R-squared (R²) and Mean Squared Error (MSE).  
   * A scatter plot of actual vs. predicted sales is created, showing a strong positive fit.
   * The project includes a basic Streamlit app for interactive prediction demonstration.
   * We have a simple Streamlit application (app.py) for live prediction based on user-input marketing budgets.

## **Final Model & Results**

The Random Forest Regressor provided a very strong fit for the data.

* **Model:** Random Forest Regressor  
* **R² Score (Test Set):** **0.981**  
* **MSE (Test Set):** **0.59**

## **How to Run**

1. Ensure you have the required libraries installed:  
   pip install \-r requirements.txt

2. Make sure the Advertising.csv file is in this folder.  
3. Launch Jupyter and open the Sales\_prediction.ipynb notebook to see the full analysis.
