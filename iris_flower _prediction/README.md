# **Iris Flower Species Classification**

This project uses machine learning to classify iris flowers into one of three species: *Iris-setosa*, *Iris-versicolor*, or *Iris-virginica*.

## **Project Goal**

The objective is to build a model that accurately predicts the species of an iris flower based on four physical features:

* Sepal Length (cm)  
* Sepal Width (cm)  
* Petal Length (cm)  
* Petal Width (cm)

## **Workflow**

1. **Data Loading & Inspection:** The iris.csv file is loaded into a Pandas DataFrame. Initial analysis is performed (head, info, describe, value\_counts) to understand the data's structure, check for missing values, and see the distribution of the three species.
2.  **Exploratory Data Analysis (EDA):**  
   * Histograms and boxplots were created to understand feature distributions and identify outliers.  
   * Scatter plots and a sns.pairplot were used to visualize the relationships between features, colored by species. This revealed that *Iris-setosa* is highly separable from the other two species.  
   * A correlation heatmap was plotted to quantify the relationships between features. 
3. **Data Preprocessing:**  
   * The Id column was dropped.  
   * The categorical Species target variable was converted into numerical labels (0, 1, 2\) using sklearn.preprocessing.LabelEncoder.
4.   **Baseline Model Training:**  
   * The data was split into a 70% training set and a 30% testing set.  
   * Five different classification models were trained and evaluated on their accuracy:  
     * Logistic Regression (93.3% accuracy)  
     * K-Nearest Neighbors (KNN) (93.3% accuracy)  
     * Decision Tree Classifier (91.1% accuracy)  
     * Random Forest Classifier (91.1% accuracy)  
     * Support Vector Classifier (SVC) (95.6% accuracy)
5.   **Data Cleaning & Final Modeling:**  
   * Based on the boxplots from the EDA, four outliers were identified and removed from the SepalWidthCm feature using the 1.5 \* IQR (Interquartile Range) rule.  
   * The cleaned data was re-split, and the top-performing models were re-trained.
  
6. **App for hosting our model**
   * Simple Streamlit application (app.py) for Iris Flower classification.


## **Final Model & Results**

Removing the outliers significantly improved model performance. The final models achieved the following on the test set:

| Model | Accuracy (After Cleaning) |
| :---- | :---- |
| **Support Vector Classifier (SVC)** | **100.0%** |
| **Logistic Regression** | **100.0%** |
| K-Nearest Neighbors (KNN) | 97.8% |

### **Classification Reports (Cleaned Data)**

**Logistic Regression & SVC (Accuracy: 100%)**

              precision    recall  f1-score   support

           0       1.00      1.00      1.00        15  
           1       1.00      1.00      1.00        16  
           2       1.00      1.00      1.00        14

    accuracy                           1.00        45  
   macro avg       1.00      1.00      1.00        45  
weighted avg       1.00      1.00      1.00        45

**K-Nearest Neighbors (Accuracy: 97.8%)**

              precision    recall  f1-score   support

           0       1.00      1.00      1.00        15  
           1       1.00      0.94      0.97        16  
           2       0.93      1.00      0.97        14

    accuracy                           0.98        45  
   macro avg       0.98      0.98      0.98        45  
weighted avg       0.98      0.98      0.98        45

## **How to Run**

1. Ensure you have the required libraries installed:  
   pip install \-r requirements.txt

2. Make sure the Iris.csv file is in this folder.  
3. Launch Jupyter and open the iris\_flower\_prediction.ipynb notebook.
