Credit Risk Analysis with Logistic Regression
This project analyzes credit risk using a logistic regression model applied to a dataset of lending data. The analysis includes data preprocessing, model training, evaluation, and a comprehensive review of the results.

Table of Contents
Introduction

Data Description

Project Setup

Data Preprocessing

Model Training and Evaluation

Results and Analysis

Conclusion

Acknowledgements

Introduction
This project aims to build and evaluate a logistic regression model to predict credit risk. The dataset consists of lending data, including various features related to loan status. The goal is to determine the effectiveness of the model in identifying high-risk loans.

Data Description
The dataset lending_data.csv contains various attributes of loans. The key target variable is loan_status, which indicates whether a loan is healthy (0) or high-risk (1). The dataset includes numerical and categorical features related to the borrowers and the loans.

Project Setup
Clone the repository:

bash
git clone <repository-url>
cd <repository-directory>
Install required dependencies:

bash
pip install -r requirements.txt
Load the dataset:

python
import pandas as pd
df = pd.read_csv("Resources/lending_data.csv")
Data Preprocessing
Separate Labels and Features:

python
y = df["loan_status"]
X = df.drop(columns=["loan_status"])
Split Data into Training and Testing Sets:

python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Model Training and Evaluation
Train Logistic Regression Model:

python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=1)
model.fit(X_train, y_train)
Make Predictions and Evaluate Model:

python
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(conf_matrix)
print(class_report)
Results and Analysis
The logistic regression model achieves high accuracy in predicting both healthy and high-risk loans. The classification report provides detailed metrics for each class, indicating strong performance with a slight trade-off in precision and recall for high-risk loans.

Confusion Matrix:

[[TP  FP]
 [FN  TN]]
Classification Report:

Precision    Recall    F1-Score    Support
0            1.00      0.99      1.00       18751
1            0.85      0.90      0.87         633
Conclusion
The logistic regression model is highly effective in predicting credit risk, with overall accuracy of 99%. The model performs exceptionally well in identifying healthy loans, while also demonstrating strong performance in detecting high-risk loans.