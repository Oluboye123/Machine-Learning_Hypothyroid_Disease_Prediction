
# 🩺 Thyroid Disease Prediction using Machine Learning

## 📌 Project Overview
This project applies **machine learning techniques** to predict and classify **hypothyroid disease** based on medical datasets. The study explores **feature selection techniques** and **classification models**, comparing their performance in diagnosing thyroid conditions. The goal is to improve early detection and assist medical professionals in providing accurate diagnoses.

## 🚀 Key Features
- **Feature Selection Methods**: 
  - Recursive Feature Elimination (RFE)
  - Univariate Feature Selection (UFS)
  - Feature Importance (FI)
  - Principal Component Analysis (PCA)
- **Machine Learning Models Used**:
  - Support Vector Machine (SVM)
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Artificial Neural Network (ANN)
- **Dataset**: Kaggle thyroid disease dataset with **3,772 patient records** and **30 features**.
- **Performance Metrics**: Accuracy, Confusion Matrix, Model Comparison.

---

## 🔍 Data Preprocessing
- **Handling Missing Values**: Features with over 91% missing values were removed. Mean and mode imputation were used for remaining missing data.
- **Feature Encoding**: Categorical variables were converted into numerical values.
- **Normalization**: Standard scaling was applied to numerical data for consistency.
- **Final Dataset**: **2,291 rows** and **8 selected features**.

---

## 📊 Exploratory Data Analysis (EDA)
- **Distribution of positive thyroid cases by age**.
- **Gender distribution** of affected individuals.
- **Sick vs. healthy patient comparison**.
- **Statistical visualization** of key numerical features.

---

## 🏆 Feature Selection & Model Performance
| Feature Selection Method  | SVM Accuracy | Random Forest Accuracy | Logistic Regression Accuracy | KNN Accuracy |
|--------------------------|-------------|------------------------|-----------------------------|-------------|
| **Feature Importance**   | **99.05%**  | 98.91%                 | 98.91%                      | 94.29%      |
| **Univariate Selection** |
