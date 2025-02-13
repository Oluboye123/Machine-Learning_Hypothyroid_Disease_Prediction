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
| **Univariate Selection** | 96.46%      | **99.72%**             | 95.65%                      | 94.56%      |
| **Recursive Elimination** | 97.28%     | 99.72%                 | 96.19%                      | 93.61%      |
| **PCA (on ANN)**         | -           | -                      | -                           | **97.55%**  |

✅ **Best Performing Model**: **SVM with Feature Importance (99.05% accuracy)**  
✅ **Artificial Neural Network (ANN) achieved 97.55% accuracy using PCA**

---

## 🤖 Machine Learning Models Used
1. **Logistic Regression**: Used for binary classification of thyroid disease.
2. **Support Vector Machine (SVM)**: Identified the optimal hyperplane for classification.
3. **K-Nearest Neighbors (KNN)**: Classified new patients based on similarity.
4. **Random Forest**: An ensemble model improving accuracy using multiple decision trees.
5. **Artificial Neural Network (ANN)**: Applied deep learning using PCA for feature reduction.

---

## 🎯 Results & Findings
- **Feature Importance** selection method yielded the **best results (99.05%)**.
- **Males and young adults (18-24) were the most frequently diagnosed**.
- **SVM and Random Forest performed exceptionally well**, outperforming other models.
- **ANN with PCA showed strong predictive capability** (97.55%).

---

## 🔮 Future Enhancements
- Implement **deep learning models** such as CNNs for better feature extraction.
- Apply **ensemble learning techniques** to improve model robustness.
- Expand dataset size for **better generalization** in real-world medical diagnostics.

---

## 💻 Technologies Used
- **Python** (Pandas, NumPy, Scikit-Learn, TensorFlow)
- **Jupyter Notebook**
- **Matplotlib & Seaborn** (Data Visualization)
- **Google Colab** (for model training)
