# Cardio Predict – Heart Disease Risk Prediction App

A simple, educational desktop application that demonstrates machine learning-based heart disease risk assessment using synthetic data.

**Important Disclaimer**:  
This tool is for **educational and demonstration purposes only**.  
It is **NOT** a medical device, does **NOT** provide medical advice, and should **never** be used for real clinical decisions.

---

### Features

- Predicts heart disease risk using two models:
  - Random Forest Classifier
  - Logistic Regression
  - SVM Classifier
  - Gradient Boosting
  - KNN Algorithm
- Ensemble result with easy-to-understand risk levels (LOW / MODERATE / HIGH)
- Clean, modern Tkinter GUI
- Handles categorical and numerical features automatically
- Uses robust preprocessing (missing value imputation, label encoding, robust scaling)
- Trains instantly on realistic synthetic data at startup

---

### Requirements

- Python 3.7+
- Required packages:
  ```bash
  pip install pandas numpy scikit-learn tkinter

### The app will
- Generate synthetic training data
- Train the models in the background
- Open the GUI window

### Output
- Overall risk probability (%)
- Risk Level: LOW, MODERATE, or HIGH
- Basic lifestyle recommendations
