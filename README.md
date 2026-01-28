#  Non-Linear Value Prediction using Hybrid SVR & Residual Learning

##  Overview
This project implements a machine learning pipeline to predict numerical target values from a dataset containing mixed data types (categorical, numerical, and high-cardinality features). 

Instead of relying on a single model, this solution utilizes a **Hybrid Residual Learning strategy**: a Support Vector Regressor (SVR) captures the main non-linear trends, while a Ridge Regression model is trained specifically on the *residuals* (errors) of the SVR to correct bias and improve final accuracy.

##  Key Features
* **Advanced Preprocessing:**
    * **Imputation:** Utilizes `KNNImputer` to estimate missing values based on nearest feature neighbors.
    * **Feature Engineering:** Custom Binary Encoding for high-cardinality features (bit-level representation).
    * **Outlier Handling:** IQR-based Winsorization to cap extreme values in numerical features.
* **Hybrid Modeling Architecture:**
    * **Base Model:** SVR (RBF Kernel) for capturing non-linear relationships.
    * **Residual Correction:** Ridge Regression trained on $(y_{true} - y_{pred})$ to capture patterns missed by the SVR.
* **Feature Selection:** ANOVA (f_regression) statistical testing to filter non-significant predictors.

##  Technologies Used
* **Language:** Python 3.x
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `xgboost`, `matplotlib` (for analysis)


