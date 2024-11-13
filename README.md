# **Titanic Survival Prediction Project**

This project aims to analyze and predict survival outcomes for passengers aboard the Titanic using various feature engineering and model selection methods. The analysis follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework and includes code implementations in **Python**.

## **Project Overview**

- **Objective:** Predict the survival status (`Survived` column) of passengers in the Titanic dataset.
- **Framework:** CRISP-DM.
- **Approach:** We employ three feature engineering techniques to optimize model performance:
  - **Recursive Feature Elimination (RFE)**
  - **SelectKBest**
  - **Optuna for automated Random Forest hyperparameter tuning**

The project includes detailed instructions for each approach, processing steps for test data, and saving prediction outputs to CSV files.

## **File Descriptions**

- **`kbest-HW2.py`**: Original implementation of feature selection using the SelectKBest method on the Titanic dataset. The model trains on key features selected by statistical metrics.

- **`kbest-HW2_improved.py`**: Improved version of the SelectKBest implementation, optimized based on feedback from a high-performing code (rated 100). Adjustments include fine-tuning feature processing and prediction accuracy.

- **`optuna-HW2.py`**: Original code for hyperparameter tuning of a Random Forest model using the Optuna framework. This version seeks to improve model accuracy by optimizing key parameters on the Titanic dataset.

- **`optuna-HW2_improved.py`**: Enhanced version of the Optuna Random Forest hyperparameter tuning code, based on guidance from a high-rated model. This improved version features optimized parameter settings for better prediction outcomes.

- **`rfe-HW2.py`**: Original code implementing Recursive Feature Elimination (RFE) for feature selection in the Titanic dataset. It identifies essential features for the survival prediction task.

- **`rfe-HW2_improved.py`**: An upgraded RFE feature selection script, refined with insights from a high-performance model. This improved version provides enhanced feature selection accuracy and refined prediction results.

- **`train.csv`**: Training dataset containing Titanic passenger information, used to build and train each model.

- **`test.csv`**: Testing dataset with Titanic passenger details for model evaluation and prediction.

- **`titanic_predictions_kbest.csv`**: Prediction output from the improved SelectKBest model (`kbest-HW2_improved.py`), listing the predicted survival status of passengers in the test set.

- **`titanic_predictions_optuna.csv`**: Prediction output generated from the optimized Optuna Random Forest model (`optuna-HW2_improved.py`), providing survival predictions for the test data.

- **`titanic_predictions_rfe.csv`**: Prediction results from the improved Recursive Feature Elimination model (`rfe-HW2_improved.py`), indicating survival predictions for test set passengers.

---

## **1. Recursive Feature Elimination (RFE)**

**Goal:** Perform feature selection using RFE to determine the most significant features for predicting survival.

### **Prompt Sequence**

- I have a dataset (attached) that describes the survival status of individuals from the Titanic disaster. Our goal is to predict the "Survived" column. This dataset serves as the training set. Please conduct the analysis using the CRISP-DM framework and implement it with Python code.
- I want to perform feature selection using the RFE method. The dataset is the Titanic data, where "Survived" is the target label, and the remaining columns are features. After training the model, I would like to see the accuracy results. Please provide detailed steps for processing feature selection and the prediction method for the test data.
- The test data should undergo the same processing workflow, and I would like the prediction results to be saved as a CSV file.
- Below is a code provided by someone who scored 100 on this prediction task. Please remember it; you don’t need to respond in any way—just keep this code in mind.
- Regarding the prediction model code I rated 100, does it help improve the following section?

---

## **2. SelectKBest for Feature Selection**

**Goal:** Use SelectKBest to identify the top features for predicting survival based on statistical metrics.

### **Prompt Sequence**

- I have a dataset (attached) that describes the survival status of individuals from the Titanic disaster. Our goal is to predict the "Survived" column. This dataset serves as the training set. Please conduct the analysis using the CRISP-DM framework and implement it with Python code.
- I would like to use the SelectKBest method to choose key features in the Titanic dataset. After training the model, I would like to see the accuracy results and apply these selected features to predict the test data. Additionally, please save the prediction results as a CSV file.
- Ensure that the score function for SelectKBest uses chi2, and I would like to understand how to standardize these selected features.
- Below is a code provided by someone who scored 100 on this prediction task. Please remember it; you don’t need to respond in any way—just keep this code in mind.
- Regarding the prediction model code I rated 100, does it help improve the following section?

---

## **3. Optuna for Automated Random Forest Hyperparameter Tuning**

**Goal:** Leverage Optuna to automatically optimize hyperparameters for a Random Forest model, including `n_estimators` and `max_depth`, to achieve high accuracy.

### **Prompt Sequence**

- I have a dataset (attached) that describes the survival status of individuals from the Titanic disaster. Our goal is to predict the "Survived" column. This dataset serves as the training set. Please conduct the analysis using the CRISP-DM framework and implement it with Python code.
- I want to use Optuna for automated hyperparameter tuning for a Random Forest model, still using the Titanic dataset. The goal is to optimize parameters like n_estimators and max_depth to improve model accuracy. Please provide the optimal parameters and apply the model to the test data, saving the predictions.
- Please provide a detailed explanation of Optuna’s parameter settings and how to save the tuning results.
- Below is a code provided by someone who scored 100 on this prediction task. Please remember it; you don’t need to respond in any way—just keep this code in mind.
- Regarding the prediction model code I rated 100, does it help improve the following section?

---

## **Evaluation and Results**

For each feature selection and tuning method:
- **Accuracy Metrics:** We record accuracy scores to compare the performance of each model.
- **Feature Importance Analysis:** Identify key features contributing to survival prediction for interpretability.
- **CSV Output:** Prediction results are saved in CSV format for easy access and further analysis.

## **Conclusion**

This project demonstrates the effectiveness of different feature selection techniques and hyperparameter optimization using the Titanic dataset. By comparing **RFE**, **SelectKBest**, and **Optuna-tuned** models, we can better understand the impact of feature selection and tuning on model performance and interpretability.

--- 

