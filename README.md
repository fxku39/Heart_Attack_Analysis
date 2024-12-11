# Heart Attack Prediction using Machine Learning

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Data Preproccesing](#data-preprocessing)
- [Results and Findings](#results-and-findings)
- [Conclusion](#conclusion)

## Project Overview
The goal of this project is to develop a machine learning model to predict whether a patient is at risk of a heart attack based on various health features. The model uses a dataset containing medical information from patients, such as age, blood pressure, cholesterol levels, chest pain type, and more, to classify whether a patient is likely to have a heart attack (1) or not (0).

## Data Sources
The dataset used in this project is publicly available on Kaggle. You can access it [here](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset). The dataset consists of the following features:

- **Age**: Age of the patient
- **Sex**: Sex of the patient (1 = male, 0 = female)
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **ca**: Number of major vessels (0-3)
- **cp**: Chest pain type
  - 1: Typical angina
  - 2: Atypical angina
  - 3: Non-anginal pain
  - 4: Asymptomatic
- **trtbps**: Resting blood pressure (in mm Hg)
- **chol**: Cholesterol level in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **rest_ecg**: Resting electrocardiographic results
  - 0: Normal
  - 1: ST-T wave abnormality
  - 2: Left ventricular hypertrophy
- **thalach**: Maximum heart rate achieved
- **output**: Whether the patient is at risk for a heart attack (1 = risk, 0 = no risk)

## Data Preprocessing
- Missing values were handled by dropping rows with missing data in the `exang` and `cp` columns, as they only represented 3 rows and did not significantly affect the dataset. For `trtbps` and `chol`, missing values were replaced with the mean of each respective column.
- The dataset was divided into categorical and continuous variables, with `output` as the target variable.
- A Chi-Square test (`chi2_contingency`) was used for continuous variables to identify the strongest relationships with the target variable, revealing significant relationships with `sex`, `chest pain type`, `exang`, `ca`, and `thalach`.
- An ANOVA test was performed for categorical variables to analyze their relationship with the target variable.

## Models Used
The following machine learning models were tested for predicting heart attack risk:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Decision Tree Classifier**

## Model Evaluation
The models were evaluated based on:
- **Accuracy**: To measure the overall performance.
- **Confusion Matrix**: To evaluate the classification results in more detail.
- **GridSearchCV**: To tune hyperparameters and improve model performance.

**KNN with default parameters**
![image](https://github.com/user-attachments/assets/bfe2bc91-c268-42bd-b180-1609ad0f7ef0)

## Results and Findings
The models produced similar accuracy scores, but the best performing model was **K-Nearest Neighbors (KNN)** with an accuracy of **0.8444**.

![image](https://github.com/user-attachments/assets/0231151e-6359-4a85-9337-cfa01bd7e856)

## Technologies Used
- **Python**
- **Scikit-Learn**: For machine learning algorithms and model evaluation.
- **Pandas**: For data manipulation and preprocessing.
- **Numpy**: For numerical operations.
- **Matplotlib** & **Seaborn**: For data visualization.

## How to Run the Project
1. Clone or download the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Conclusion
This project demonstrates how machine learning models can be applied to medical datasets to predict the likelihood of a heart attack based on patient health data. The KNN model achieved the highest accuracy and can be used to make predictions on new patient data.
