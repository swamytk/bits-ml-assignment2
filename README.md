# ML Assignment 2: Classification Deployment

**a. Problem statement**

To build, evaluate, and deploy multiple machine learning classification models to accurately predict Alzheimer's disease diagnosis based on patient lifestyle, demographic, and medical data, and to present these models via an interactive web application.

**b. Dataset description**

- **Name:** Alzheimer's Disease Dataset

- **Features:** 32 predictive features including physical (BMI, Blood Pressure), lifestyle (Diet, Sleep, Alcohol), and cognitive assessments (MMSE, ADL), after removing identifying columns.

- **Instances:** 2,149 instances

- **Target:** Binary classification (Diagnosis: 0 for Negative, 1 for Positive).

**c. Github Repository Link**
https://github.com/swamytk/bits-ml-assignment2/

**d. Models used:**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8302 | 0.8944 | 0.7899 | 0.7124 | 0.7491 | 0.6232 |
| Decision Tree | 0.9070 | 0.8927 | 0.8897 | 0.8431 | 0.8658 | 0.7953 |
| KNN | 0.7186 | 0.7491 | 0.6429 | 0.4706 | 0.5434 | 0.3558 |
| Naive Bayes | 0.8326 | 0.8999 | 0.7872 | 0.7255 | 0.7551 | 0.6294 |
| Support Vector Machine | 0.8326 | 0.8971 | 0.8000 | 0.7059 | 0.7500 | 0.6276 |


