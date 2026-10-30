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

**Observations:**

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Demonstrats high AUC (0.8944), in classification it means our model has a strong ability to distinguish between positive and negative classes across all possible decision thresholds |
| Decision Tree | Achieved the highest overall accuracy (90.7%) and balanced precision/recall, easily capturing the complex, non-linear feature interactions in patient data without requiring scaling. |
| KNN | Poorly performed across all models. Struggled with the high dimensionality (32 features) of the dataset, yielding the lowest predictive performance across all metrics. |
| Naive Bayes | Performed well (AUC 0.8999), performance is close to base Logistic regression. |
| Support Vector Machine | Matched Naive Bayes in accuracy (83.26%) but required significant computational overhead. |
| **Best model for the dataset given?** | **Decision Tree Classifier**, yielding the highest Accuracy (90.7%), F1 Score (0.8658), and MCC (0.7953). |
