import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.title("Alzheimer's Disease Classification App")

# Test Dataset upload 
test_file = st.file_uploader("Upload test data CSV file", type="csv")

if test_file is not None:
    data = pd.read_csv(test_file)
    st.write("Test Data Preview:", data.head())
    
    if 'Diagnosis' not in data.columns:
        st.error("The uploaded CSV must contain a 'Diagnosis' column.")
    else:
        X_test = data.drop(columns=['Diagnosis'])
        y_test = data['Diagnosis']
        
        # b. Model selection dropdown
        model_name = st.selectbox(
            "Select a Machine Learning Model",
            ("logistic_regression", "decision_tree", "knn", "naive_bayes", "svm")
        )
        
        # Load the selected model pipeline
        model = joblib.load(f'model/model_{model_name}.pkl')
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # c. Display of evaluation metrics
        st.subheader("Evaluation Metrics")
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC Score": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A",
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "MCC Score": matthews_corrcoef(y_test, y_pred)
        }
        
        metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        st.table(metrics_df)
        
        # d. Confusion matrix or classification report
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
      
