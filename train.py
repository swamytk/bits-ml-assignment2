####### Alzheimers Disease Prediction #######

# os and file system access
import joblib
import os
# data processing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# models libraries
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# for reproducibility, the seed like value is defined and used
my_seed = 42
# training vs test data split ratio
test_percentage = 0.2 # 20%

# Load dataset
df = pd.read_csv('alzheimers_disease_data.csv')
# Drop all the identifiers which are not relevant for prediction
df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')
# Split the features and target columns
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=my_seed)

# Save test data for Streamlit upload
test_data = X_test.copy()
test_data['Diagnosis'] = y_test
test_data.to_csv('test_data.csv', index=False)

# Initialize models within scaling pipelines
# pipeline =>  to chain multiple data preprocessing transformers and 
#             an estimator (model) into a single, cohesive workflow
models = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=my_seed)),
    "decision_tree": make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=my_seed)),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "naive_bayes": make_pipeline(StandardScaler(), GaussianNB()),
    "svm": make_pipeline(StandardScaler(), SVC(probability=True, random_state=my_seed)) 
}

# create model directory
os.makedirs('model', exist_ok=True)
# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'model/model_{name}.pkl')
    print(f"model_{name} trained and saved.")
