import pandas as pd

# Create a DataFrame with the crime data
data = {
    'Name': ['Michael', 'Maureen', 'Gordon', 'Borniface', 'Dennis', 'Derrick', 'Emma', 'Nelly', 'Tom'],
    'Age': ['R', 'Y', 'N', 'R', 'N', 'R', 'Y', 'Y', 'A'],
    'DrugTest': ['Y', 'N', 'N', 'N', 'N', 'Y', 'N', 'N', 'Y'],
    'Obedient': ['N', 'N', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'N'],
    'EmotionsScore': [90, 85, 10, 75, 20, 92, 60, 75, 90],
    'ConfidenceScore': [85, 51, 17, 71, 30, 79, 59, 33, 78],
    'ConsistencyScore': [50, 67, 81, 45, 73, 21, 59, 69, 88],
    'Gender': ['M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M'],
    'Crime': ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']  # Placeholder for crime detection
}
print(data)
df = pd.DataFrame(data)

# Define features and outcome
feature_names = ['Age', 'DrugTest', 'Obedient', 'EmotionsScore', 'ConfidenceScore', 'ConsistencyScore', 'Gender']
training_features = df[feature_names]
outcome_name = 'Crime'
outcome_labels = df[outcome_name]

# Update the Age column based on the provided criteria
df['Age'] = df['Age'].apply(lambda x: 'Y' if x == 'R' else ('N' if x == 'A' else 'N'))

# Perform one-hot encoding for categorical features
categorical_feature_names = ['Age', 'DrugTest', 'Obedient', 'Gender']
training_features = pd.get_dummies(training_features, columns=categorical_feature_names)

# Define numeric and categorical features
numeric_feature_names = ['EmotionsScore', 'ConfidenceScore', 'ConsistencyScore']
categorical_engineered_features = list(set(training_features.columns) - set(numeric_feature_names))

# Model building (adapted from your original code)
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create a Logistic Regression model
lr = LogisticRegression()
model = lr.fit(training_features, np.array(outcome_labels))

# Print the model parameters
print(model)

# Simple evaluation on training data (adapted from your original code)
pred_labels = model.predict(training_features)
actual_labels = np.array(outcome_labels)

# Evaluate the model performance
from sklearn.metrics import accuracy_score, classification_report

print('Accuracy: ', float(accuracy_score(actual_labels, pred_labels)) * 100, '%')
print('Classification Stats:')
print(classification_report(actual_labels, pred_labels))

# Save the model and scaler (adapted from your original code)
import joblib
import os

if not os.path.exists('Model'):
    os.mkdir('Model')

joblib.dump(model, r'Model/model.pickle')