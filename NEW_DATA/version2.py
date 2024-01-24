import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class CrimePredictor:
    def __init__(self):
        # Turn off warning messages
        pd.options.mode.chained_assignment = None
        # Load the data
        self.df = pd.read_csv('crime.csv')
        self.feature_names = ['Age', 'DrugTest', 'Obedient', 'EmotionScore', 'ConfidenceScore', 'ConsistencyScore', 'Gender']
        self.training_features = self.df[self.feature_names]
        self.outcome_name = ['Criminal']
        self.outcome_labels = self.df[self.outcome_name]
        # Numeric and categorical feature names
        self.numeric_feature_names = ['Age', 'EmotionScore', 'ConfidenceScore', 'ConsistencyScore']
        self.categorical_feature_names = ['DrugTest', 'Obedient', 'Gender']
        # Initialize the scaler
        self.ss = StandardScaler()
        # Fit scaler on numeric features
        self.ss.fit(self.training_features[self.numeric_feature_names])
        # Scale numeric features
        self.training_features[self.numeric_feature_names] = self.ss.transform(self.training_features[self.numeric_feature_names])
        # Engineering categorical training features (one-hot encoding)
        self.training_features = pd.get_dummies(self.training_features, columns=self.categorical_feature_names)
        # Get a list of new categorical features
        self.categorical_engineered_features = list(set(self.training_features.columns) - set(self.numeric_feature_names))
        # Model initialization and training
        self.lr = LogisticRegression()
        self.model = self.lr.fit(self.training_features, self.outcome_labels.values.ravel())  # Use .ravel() to flatten the outcome_labels
        # Save the model and scaler
        if not os.path.exists('Model'):
            os.mkdir('Model')
        if not os.path.exists('Scaler'):
            os.mkdir('Scaler')
        joblib.dump(self.model, 'Model/model.pickle')
        joblib.dump(self.ss, 'Scaler/scaler.pickle')

    def predict_criminal(self, name, age, drug_test, obedient, emotion_score, confidence_score, consistency_score, gender):
        # Load the model and scalar objects
        self.model = joblib.load('Model/model.pickle')
        self.scaler = joblib.load('Scaler/scaler.pickle')
        # Prepare new data
        new_data = pd.DataFrame([{
            'Name': name,
            'Age': age,
            'DrugTest': drug_test,
            'Obedient': obedient,
            'EmotionScore': emotion_score,
            'ConfidenceScore': confidence_score,
            'ConsistencyScore': consistency_score,
            'Gender': gender
        }])
        # Data preparation
        prediction_features = new_data.copy()
        # Scaling for numeric features
        prediction_features[self.numeric_feature_names] = self.scaler.transform(prediction_features[self.numeric_feature_names])
        # Engineering categorical variables (one-hot encoding)
        prediction_features = pd.get_dummies(prediction_features, columns=['DrugTest', 'Obedient', 'Gender'])
        # Ensure the order and presence of categorical features
        for feature in self.categorical_feature_names:
            if feature not in prediction_features.columns:
                prediction_features[feature] = 0  # Add missing categorical feature columns with 0 values
        # Reorder columns to match the training data
        prediction_features = prediction_features[self.training_features.columns]
        # Predict using the model
        predictions = self.model.predict(prediction_features)
        new_data['Criminal'] = predictions
        return new_data

# Usage:
if __name__ == "__main__":
    predictor = CrimePredictor()
    name = input("Enter your name: ")
    age = int(input("Enter your age: "))
    drug_test = input("Have you ever taken any drug test (Yes/No): ")
    obedient = input("Are you obedient (Yes/No): ")
    emotion_score = int(input("What is your emotion score (0 - 100): "))
    confidence_score = int(input("What is your confidence score (0 - 100): "))
    consistency_score = int(input("What is your consistency score (0 - 100): "))
    gender = input("Enter your gender (Male or Female): ")
    result = predictor.predict_criminal(name, age, drug_test, obedient, emotion_score, confidence_score, consistency_score, gender)
    print(result)
