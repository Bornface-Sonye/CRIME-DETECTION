import pandas as pd
#turn off warning messages
pd.options.mode.chained_assignment = None

#get the data
df = pd.read_csv('crime.csv')
print(df)

#get the features and their corresponding outcomes
feature_names = ['Age','DrugTest','Obedient','EmotionScore','ConfidenceScore','ConsistencyScore','Gender']
training_features = df[feature_names]

outcome_name = ['Criminal']
outcome_labels =df[outcome_name]

#View training features
print(training_features)

#view outcome labels
print(outcome_labels)

#list down features based on data type
numeric_feature_names = ['Age','EmotionScore','ConfidenceScore','ConsistencyScore']
categorical_feature_names = ['DrugTest','Obedient','Gender']

#Numeric feature scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

#Fit scaler on numeric features
ss.fit(training_features[numeric_feature_names])

#Scale numeric features now
training_features[numeric_feature_names] = ss.transform(training_features[numeric_feature_names])

#view updated feature set
print(training_features)

#Engineering categorical training features also called one hot encoding
training_features = pd.get_dummies(training_features,columns=categorical_feature_names)

#View training features after perfroming one hot encoding
print(training_features)

#Get a list of new categorical features
categorical_engineered_features = list(set(training_features.columns)-set(numeric_feature_names))
print(categorical_engineered_features)

#Modeling
from sklearn.linear_model import LogisticRegression
import numpy as np

#fit the model
lr = LogisticRegression()
model = lr.fit(training_features,np.array(outcome_labels['Criminal']))

#view model parameters
print(model)

#Simple evaluation on training data
pred_labels = model.predict(training_features)
actual_labels = np.array(outcome_labels['Criminal'])

#Evaluate the performance of the model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print('Accuracy: ',float(accuracy_score(actual_labels,pred_labels))*100,'%')
print('Classification Stats: ')
print(classification_report(actual_labels,pred_labels))

import joblib
import os

#Save the model to be deployed on your server
if not os.path.exists('Model'):
    os.mkdir('Model')
if not os.path.exists('Scaler'):
    os.mkdir('Scaler')

joblib.dump(model,r'Model/model.pickle')
joblib.dump(ss,r'Scaler/scaler.pickle')


#Prediction in action
#load the model and the scalar objects
model = joblib.load(r'Model/model.pickle')
scaler = joblib.load(r'Scaler/scaler.pickle')

#Now get inputs from the user
Name = input("Enter your name: ")
Age = int(input("Enter your age: "))
DrugTest =input("Have you ever taken any drug test (Yes/No): ")
Obedient = input("Are you obedient (Yes/No): ")
EmotionScore = int(input("What is your emotion score (0 - 100): "))
ConfidenceScore = int(input("What is your confidence score (0 - 100): "))
ConsistencyScore = int(input("What is your consitency score (0 - 100): "))
Gender = input("Enter your gender (Male or Female): ")

#Data retrieval
new_data = pd.DataFrame([{'Name': Name,
                           'Age': Age,
                           'DrugTest': DrugTest,
                           'Obedient': Obedient,
                           'EmotionScore': EmotionScore,
                           'ConfidenceScore': ConfidenceScore,
                           'ConsistencyScore': ConsistencyScore,
                           'Gender': Gender
                           }])

#Data preparation
prediction_features = new_data.copy()

#Ensure order and presence of numeric features
numeric_feature_names = ['Age','EmotionScore','ConfidenceScore','ConsistencyScore']
categorical_feature_names = ['DrugTest_Yes','DrugTest_No','Obedient_Yes','Obedient_No','Gender_Male','Gender_Female']

#Scaling for numeric features
prediction_features[numeric_feature_names] = scaler.transform(prediction_features[numeric_feature_names])

#Engineering categorical variables (one-hot encoding)
prediction_features = pd.get_dummies(prediction_features,columns=['DrugTest','Obedient','Gender'])

#Ensure the order and presence of categorical features
for feature in categorical_feature_names:
    if feature not in prediction_features.columns:
        prediction_features[feature] = 0 #Add missing categorical feature columns with 0 columns
        
#Reorder columns to match the training data
prediction_features = prediction_features[training_features.columns]

#Predict using the model
predictions = model.predict(prediction_features)

#Display the results
new_data['Criminal'] = predictions
print(new_data)
