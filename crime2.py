import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
import os

class Crime:
           
    def __init__(self):
        self.new_data = None
        self.prediction = None
        self.numeric_feature = None
        self.categorical_feature = None 
        self.training_features = None
        self.predictions = None
        self.model = None
        self.scaler = None
        self.actual_labels = None
        self.actual_labels = None
        pd.options.mode.chained_assignment = None
        df = pd.read_csv('crime.csv')
        self.feature_names = ['Age','DrugTest','Obedient','EmotionScore','ConfidenceScore','ConsistencyScore','Gender']
        self.training_features = df[self.feature_names]
        outcome_name = ['Criminal']
        outcome_labels =df[outcome_name]
        numeric_feature_names = ['Age','EmotionScore','ConfidenceScore','ConsistencyScore']
        categorical_feature_names = ['DrugTest','Obedient','Gender']
        ss = StandardScaler()
        ss.fit(self.training_features[numeric_feature_names])
        self.training_features[numeric_feature_names] = ss.transform(self.training_features[numeric_feature_names])
        self.training_features = pd.get_dummies(self.training_features,columns=categorical_feature_names)
        categorical_engineered_features = list(set(self.training_features.columns)-set(numeric_feature_names))
        lr = LogisticRegression()
        self.model = lr.fit(self.training_features,np.array(outcome_labels['Criminal']))
        self.pred_labels = self.model.predict(self.training_features)
        self.actual_labels = np.array(outcome_labels['Criminal'])
        if not os.path.exists('Model'):
            os.mkdir('Model')
        if not os.path.exists('Scaler'):
            os.mkdir('Scaler')
        joblib.dump(self.model,r'Model/model.pickle')
        joblib.dump(ss,r'Scaler/scaler.pickle')
        self.model = joblib.load(r'Model/model.pickle')
        self.scaler = joblib.load(r'Scaler/scaler.pickle')

    
    def accuracy(self):
        print('Accuracy: ',float(accuracy_score(self.actual_labels,self.pred_labels))*100,'%')
        print('Classification Stats: ')
        print(classification_report(self.actual_labels,self.pred_labels))
        
        
    #Data retrieval    
    def data(self,name,age,drug_test,obedient,emotion_score,confidence_score,consistency_score,gender):
        self.new_data = pd.DataFrame([{'Name': Name,
                           'Age': Age,
                           'DrugTest': DrugTest,
                           'Obedient': Obedient,
                           'EmotionScore': EmotionScore,
                           'ConfidenceScore': ConfidenceScore,
                           'ConsistencyScore': ConsistencyScore,
                           'Gender': Gender
                           }])
    
    #Data preparation
    def prepare(self):
        self.prediction = self.new_data.copy()
        self.numeric_feature = ['Age','EmotionScore','ConfidenceScore','ConsistencyScore']
        self.categorical_feature = ['DrugTest_Yes','DrugTest_No','Obedient_Yes','Obedient_No','Gender_Male','Gender_Female']
        self.prediction[self.numeric_feature] = self.scaler.transform(self.prediction[self.numeric_feature])
        self.prediction= pd.get_dummies(self.prediction,columns=['DrugTest','Obedient','Gender'])
        for feature in self.categorical_feature:
            if feature not in self.prediction.columns:
                self.prediction[feature] = 0 #Add missing categorical feature columns with 0 columns
        self.prediction = self.prediction[self.training_features.columns]
        self.predictions = self.model.predict(self.prediction)
        self.new_data['Criminal'] = self.predictions
        print(self.new_data)


crime1 = Crime()
#Now get inputs from the user
Name = input("Enter your name: ")
Age = int(input("Enter your age: "))
DrugTest =input("Have you ever taken any drug test (Yes/No): ")
Obedient = input("Are you obedient (Yes/No): ")
EmotionScore = int(input("What is your emotion score (0 - 100): "))
ConfidenceScore = int(input("What is your confidence score (0 - 100): "))
ConsistencyScore = int(input("What is your consitency score (0 - 100): "))
Gender = input("Enter your gender (Male or Female): ")
crime1.data(Name,Age,DrugTest,Obedient,EmotionScore,ConfidenceScore,ConsistencyScore,Gender)
crime1.accuracy()
crime1.prepare()        