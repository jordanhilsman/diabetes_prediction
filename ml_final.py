import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn.metrics.tests 

"""
For the purposes of this project, the target column Diabetes_012 is mapped:
    0: Healthy
    1: Pre-Diabetic
    2: Diabetes
"""

df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

#From the value_counts, we see that 213703 do not have diabetes, 4631 are pre-diabetic, and 35346 have diabetes.

#This is probably considered class imbalance, so we should do undersampling or something to address it.

X = df.drop(columns=['Diabetes_012'])

y = df['Diabetes_012']

print(df.corr())

corr = df.corr()

for index, row in corr.iterrows():
    threshold = 0.5
    print(row.values())
    if any in row > threshold:
        print("True")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

models = [KNeighborsClassifier(n_neighbors=5), GaussianNB(), 
        RandomForestClassifier()]

for model in models:
    print(f"Fitting {model}")
#    model.fit(X_train, y_train)
#    predictions = model.predict(X_test)
#    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    print(f"Score for {model}: \n Accuracy: {accuracy:.2f} \n Precision: {precision:.2f} \n F1: {f1:.2f} \n Recall: {recall:.2f} \n")

