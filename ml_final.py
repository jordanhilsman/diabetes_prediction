import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn.metrics.tests 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel


"""
For the purposes of this project, the target column Diabetes_012 is mapped:
    0: Healthy
    1: Pre-Diabetic
    2: Diabetes
"""

df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

dups = df[df.duplicated()]
print("Number of dropped duplicate rows: ", len(dups))

df.drop_duplicates(inplace=True)

#From the value_counts, we see that 213703 do not have diabetes, 4631 are pre-diabetic, and 35346 have diabetes.

#This is probably considered class imbalance, so we should do undersampling or something to address it.


X = df.drop(columns=['Diabetes_012'])

y = df['Diabetes_012']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

standardizer = StandardScaler().fit(X_train)

X_train_st = standardizer.transform(X_train)
X_test_st = standardizer.transform(X_test)

selector = SelectFromModel(RandomForestClassifier(), threshold="median")
selector.fit(X_train_st, y_train)

X_train_select = selector.transform(X_train_st)
X_test_select = selector.transform(X_test_st)

models = [KNeighborsClassifier(n_neighbors=5), GaussianNB(), 
        RandomForestClassifier()]

for model in models:
    print(f"Fitting {model}")
    model.fit(X_train_select, y_train)
    predictions = model.predict(X_test_select)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    print(f"Normalized Score for {model}: \n Accuracy: {accuracy:.2f} \n Precision: {precision:.2f} \n F1: {f1:.2f} \n Recall: {recall:.2f} \n")

