import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn.metrics.tests 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.under_sampling import NearMiss
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
"""
For the purposes of this project, the target column Diabetes_012 is mapped:
    0: Healthy
    1: Pre-Diabetic
    2: Diabetes
"""

df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

#df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')


dups = df[df.duplicated()]
print("Number of dropped duplicate rows: ", len(dups))

df.drop_duplicates(inplace=True)

#From the value_counts, we see that 213703 do not have diabetes, 4631 are pre-diabetic, and 35346 have diabetes.

#This is probably considered class imbalance, so we should do undersampling or something to address it.

nm = NearMiss(version=1, n_neighbors=6)




X = df.drop(columns=['Diabetes_012'])

y = df['Diabetes_012']


#X = df.drop(columns=['Diabetes_binary'])

#y = df['Diabetes_binary']


X_under, y_under = nm.fit_resample(X,y)


models = [KNeighborsClassifier(n_neighbors=6), GaussianNB(), 
        RandomForestClassifier()]

scoring_metrics = ['accuracy', 'precision_weighted', 'f1_weighted', 'roc_auc_ovr', 'recall_weighted']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)



for model in models:
    print(f"Evaluating {model} with 5-fold Cross Validation.")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=15)),
        ('model', model)
        ])

    y_pred = cross_val_predict(pipeline, X_under, y_under,  cv=cv)

    for metric in scoring_metrics:
        scores = cross_val_score(pipeline, X_under, y_under, cv=cv, scoring = metric)
        print(f"{metric.capitalize()}: {scores.mean():.2f}")

    cm = confusion_matrix(y_under, y_pred, labels = np.unique(y_under)
                          )
    disp = ConfusionMatrixDisplay(confusion_matrix
                                  = cm, display_labels = np.unique( y_under)
                                  )
    disp.plot()
    plt.title(f"Confusion Matrix for {model}")
    plt.show()
