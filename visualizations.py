import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

df.drop_duplicates(inplace=True)

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True)
plt.title("Feature correlation")
plt.show()

plt.pie(df['Diabetes_012'].value_counts(), labels=['Non Diabetic', 'Prediabetic', 'Diabetic'], autopct='%0.2f')
plt.show()

low_bmi = df[df['BMI'] <= 20]
medium_bmi = df.iloc[np.where((df['BMI']>20) & (df['BMI']<=50))] 
