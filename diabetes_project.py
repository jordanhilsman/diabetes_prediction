from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo


# Get dataset using the library!

cdc_dataset = fetch_ucirepo(id=891)

X = cdc_dataset.data.features
y = cdc_dataset.data.targets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1_337)
print(X.corr())
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train, y_train)
models = [LogisticRegression(), KNeighborsClassifier(n_neighbors=5), GaussianNB(), RandomForestClassifier(),
        SVC(kernel="rbf")]

for model in models:
    print(f"Finding Coefs for {model}.")
    model.fit(X_train, y_train['Diabetes_binary'])
    try:
        importance = model.coef_
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v[0]))
    except Exception as e:
        print(f"No coefficients for {model}.")
