from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


# Get dataset using the library!

cdc_dataset = fetch_ucirepo(id=891)

X = cdc_dataset.data.features
y = cdc_dataset.data.targets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1_337)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
