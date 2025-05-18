import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('Student-Pass-Fail.csv')

X = data[['Self_Study_Daily', 'Tution_Monthly']]
y = data['Pass_Or_Fail']

logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)

#Split the data into training and testing sets
def train_test_split(X, y, test_size=0.2):
    data = pd.concat([X, y], axis=1)

    data = data.sample(frac=1).reset_index(drop=True)
    
    test_samples = int(test_size * len(data))
    
    train_data = data[:-test_samples]
    test_data = data[-test_samples:]
    
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y)

logistic_regression.fit(X_train, y_train)

y_pred = logistic_regression.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
