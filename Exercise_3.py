import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Bank-data.csv')

data = data.drop(data.columns[0], axis=1)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

y = y.map({'no': 0, 'yes': 1})

logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)

data_points = np.array([[1.335, 0, 1, 0, 0, 109], [1.25, 0, 0, 1, 0, 279]])

probabilities = logistic_regression.predict_proba(data_points)

for i, point in enumerate(data_points):
    print(f"Data point {point}: Probability of subscribing = {probabilities[i][1]:.4f}, Probability of not subscribing = {probabilities[i][0]:.4f}")

odds = np.exp(logistic_regression.coef_)
print(f"Odds: {odds}")
