import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Student-Pass-Fail.csv')

X = data[['Self_Study_Daily', 'Tution_Monthly']]
y = data['Pass_Or_Fail']

logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)

data_points = np.array([[7, 28], [10, 34], [2, 39]])

probabilities = logistic_regression.predict_proba(data_points)

# Print the probabilities for each data point
for i, point in enumerate(data_points):
    print(f"Data point {point}: Probability of passing = {probabilities[i][1]:.4f}, Probability of failing = {probabilities[i][0]:.4f}")

#Odds
odds = np.exp(logistic_regression.coef_)
print(f"Odds: {odds}")
