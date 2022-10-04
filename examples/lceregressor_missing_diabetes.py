"""
====================================================
LCERegressor on Diabetes dataset with missing values
====================================================

An example of :class:`lce.LCERegressor`
"""

import numpy as np
from lce import LCERegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Load data and generate a train/test split
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, random_state=0
)

# Input 20% of missing values per variable in the train set
np.random.seed(0)
m = 0.2
for j in range(0, X_train.shape[1]):
    sub = np.random.choice(X_train.shape[0], int(X_train.shape[0] * m))
    X_train[sub, j] = np.nan

# Train LCERegressor with default parameters
reg = LCERegressor(n_jobs=-1, random_state=0)
reg.fit(X_train, y_train)

# Make prediction
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("The mean squared error (MSE) on test set: {:.0f}".format(mse))
