"""
================================================================================
LCERegressor on Diabetes dataset with scikit-learn hyperparameter grid search
================================================================================

An example of :class:`lce.LCERegressor`
"""

from lce import LCERegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV


# Load data and generate a train/test split
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)

# Build LCERegressor with default parameters
reg = LCERegressor(n_jobs=-1, random_state=0)

# Define parameter ranges for grid search
params = {'n_estimators': list(range(10, 51, 20)),
          'max_depth': list(range(0, 3, 1))}

# Run scikit learn grid search 
grid_cv = GridSearchCV(reg, param_grid=params, cv=3, n_jobs=-1)
grid_cv.fit(X_train, y_train)

# Print best configuration
print("Best n_estimator: ", grid_cv.best_params_['n_estimators'],
      ", best max_depth: ", grid_cv.best_params_['max_depth'])