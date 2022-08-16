"""
======================================================================
LCEClassifier on Iris dataset with scikit-learn cross validation score
======================================================================

An example of :class:`lce.LCEClassifier`
"""

from lce import LCEClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, random_state=0
)

# Set LCEClassifier with default parameters
clf = LCEClassifier(n_jobs=-1, random_state=0)

# Compute cross-validation scores
cv_scores = cross_val_score(clf, X_train, y_train, cv=3)
cv_scores = [round(elem * 100, 1) for elem in cv_scores.tolist()]
print("Cross-validation scores on train set: ", cv_scores)
