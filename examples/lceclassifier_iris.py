"""
=============================
LCEClassifier on Iris dataset
=============================

An example of :class:`lce.LCEClassifier`
"""

from lce import LCEClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Load data and generate a train/test split
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, random_state=0
)

# Train LCEClassifier with default parameters
clf = LCEClassifier(n_jobs=-1, random_state=0)
clf.fit(X_train, y_train)

# Make prediction and compute accuracy score
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.1f}%".format(accuracy * 100))
