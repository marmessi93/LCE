from .._lce import LCEClassifier, LCERegressor
from sklearn.utils.estimator_checks import check_estimator


def test_classifier():
    assert check_estimator(LCEClassifier()) == None   
    
def test_regressor():
    assert check_estimator(LCERegressor()) == None
    
    