from typing import List, Set, Dict, Tuple, Optional, Union, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator


class MyClassifier(BaseEstimator):
    """
    A Custom BaseEstimator that can switch between classifiers.
        :config: config file where model parameters are stored
        :classifier_type: classifier type to use
    """

    def __init__(self, config: Dict, classifier_type: str) -> None:
        parameters = config["model_parameters"][classifier_type]
        if classifier_type == "LogisticRegression":
            self.estimator = LogisticRegression(**parameters)
        elif classifier_type == "RandomForest":
            self.estimator = RandomForestClassifier(**parameters)
        elif classifier_type == "SVC":
            self.estimator = SVC(**parameters)

    def fit(
        self, X: List[Any], y: Optional[List[str]] = None, **kwargs
    ) -> BaseEstimator:
        self.estimator.fit(X, y)
        return self

    def predict(self, X: List[str], y: Optional[List[str]] = None) -> str:
        return self.estimator.predict(X)

    def predict_proba(self, X: List[str]) -> float:
        return self.estimator.predict_proba(X)

    def score(self, X: List[str], y: List[str]) -> float:
        return self.estimator.score(X, y)
