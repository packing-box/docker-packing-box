# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DT


class OneRClassifier(DT):
    """D'hooge's classifier.
    
    This model is based on a decision tree with a maximum depth of 1, as of D'hooge et al. (2023).
    
    Attributes
    ----------
    classes_ : np.array([0, 1])
        0 is not packed, 1 is packed
    
    See the documentation of Scikit-Learn ; API Reference > sklearn.tree > DecisionTreeClassifier
    URL: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    
    Constants
    ---------
    max_depth : 1
    
    Parameters
    ----------
    feature_name : str or int
        The name of the feature to be considered.
    
    See the documentation of Scikit-Learn ; API Reference > sklearn.neural_network > MLPClassifier
    URL: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    
    References
    ----------
    Laurens D'hooge, Miel Verkerken, Tim Wauters, Filip De Turck, Bruno Volckaert,
    "Castles Built on Sand: Observations from Classifying Academic Cybersecurity Datasets with Minimalist Methods",
    IoTBDS, 2023.
    URL: https://biblio.ugent.be/publication/01GZDZT436A4ZTH93KBT5D7W87
    
    Examples
    --------
    >>> from pbox.core.model.algorithm.custom.oner import OneRClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=42, n_features=2, n_redundant=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    >>> clf = OneRClassifier(0).fit(X_train, y_train)
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    """
    classes_ = np.array([0, 1])
    
    def __init__(self, feature_name=None, **kwargs):
        kwargs.pop("max_depth", None)
        super().__init__(max_depth=1, **kwargs)
        self._feature_names = [feature_name]
    
    def fit(self, X, y):
        return super().fit(X, y)
    fit.__doc__ = DT.fit.__doc__.replace("n_features", "1")
    
    def predict(self, X):
        return super().predict(X).astype(int)
    predict.__doc__ = DT.predict.__doc__.replace("n_classes", "1").replace("n_features", "1")
    
    def predict_proba(self, X):
        return super().predict_proba(X)
    predict_proba.__doc__ = DT.predict_proba.__doc__.replace("n_classes", "2").replace("n_features", "1")

