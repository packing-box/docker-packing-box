# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.svm import SVC


class NatarajClassifier(SVC):
    """Nataraj's classifier.
    
    This model is based on the Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel as of the
     technique of Nataraj et al. (2010).
    
    Attributes
    ----------
    classes_ : np.array([0, 1])
        0 is not packed, 1 is packed
    
    See the documentation of Scikit-Learn ; API Reference > sklearn.svm > SVC
    URL: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    
    Constants
    ---------
    kernel : 'rbf'
    
    _feature_names : [f"histogram_bigram_{i}" for i in range(5000)]
    
    Parameters
    ----------
    See the documentation of Scikit-Learn ; API Reference > sklearn.svm > SVC
    URL: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    
    References
    ----------
    Lakshmanan Nataraj, Grégoire Jacob, B.S. Manjunath,
    "Detecting packed executables based on raw binary data",
    Technical report, 2010.
    URL: https://api.semanticscholar.org/CorpusID:5876296
    
    Examples
    --------
    >>> from pbox.core.model.algorithm.custom.nataraj import NatarajClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=42, n_features=5000, n_redundant=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    >>> clf = NatarajClassifier().fit(X_train, y_train)
    >>> clf.predict(X_test[:5, :])
    array([0, 0, 1, 1, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    """
    classes_ = np.array([0, 1])
    _feature_names = [f"histogram_bigram_{i}" for i in range(5000)]
    
    def __init__(self, **kwargs):
        kwargs.pop("kernel", None)
        super().__init__(kernel="rbf", **kwargs)
    
    def fit(self, X, y):
        return super().fit(X, y)
    fit.__doc__ = SVC.fit.__doc__.replace("n_features", "5000")
    
    def predict(self, X):
        return super().predict(X).astype(int)
    predict.__doc__ = SVC.predict.__doc__.replace("n_classes", "2").replace("n_features", "5000")
    
    def predict_proba(self, X):
        return super().predict_proba(X)
    predict_proba.__doc__ = SVC.predict_proba.__doc__.replace("n_classes", "2").replace("n_features", "5000")

