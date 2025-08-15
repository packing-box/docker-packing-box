# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.neural_network import MLPClassifier


class PerdisciClassifier(MLPClassifier):
    """Perdisci's classifier.
    
    This model is based on the Multi-layer Perceptron as of the technique of Perdisci et al. (2008).
    
    Attributes
    ----------
    classes_ : np.array([0, 1])
        0 is not packed, 1 is packed
    
    See the documentation of Scikit-Learn ; API Reference > sklearn.neural_network > MLPClassifier
    URL: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    
    Constants
    ---------
    hidden_layer_sizes : (5, )
    
    activation : 'logistic'
    
    _feature_names : [
            "number_standard_sections",
            "number_non_standard_sections",
            "number_x_sections",
            "number_rwx_sections",
            "number_addresses_in_iat",
            "entropy_pe_header",
            "entropy_code_sections",
            "entropy_data_sections",
            "entropy",
        ]
    
    Parameters
    ----------
    See the documentation of Scikit-Learn ; API Reference > sklearn.neural_network > MLPClassifier
    URL: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    
    References
    ----------
    Roberto Perdisci, Andrea Lanzi, Wenke Lee,
    "Classification of packed executables for accurate computer virus detection",
    Pattern Recognition Letters, 2008.
    URL: http://www.sciencedirect.com/science/article/pii/S0167865508002110
    
    M. Zubair Shafiq, S. Momina Tabish, Muddassar Farooq,
    "PE-Probe: Leveraging Packer Detection and Structural Information to Detect Malicious Portable Executables",
    Proceedings of the Virus Bulletin Conference (VB), 2009.
    URL: https://www.semanticscholar.org/paper/PE-Probe%3A-Leveraging-Packer-Detection-and-to-Detect-Shafiq-Tabish/9811ec751f2b5bb41ee46c0ee2a3b6eccc39bb9a
    
    Examples
    --------
    >>> from pbox.core.model.algorithm.custom.perdisci import PerdisciClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=42, n_features=9, n_redundant=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    >>> clf = PerdisciClassifier().fit(X_train, y_train)
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    """
    classes_ = np.array([0, 1])
    _feature_names = [
        "number_standard_sections",
        "number_non_standard_sections",
        "number_x_sections",
        "number_rwx_sections",
        "number_addresses_in_iat",
        "entropy_pe_header",
        "entropy_code_sections",
        "entropy_data_sections",
        "entropy",
    ]
    
    def __init__(self, **kwargs):
        kwargs.pop("hidden_layer_sizes", None), kwargs.pop("activation", None)
        super().__init__(hidden_layer_sizes=(5, ), activation="logistic", **kwargs)
    
    def fit(self, X, y):
        return super().fit(X, y)
    fit.__doc__ = MLPClassifier.fit.__doc__.replace("n_features", "9")
    
    def predict(self, X):
        return super().predict(X).astype(int)
    predict.__doc__ = MLPClassifier.predict.__doc__.replace("n_classes", "2").replace("n_features", "9")
    
    def predict_proba(self, X):
        return super().predict_proba(X)
    predict_proba.__doc__ = MLPClassifier.predict_proba.__doc__.replace("n_classes", "2").replace("n_features", "9")

