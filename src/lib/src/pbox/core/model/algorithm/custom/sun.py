# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted


class SunClassifier(ClassifierMixin, BaseEstimator):
    """Sun's classifier.
    
    This model is based on statistical features based on a randomness profile of Sun et al. (2010).
    
    Attributes
    ----------
    classes_ : np.array([0, 1])
        0 is not packed, 1 is packed
    
    Constants
    ---------
    _feature_names : ["randomness_profile_value_[0-49]"]
    
    References
    ----------
    Li Sun, Steven Versteeg, Serdar Boztas, Trevor Yann
    "Pattern Recognition Techniques for the Classification of Malware Packers",
    Information Security and Privacy, 2010.
    
    Examples
    --------
    >>> from pbox.core.model.algorithm.custom.burgess import SunClassifier
    >>> from pbox.helpers import make_test_dataset
    >>> X_train, y_train, X_test, y_test = make_test_dataset(50)
    >>> clf = SunClassifier().fit(X_train, y_train)
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 1, 0])
    >>> clf.score(X_test, y_test)
    0.8...
    """
    classes_ = np.array([0, 1])
    _feature_names = [f"randomness_profile_value_{i}" for i in range(50)]
    _parameter_constraints = {
        'algorithm': [StrOptions({"BFTree", "IBk", "NB", "SMO"})],
    }
    
    def __init__(self, algorithm="IBk", **kwargs):
        from pbox.core.model.algorithm.weka import BFTree, IBk, NB, SMO
        self.algorithm = algorithm
        if algorithm == "IBk":
            kwargs.setdefault('K', 1)
        self._validate_params()
        self.__clf = locals()[algorithm](**kwargs)
    
    def __getattribute__(self, name):
        if name in ["fit", "predict", "predict_proba"]:
            return super().__getattribute__(name)
        try:
            return super().__getattribute__("_SunClassifier__clf").__getattribute__(name)
        except AttributeError:
            return super().__getattribute__(name)
    
    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 50)
            Training vectors, where `n_samples` is the number of samples
        
        y : array-like of shape (n_samples,)
            Target values (class labels in classification, real numbers in regression).
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.__clf.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Perform classification on samples in X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 50)
            Testing vectors, where `n_samples` is the number of samples
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        return self.__clf.predict(X)
    
    def predict_proba(self, X):
        """
        Predict the probability of each sample in X with the entropy score.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 50)
            The input samples. Each sample shall have 1 column: "average_per_section_top_{K}_bytes_entropy".
        
        Returns
        -------
        probabilities : array, shape (n_samples, 50)
            Returns an array where each row represents the probabilities of the sample being below and above the
            threshold, respectively.
        """
        return self.__clf.predict_proba(X)

