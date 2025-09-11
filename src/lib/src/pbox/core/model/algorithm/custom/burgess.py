# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Integral, Interval, StrOptions
from sklearn.utils.validation import check_is_fitted


class BurgessClassifier(ClassifierMixin, BaseEstimator):
    """Burgess's classifier.
    
    This model is based on a heuristic using an average of the per-section entropies for the selected K most occuring
     bytes of Burgess et al. (2014).
    
    Attributes
    ----------
    classes_ : np.array([0, 1])
        0 is not packed, 1 is packed
    
    Constants
    ---------
    _feature_names : ["average_per_section_top_{K}_bytes_entropy"]
    
    Parameters
    ----------
    K : int belonging to ]0,256], default=100
        The number of top bytes to be kept in the Shanon entropy computation of each binary's section.
    
    References
    ----------
    Colin Burgess, Sakir Sezer, Kieran McLaughlin, Eul Gyu Im,
    "Feature set reduction for the detection of packed executables",
    ISSC/CIICT, 2014.
    
    Examples
    --------
    >>> from pbox.core.model.algorithm.custom.burgess import BurgessClassifier
    >>> from pbox.helpers import make_test_dataset
    >>> X_train, y_train, X_test, y_test = make_test_dataset(1)
    >>> clf = BurgessClassifier().fit(X_train, y_train)
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 0])
    >>> clf.score(X_test, y_test)
    0.8...
    """
    classes_ = np.array([0, 1])
    _parameter_constraints = {
        'K':         [Interval(Integral, 0, 256, closed="right")],
        'algorithm': [StrOptions({"KStar", "SMO"})],
    }
    
    def __init__(self, K=100, algorithm="SMO", **kwargs):
        from pbox.core.model.algorithm.weka import KStar, SMO
        self.algorithm = algorithm
        self.K = K
        self._validate_params()
        self.__clf = locals()[algorithm](**kwargs)
        self.__clf._feature_names = [f"average_per_section_top_{K}_bytes_entropy"]
    
    def __getattribute__(self, name):
        if name in ["fit", "predict", "predict_proba"]:
            return super().__getattribute__(name)
        try:
            return super().__getattribute__("_BurgessClassifier__clf").__getattribute__(name)
        except AttributeError:
            return super().__getattribute__(name)
    
    def fit(self, X, y):
        """
        Fit the SVM model with SMO according to the given training data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 1)
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
        X : {array-like, sparse matrix} of shape (n_samples, 1)
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
        X : {array-like, sparse matrix} of shape (n_samples, 1)
            The input samples. Each sample shall have 1 column: "average_per_section_top_{K}_bytes_entropy".
        
        Returns
        -------
        probabilities : array, shape (n_samples, 1)
            Returns an array where each row represents the probabilities of the sample being below and above the
            threshold, respectively.
        """
        return self.__clf.predict_proba(X)

