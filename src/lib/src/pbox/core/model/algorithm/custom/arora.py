# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import ArrayOfLengthN, DictOfLengthAtLeastN, Interval, RealNotInt
from sklearn.utils.validation import check_is_fitted


DEFAULT_WEIGHTS   = np.array([1, .5, 1, 1, 5, 5, 5, 5, 5, 5, 3, 3, 7, 3])
RISK_COEFFICIENTS = np.array([5, 1,  2, 2, 3, 4, 4, 4, 2, 3, 2, 2, 3, 2])
THRESHOLD_FROM_STUDY = 3.


class AroraClassifier(ClassifierMixin, BaseEstimator):
    """Arora's classifier.
    
    This model is based on the static heuristics of Arora et al. (2013).
    
    Attributes
    ----------
    classes_ : np.array([0, 1])
        0 is not packed, 1 is packed
    
    threshold_ : float, pre-fitted to 3.
        The threshold of the risk score computed with the 14 features, weights and risk coefficients.
    
    Constants
    ---------
    _feature_names : [
            "number_known_packer_section_names",
            "has_section_name_not_known",
            "number_sections_name_not_printable",
            "number_sections_name_empty",
            "has_no_code_section",
            "has_code_section_not_x",
            "has_wx_section",
            "is_data_section_x",
            "has_code_data_section",
            "is_ep_section_not_code_or_not_x",
            "is_ep_in_tls_section",
            "has_less_than_20_imports",
            "is_iat_in_non_standard_section",
            "highest_section_entropy_normalized",
        ]
    
    Parameters
    ----------
    confidence : float, default=.99
        The confidence level for computing the threshold of the risk score, based on the not-packed label.
    
    risk_coefficients : {array-like, dictionary} of length 14 with numbers, default=np.array([5,1,2,2,3,4,4,4,2,3,2,2,3,2])
        The list of risk coefficients for the 14 features. By default, this is set to the values found by the
         authors of this method.
    
    weights : {array-like, dictionary} of length 14 with numbers, default=np.array([1,.5,1,1,5,5,5,5,5,5,3,3,7,3])
        The list of weights for the 14 features. By default, this is set to the values found by the authors of this
         method.
    
    References
    ----------
    Rohit Arora, Anishka Singh, Himanshu Pareek, Usha Rani Edara,
    "A Heuristics-based Static Analysis Approach for Detecting Packed PE Binaries",
    International Journal of Security and Its Applications, 2013.
    URL: http://article.nadiapub.com/IJSIA/vol7_no5/24.pdf
    
    Examples
    --------
    >>> from pbox.core.model.algorithm.custom.arora import AroraClassifier
    >>> from pbox.helpers import make_test_dataset
    >>> X_train, y_train, X_test, y_test = make_test_dataset(14)
    >>> clf = AroraClassifier().fit(X_train, y_train)
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    """
    classes_ = np.array([0, 1])
    # features, in the order specified in the paper to match to the weights and risk coefficients
    _feature_names = [
        "number_known_packer_section_names",
        "has_section_name_not_known",
        "number_sections_name_not_printable",
        "number_sections_name_empty",
        "has_no_code_section",
        "has_code_section_not_x",
        "has_wx_section",
        "is_data_section_x",
        "has_code_data_section",
        "is_ep_section_not_code_or_not_x",
        "is_ep_in_tls_section",
        "has_less_than_20_imports",
        "is_iat_in_non_standard_section",
        "highest_section_entropy_normalized",
    ]
    _parameter_constraints = {
        'confidence':        [Interval(RealNotInt, 0., 1., closed="both")],
        'risk_coefficients': [ArrayOfLengthN(14), DictOfLengthAtLeastN(14), None],
        'weights':           [ArrayOfLengthN(14), DictOfLengthAtLeastN(14), None],
    }
    
    def __init__(self, confidence=.99, weights=None, risk_coefficients=None):
        self.confidence = confidence
        self.weights = DEFAULT_WEIGHTS if weights is None else weights
        self.risk_coefficients = RISK_COEFFICIENTS if risk_coefficients is None else risk_coefficients
        self._validate_params()
        if isinstance(self.weights, dict):
            self.weights = np.array([self.weights[f] for f in self._feature_names])
        if isinstance(self.risk_coefficients, dict):
            self.risk_coefficients = np.array([self.risk_coefficients[f] for f in self._feature_names])
        self.threshold_ = THRESHOLD_FROM_STUDY
    
    def _compute_scores(self, X):
        """ Risk computation method. """
        return np.sqrt(np.sum((np.array(X, dtype="float") * self.weights) ** self.risk_coefficients, axis=1))
    
    def fit(self, X, y):
        """
        Fit the model to the data X by computing the risk score.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 14)
            The training input samples. Each sample shall have fourteen columns:
            "has_code_data_section", "has_code_section_not_x", "has_few_imports", "has_known_packer_section_names",
            "has_no_code_section", "has_section_name_empty", "has_section_name_not_known",
            "has_section_name_not_printable", "has_wx_section", "highest_section_entropy", "is_data_section_x",
            "is_ep_in_tls_section", "is_ep_section_not_code_or_not_x" and "is_iat_in_non_standard_section".
        
        y : array-like of shape (n_samples,)
            The target values (class labels in classification).
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # (1) compute the risk score for each sample
        scores = self._compute_scores(X)
        # (2) select the minimal value from the computed risk score given the confidence parameter
        self.threshold_ = np.quantile(scores[y == 0], self.confidence)
        self.score_range_y0_ = (float(min(scores[y == 0])), float(max(scores[y == 0])))
        self.score_range_y1_ = (float(min(scores[y == 1])), float(max(scores[y == 1])))
        return self
    
    def predict(self, X):
        """
        Predict the target for the input data X based on the risk score given the threshold computed with the fit()
         method.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 14)
            The input samples. Each sample shall have fourteen columns:
            "has_code_data_section", "has_code_section_not_x", "has_few_imports", "has_known_packer_section_names",
            "has_no_code_section", "has_section_name_empty", "has_section_name_not_known",
            "has_section_name_not_printable", "has_wx_section", "highest_section_entropy", "is_data_section_x",
            "is_ep_in_tls_section", "is_ep_section_not_code_or_not_x" and "is_iat_in_non_standard_section".
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted labels (0 or 1).
        """
        check_is_fitted(self, attributes=["threshold_"])
        return np.where(self._compute_scores(X) >= self.threshold_, 1, 0)
    
    def predict_proba(self, X):
        """
        Predict the probability of each sample in X being above or below the threshold.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 14)
            The input samples. Each sample shall have fourteen columns:
            "has_code_data_section", "has_code_section_not_x", "has_few_imports", "has_known_packer_section_names",
            "has_no_code_section", "has_section_name_empty", "has_section_name_not_known",
            "has_section_name_not_printable", "has_wx_section", "highest_section_entropy", "is_data_section_x",
            "is_ep_in_tls_section", "is_ep_section_not_code_or_not_x" and "is_iat_in_non_standard_section".
        
        Returns
        -------
        probabilities : array, shape (n_samples, 2)
            Returns an array where each row represents the probabilities of the sample being below and above the
            threshold, respectively.
        """
        check_is_fitted(self, attributes=["threshold_"])
        # sigmoid function for probability estimation
        proba = 1 / (1 + np.exp(-self._compute_scores(X)))
        return np.vstack((1 - proba, proba)).T

