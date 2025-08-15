# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import ArrayOfLengthN, DictOfLengthAtLeastN, Interval, RealNotInt
from sklearn.utils.validation import check_is_fitted


DEFAULT_WEIGHTS   = np.array([6,  6, 12, 50, 20, 100, 35, 100])
RISK_COEFFICIENTS = np.array([10, 1, 5,  10, 5,  10,  10, 10])
THRESHOLD_FROM_STUDY = 3.


class TreadwellClassifier(BaseEstimator, ClassifierMixin):
    """Treadwell's classifier.
    
    This model is based on the static analysis approach of Treadwell et al. using risk scoring (2009).
    
    Attributes
    ----------
    classes_ : np.array([0, 1])
        0 is not packed, 1 is packed
    
    threshold_ : float, pre-fitted to 3.
        The threshold of the risk score computed with the 8 features, weights and risk coefficients.
    
    Constants
    ---------
    _feature_names : [
            "has_non_standard_section",
            "has_known_packer_section_names",
            "is_ep_not_in_text_section",
            "has_tls_data_directory_entry",
            "has_dll_with_no_export",
            "has_known_packer_section_names",
            "is_import_functions_count<=2",
            "is_iat_malformed",
        ]
    
    Parameters
    ----------
    confidence : float, default=.99
        The confidence level for computing the threshold of the risk score, based on the not-packed label.
    
    risk_coefficients : {array-like, dictionary} of length 8 with numbers, default=np.array([10,1,5,10,5,10,10,10])
        The list of risk coefficients for the 8 features. By default, this is set to the values found by the
         authors of this method.
    
    weights : {array-like, dictionary} of length 8 with numbers, default=np.array([6,6,12,50,20,100,35,100])
        The list of weights for the 8 features. By default, this is set to the values found by the authors of this
         method.
    
    References
    ----------
    Scott Treadwell, Mian Zhou,
    "A Heuristic Approach for Detection of Obfuscated Malware",
    IEEE International Conference on Intelligence and Security Informatics, 2009.
    URL: https://ieeexplore.ieee.org/document/5137328
    
    Examples
    --------
    >>> from pbox.core.model.algorithm.custom.treadwell import TreadwellClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=42, n_features=8, n_redundant=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    >>> clf = TreadwellClassifier().fit(X_train, y_train)
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 1, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    """
    classes_ = np.array([0, 1])
    # features, in the order specified in the paper to match to the weights and risk coefficients
    _feature_names = [
        "has_non_standard_section",
        "has_known_packer_section_names",
        "is_ep_not_in_text_section",
        "has_tls_data_directory_entry",
        "has_dll_with_no_export",
        "has_known_packer_section_names",
        "is_import_functions_count<=2",
        "is_iat_malformed",
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
        return np.sum(np.array(X, dtype="float") * self.weights * self.risk_coefficients, axis=1) / sum(self.weights)
    
    def fit(self, X, y):
        """
        Fit the model to the data X by computing the risk score.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 8)
            The training input samples. Each sample shall have fourteen columns:
            "has_non_standard_section", "has_known_packer_section_names", "is_ep_not_in_text_section",
            "has_tls_data_directory_entry", "has_dll_with_no_export", "has_known_packer_section_names",
            "is_import_functions_count<=2", "is_iat_malformed".
        
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
        X : {array-like, sparse matrix} of shape (n_samples, 8)
            The training input samples. Each sample shall have fourteen columns:
            "has_non_standard_section", "has_known_packer_section_names", "is_ep_not_in_text_section",
            "has_tls_data_directory_entry", "has_dll_with_no_export", "has_known_packer_section_names",
            "is_import_functions_count<=2", "is_iat_malformed".
        
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
        X : {array-like, sparse matrix} of shape (n_samples, 8)
            The input samples. Each sample shall have eight columns:
            "number_wx_sections", "number_x_not_code_or_not_x_code_sections", "number_sections_name_not_printable",
            "has_no_x_section", "is_sum_of_all_sections>file_size", "is_pos_pe_sig<size_of_image_dos_header",
            "is_ep_not_in_x_section" and "is_ep_not_in_code_section".
        
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

