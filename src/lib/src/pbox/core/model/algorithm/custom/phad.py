# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval, RealNotInt
from sklearn.utils.validation import check_is_fitted


THRESHOLD_FROM_STUDY = 1.4


class PHADClassifier(BaseEstimator, ClassifierMixin):
    """Choi's PHAD-based classifier.
    
    This model is based on the technique of Choi et al. called PE Header Analysis-based packed file Detection (2008).
    
    Attributes
    ----------
    classes_ : np.array([0, 1])
        0 is not packed, 1 is packed
    
    threshold_ : float, pre-fitted to 1.4
        The threshold of the Euclidian distance of a packed file.
    
    Constants
    ---------
    _feature_names : [
            "number_wx_sections",
            "number_x_not_code_or_not_x_code_sections",
            "number_sections_name_not_printable",
            "has_no_x_section",
            "is_sum_of_all_sections>file_size",
            "is_pos_pe_sig<size_of_image_dos_header",
            "is_ep_not_in_x_section",
            "is_ep_not_in_code_section",
        ]
    
    Parameters
    ----------
    confidence : float, default=1.
        The confidence level for computing the threshold if the minimum Euclidean distance for packed samples is
        beneath the Euclidean distance for not-packed samples.
    
    References
    ----------
    Yang-seo Choi, Ik-kyun Kim, Jin-tae Oh, Jae-cheol Ryou,
    "PE File Header Analysis-based Packed PE File Detection Technique (PHAD)",
    IEEE International Symposium on Computer Science and its Applications, 2008.
    
    Examples
    --------
    >>> from pbox.core.model.algorithm.custom.phad import PHADClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=42, n_features=8, n_redundant=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    >>> clf = PHADClassifier().fit(X_train, y_train)
    >>> clf.predict(X_test[:5, :])
    array([0, 1, 0, 0, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    """
    classes_ = np.array([0, 1])
    _feature_names = [
        "number_wx_sections",
        "number_x_not_code_or_not_x_code_sections",
        "number_sections_name_not_printable",
        "has_no_x_section",
        "is_sum_of_all_sections>file_size",
        "is_pos_pe_sig<size_of_image_dos_header",
        "is_ep_not_in_x_section",
        "is_ep_not_in_code_section",
    ]
    _parameter_constraints = {
        'confidence':  [Interval(RealNotInt, 0., 1., closed="both")],
    }
    
    def __init__(self, confidence=1.):
        self.confidence = confidence
        self._validate_params()
        self.threshold_ = THRESHOLD_FROM_STUDY
    
    def fit(self, X, y):
        """
        Fit the model to the data X by computing the Euclidian distances.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 8)
            The training input samples. Each sample shall have eight columns:
            "number_wx_sections", "number_x_not_code_or_not_x_code_sections", "number_sections_name_not_printable",
            "has_no_x_section", "is_sum_of_all_sections>file_size", "is_pos_pe_sig<size_of_image_dos_header",
            "is_ep_not_in_x_section" and "is_ep_not_in_code_section".
        
        y : array-like of shape (n_samples,)
            The target values (class labels in classification).
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # (1) compute Euclidian distances of X's rows to the null vector (not-packed characteristics are all 0's) for
        #      label 1 (packed)
        d = np.sort(np.linalg.norm(X[y == 1], axis=1))
        # (2) select the minimal value from the computed Euclidian distances given the confidence parameter
        self.threshold_ = d[max(min(int(np.floor(self.confidence * len(d)) - 1), len(d) - 1), 0)]
        # for information, compute the resulting confidence on label 0 (not packed)
        d = np.sort(np.linalg.norm(X[y == 0], axis=1))
        j = len(d) - 1
        while d[j] >= self.threshold_:
            j -= 1
        self.confidence_y0_ = (j + 1) / len(d)
        return self
    
    def predict(self, X):
        """
        Predict the target for the input data X based on the Euclidian distance from Characteristics Vector (CV)
        to the null vector given the threshold computed with the fit() method.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 8)
            The input samples. Each sample shall have eight columns:
            "number_wx_sections", "number_x_not_code_or_not_x_code_sections", "number_sections_name_not_printable",
            "has_no_x_section", "is_sum_of_all_sections>file_size", "is_pos_pe_sig<size_of_image_dos_header",
            "is_ep_not_in_x_section" and "is_ep_not_in_code_section".
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted labels (0 or 1).
        """
        check_is_fitted(self, attributes=["threshold_"])
        return (np.linalg.norm(X, axis=1) > self.threshold_).astype(int)
    
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
        d = np.linalg.norm(X, axis=1)
        proba_packed = np.clip((d - self.threshold_) / np.max(d - self.threshold_), 0, 1)
        return np.vstack((1 - proba_packed, proba_packed)).T

