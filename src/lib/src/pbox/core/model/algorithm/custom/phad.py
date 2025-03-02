# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class PHADClassifier(BaseEstimator, ClassifierMixin):
    classes_ = np.array([0, 1])  # binary classifier
    
    def __init__(self, confidence=1.):
        """
        A classifier based on the technique of Choi et al. called PE Header Analysis-based packed file Detection (PHAD).
        
        Parameters
        ----------
        confidence : float, default=1.
            The confidence level for computing the threshold if the minimum Euclidean distance for packed samples is
            beneath the Euclidean distance for not-packed samples.
        """
        self.confidence = confidence
        self._feature_names = ["number_wx_sections", "number_x_not_code_or_not_x_code_sections",
                               "number_sections_name_not_printable", "has_no_x_section",
                               "is_sum_of_all_sections>file_size", "is_pos_pe_sig<size_of_image_dos_header",
                               "is_ep_not_in_x_section", "is_ep_not_in_code_section"]
    
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
        proba_above = np.clip((d - self.threshold_) / np.max(d - self.threshold_), 0, 1)
        proba_below = 1 - proba_above
        return np.vstack((proba_below, proba_above)).T

