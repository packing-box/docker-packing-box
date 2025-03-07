# -*- coding: UTF-8 -*-
import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator, ClassifierMixin


class REMINDerClassifier(BaseEstimator, ClassifierMixin):
    classes_ = np.array([0, 1])  # binary classifier
    
    def __init__(self, confidence=0.9999):
        """
        A classifier based on REMINDer heuristic of Han et al. using EP section executable flag and entropy.
        
        Parameters
        ----------
        confidence : float, default=0.9999
            The confidence level for computing the threshold for the entropy of the entry point section.
        """
        self.confidence = confidence
        self._feature_names = ["is_ep_in_w_section", "entropy_ep_section"]
    
    def fit(self, X, y):
        """
        Fit the model to the data X by computing the entropy threshold.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 2)
            The training input samples. Each sample shall have two columns:
            "is_ep_in_w_section" and "entropy_ep_section".
        
        y : array-like of shape (n_samples,)
            The target values (class labels in classification).
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        Xe = X[y == 1]
        entropies = Xe[Xe[:, 0] == True][:, 1]
        self.entropy_threshold_ = np.mean(entropies) - stats.norm.ppf((1 + self.confidence) / 2) * np.std(entropies)
        return self
    
    def predict(self, X):
        """
        Predict the target for the input data X based on the entropy thresholds.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 2)
            The training input samples. Each sample shall have two columns:
            "is_ep_in_w_section" and "entropy_ep_section".
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted labels (0 or 1).
        """
        check_is_fitted(self, attributes=["entropy_threshold_"])
        is_ep_in_w_section, entropy_ep_section = X[:, 0], X[:, 1]
        y_pred = (is_ep_in_w_section == True) & (entropy_ep_section >= self.entropy_threshold_)
        return y_pred.astype(int)

