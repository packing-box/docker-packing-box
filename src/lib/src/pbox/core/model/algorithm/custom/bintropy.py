# -*- coding: UTF-8 -*-
import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator, ClassifierMixin


class BintropyClassifier(BaseEstimator, ClassifierMixin):
    classes_ = np.array([0, 1])  # binary classifier
    
    def __init__(self, blocksize=256, confidence=0.9999, per_section=False):
        """
        A classifier based on Bintropy heuristic using block entropy metrics.
        
        Parameters
        ----------
        blocksize : int, default=256
            The size of blocks in bytes for computing the block entropies.
        
        confidence : float, default=0.9999
            The confidence level for computing thresholds based on the highest and average block entropies.
        
        per_section : bool, default=False
            Whether block entropy features shall be computed on the whole binary sample or with an average of block
             entropies per section.
        """
        self.blocksize = blocksize
        self.confidence = confidence
        self.per_section = per_section
        self._feature_names = [f"average{['', '_per_section'][per_section]}_{blocksize}B_block_entropy",
                               f"highest{['', '_per_section'][per_section]}_{blocksize}B_block_entropy"]
    
    def fit(self, X, y):
        """
        Fit the model to the data X by computing the entropy thresholds.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 2)
            The training input samples. Each sample shall have two (dynamically named) columns:
            "average(_per_section)_[N]B_block_entropy" and "highest(_per_section)_[N]B_block_entropy".
        
        y : array-like of shape (n_samples,)
            The target values (class labels in classification).
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        average_entropies, highest_entropies = X[y == 1][:, 0], X[y == 1][:, 1]
        z_value = stats.norm.ppf((1 + self.confidence) / 2)  # z-value for the confidence level
        self.highest_entropy_threshold_ = np.mean(highest_entropies) - z_value * np.std(highest_entropies)
        self.average_entropy_threshold_ = np.mean(average_entropies) - z_value * np.std(average_entropies)
        return self

    def predict(self, X):
        """
        Predict the target for the input data X based on the entropy thresholds.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 2)
            The input samples. Each sample shall have two (dynamically named) columns:
            "average(_per_section)_[N]B_block_entropy" and "highest(_per_section)_[N]B_block_entropy".
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted labels (0 or 1).
        """
        check_is_fitted(self, attributes=["average_entropy_threshold_", "highest_entropy_threshold_"])
        average_entropies, highest_entropies = X[:, 0], X[:, 1]
        y_pred = (highest_entropies >= self.highest_entropy_threshold_) & \
                 (average_entropies >= self.average_entropy_threshold_)
        return y_pred.astype(int)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 2)
            The training input samples. Each sample shall have two columns:
            "highest_block_entropy" and "average_block_entropy".
        
        Returns
        -------
        proba : array of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, attributes=["average_entropy_threshold_", "highest_entropy_threshold_"])
        average_entropies, highest_entropies = X[:, 0], X[:, 1]
        P_packed = np.clip((highest_entropies - self.highest_entropy_threshold_ + \
                            average_entropies - self.average_entropy_threshold_) / 2, 0, 1)
        return np.vstack([1 - P_packed, P_packed]).T

