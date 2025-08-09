# -*- coding: UTF-8 -*-
import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval, IntOptions, RealNotInt
from sklearn.utils.validation import check_is_fitted


AVERAGE_ENTROPY_THRESHOLD = 6.677
HIGHEST_ENTROPY_THRESHOLD = 7.199


class BintropyClassifier(BaseEstimator, ClassifierMixin):
    """Lyda's Bintropy-based classifier.
    
    This model is based on Bintropy heuristic using block entropy metrics as of Lyda et al. (2007).
    
    Attributes
    ----------
    average_entropy_threshold_ : float, pre-fitted to 6.677
        The threshold of the average entropy among the X-bytes blocks (X is 256 or 512).
    
    classes_ : np.array([0, 1])
        0 is not packed, 1 is packed
    
    highest_entropy_threshold_ : float, pre-fitted to 7.199
        The threshold of the highest entropy among the X-bytes blocks (X is 256 or 512).
    
    Constants
    ---------
    _feature_names : ["average_256B_block_entropy", "highest_256B_block_entropy"] OR
                     ["average_per_section_256B_block_entropy", "highest_per_section_256B_block_entropy"] OR
                     ["average_512B_block_entropy", "highest_512B_block_entropy"] OR
                     ["average_per_section_512B_block_entropy", "highest_per_section_512B_block_entropy"]
    
    Parameters
    ----------
    blocksize : int, default=256
        The size of blocks in bytes for computing the block entropies.
    
    confidence : float, default=0.9999
        The confidence level for computing thresholds based on the highest and average block entropies.
    
    per_section : bool, default=False
        Whether block entropy features shall be computed on the whole binary sample or with an average of block
         entropies per section.
    
    References
    ----------
    Robert Lyda, James Hamrock,
    "Using Entropy Analysis to Find Encrypted and Packed Malware",
    IEEE Security & Privacy, 2007.
    
    Examples
    --------
    >>> from pbox.core.model.algorithm.custom.bintropy import BintropyClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=42, n_features=2, n_redundant=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    >>> clf = BintropyClassifier().fit(X_train, y_train)
    >>> clf.predict(X_test[:5, :])
    array([1, 1, 1, 1, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    """
    classes_ = np.array([0, 1])
    _parameter_constraints = {
        'blocksize':   [IntOptions({256, 512})],
        'confidence':  [Interval(RealNotInt, 0., 1., closed="both")],
        'per_section': ["boolean"],
    }
    
    def __init__(self, blocksize=256, confidence=0.9999, per_section=False):
        self.blocksize = blocksize
        self.confidence = confidence
        self.per_section = per_section
        self._feature_names = [f"average{['', '_per_section'][per_section]}_{blocksize}B_block_entropy",
                               f"highest{['', '_per_section'][per_section]}_{blocksize}B_block_entropy"]
        self._validate_params()
        # pre-fitting with values from the reference paper
        self.average_entropy_threshold_ = AVERAGE_ENTROPY_THRESHOLD
        self.highest_entropy_threshold_ = HIGHEST_ENTROPY_THRESHOLD
    
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
        self.average_entropy_threshold_ = np.mean(average_entropies) - z_value * np.std(average_entropies)
        self.highest_entropy_threshold_ = np.mean(highest_entropies) - z_value * np.std(highest_entropies)
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
        y_pred = (average_entropies >= self.average_entropy_threshold_) & \
                 (highest_entropies >= self.highest_entropy_threshold_)
        return y_pred.astype(int)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 2)
            The training input samples. Each sample shall have two (dynamically named) columns:
            "average(_per_section)_[N]B_block_entropy" and "highest(_per_section)_[N]B_block_entropy".
        
        Returns
        -------
        proba : array of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, attributes=["average_entropy_threshold_", "highest_entropy_threshold_"])
        average_entropies, highest_entropies = X[:, 0], X[:, 1]
        P_packed = np.clip((average_entropies - self.average_entropy_threshold_ + \
                            highest_entropies - self.highest_entropy_threshold_) / 2, 0, 1)
        return np.vstack([1 - P_packed, P_packed]).T

