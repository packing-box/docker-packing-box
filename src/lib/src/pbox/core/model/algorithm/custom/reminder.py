# -*- coding: UTF-8 -*-
import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval, RealNotInt
from sklearn.utils.validation import check_is_fitted


THRESHOLD_FROM_STUDY = 6.85


class REMINDerClassifier(BaseEstimator, ClassifierMixin):
    """Classifier based on Han's REMINDer tool for malware forensics.
    
    This model is based on REMINDer heuristic of Han et al. using EP section executable flag and entropy (2009).
    
    Attributes
    ----------
    classes_ : np.array([0, 1])
        0 is not packed, 1 is packed
    
    entropy_threshold_ : float, pre-fitted to 6.85
        The threshold of the entropy of the entry point section.
    
    Constants
    ---------
    _feature_names : ["is_ep_in_w_section", "entropy_ep_section"]
    
    Parameters
    ----------
    confidence : float, default=0.9999
        The confidence level for computing the threshold for the entropy of the entry point section.
    
    References
    ----------
    Seungwon Han, Keungi Lee, Sangjin Lee,
    "Packed PE File Detection for Malware Forensics",
    2nd International Conference on Computer Science and its Applications, 2009.
    URL: https://ieeexplore.ieee.org/document/5404211
    
    Examples
    --------
    >>> from pbox.core.model.algorithm.custom.reminder import REMINDerClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=42, n_features=2, n_redundant=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    >>> clf = REMINDerClassifier().fit(X_train, y_train)
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    """
    classes_ = np.array([0, 1])
    _feature_names = [
        "is_ep_in_w_section",
        "entropy_ep_section",
    ]
    _parameter_constraints = {
        'confidence':  [Interval(RealNotInt, 0., 1., closed="both")],
    }
    
    def __init__(self, confidence=0.9999):
        self.confidence = confidence
        self._validate_params()
        self.entropy_threshold_ = THRESHOLD_FROM_STUDY
    
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

