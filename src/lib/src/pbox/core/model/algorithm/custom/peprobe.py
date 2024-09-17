# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.neural_network import MLPClassifier


class PEProbeClassifier(MLPClassifier):
    classes_ = np.array([0, 1])  # binary classifier
    
    def __init__(self, **kwargs):
        """
        A classifier based on the technique of Shafiq et al. implemented in their tool, PE-Probe. This classifier only
        implements M-1 from its architecture, that is, the module for distinguishing between packed and non-packed.
        
        Parameters
        ----------
        See the documentation of Scikit-Learn ; API Reference > sklearn.neural_network > MLPClassifier
        URL: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        """
        self._feature_names = ["number_standard_sections", "number_non_standard_sections", "number_x_sections",
                               "number_rwx_sections", "number_addresses_in_iat", "entropy_pe_header",
                               "entropy_code_sections", "entropy_data_sections", "entropy"]
        kwargs.pop("hidden_layer_sizes")
        kwargs.pop("activation")
        super(PEProbeClassifier, self).__init__(hidden_layer_sizes=(5, ), activation="logistic")
    
    def fit(self, X, y):
        return super(PEProbeClassifier, self).fit(X, y)
    fit.__doc__ = MLPClassifier.fit.__doc__.replace("n_features", "9")
    
    def predict(self, X):
        return super(PEProbeClassifier, self).predict(X).astype(int)
    predict.__doc__ = MLPClassifier.predict.__doc__.replace("n_classes", "2").replace("n_features", "9")
    
    def predict_proba(self, X):
        return super(PEProbeClassifier, self).predict_proba(X)
    predict_proba.__doc__ = MLPClassifier.predict_proba.__doc__.replace("n_classes", "2").replace("n_features", "9")

