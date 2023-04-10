# -*- coding: UTF-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from tinyscript import logging

from ..common.config import null_logger


__all__ = ["make_pipeline", "DebugPipeline", "DebugTransformer", "Pipeline", "PREPROCESSORS"]


PREPROCESSORS = {
    'MA':         MaxAbsScaler,
    'MM':         MinMaxScaler,
    'Norm':       Normalizer,
    'OneHot':     (OneHotEncoder, {'drop': "if_binary", 'handle_unknown': "ignore"}),
    'Ord':        OrdinalEncoder,
    'PT-bc':      (PowerTransformer, {'method': "box-cox"}),
    'PT-yj':      (PowerTransformer, {'method': "yeo-johnson"}),
    'QT-normal':  (QuantileTransformer, {'output_distribution': "normal"}),
    'QT-uniform': (QuantileTransformer, {'output_distribution': "uniform"}),
    'Rob':        (RobustScaler, {'quantile_range': (25, 75)}),
    'PCA':        (PCA, {'n_components': 2}),
    'Std':        StandardScaler,
}


def make_pipeline(pipeline, preprocessors, logger=null_logger):
    """ Make the ML pipeline by chaining the input preprocessors. """
    for p in preprocessors:
        p, params = PREPROCESSORS.get(p, p), {}
        if isinstance(p, tuple):
            try:
                p, params = p
            except ValueError:
                logger.error("Bad preprocessor format: %s" % p)
                raise
            m = "%s with %s" % (p.__name__, ", ".join("{}={}".format(*i) for i in params.items()))
        else:
            m = p.__name__
        n = p.__name__
        v = "transform" if n.endswith("Transformer") else "encode" if n.endswith("Encoder") else \
            "standardize" if n.endswith("Scaler") else "normalize" if n.endswith("Normalizer") or n == "PCA" \
             else "discretize" if n.endswith("Discretizer") else "preprocess"
        pipeline.append(("%s (%s)" % (v, m), p(**params)))


class DebugPipeline:
    """ Proxy class for attaching a logger ; NOT subclassed as logger cannot be pickled (cfr Model._save). """
    @logging.bindLogger
    def __init__(self, *args, **kwargs):
        global logger
        logger = self.logger
        self.pipeline = None
        kwargs['verbose'] = False  # ensure no verbose, this is managed by self.logger with self._log_message
        self._args = (args, kwargs)
    
    def __getattribute__(self, name):
        if name in ["_args", "_log_message", "append", "logger", "pipeline", "pop", "preprocess", "serialize"]:
            return object.__getattribute__(self, name)
        return object.__getattribute__(object.__getattribute__(self, "pipeline"), name)
    
    def _log_message(self, step_idx):
        """ Overload original method to display messages with our own logger.
        NB: verbosity is controlled via the logger, therefore we output None so that it is not natively displayed. """
        if not getattr(Pipeline, "silent", False):
            name, _ = self.steps[step_idx]
            logger.info("[%d/%d] %s" % (step_idx + 1, len(self.steps), name))
    Pipeline._log_message = _log_message
    
    def append(self, step):
        if not isinstance(step, tuple):
            step = (step.__class__.__name__, step)
        if self.pipeline is None:
            self.pipeline = Pipeline([step], *self._args[0], **self._args[1])
        else:
            self.pipeline.steps.append(step)
    
    def pop(self, idx=None):
        self.pipeline.steps.pop() if idx is None else self.pipeline.steps.pop(idx)
    
    def preprocess(self, X):
        p = self.pipeline[:-1]
        p.steps.append(("debug", DebugTransformer()))
        return p.fit_transform(X)


class DebugTransformer(BaseEstimator, TransformerMixin):
    """ Transformer class for debugging the pipeline. """
    def transform(self, X):
        self.shape = X.shape
        return X
    
    def fit(self, X, y=None, **params):
        return self

