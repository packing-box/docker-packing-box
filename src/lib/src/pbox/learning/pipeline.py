# -*- coding: UTF-8 -*-
from tinyscript import logging
from tinyscript.helpers import lazy_load_module, lazy_object

from ..common.config import null_logger

lazy_load_module("sklearn.pipeline", alias="sklpl")


__all__ = ["make_pipeline", "DebugPipeline", "DebugTransformer", "Pipeline", "PREPROCESSORS"]


Pipeline = lazy_object(lambda: sklpl.Pipeline)


def __init_pr():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder, \
                                      PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
    return {
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
PREPROCESSORS = lazy_object(__init_pr)


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
        pipeline.append((m, p(**params)))


def __init_pl():
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
    return DebugPipeline
DebugPipeline = lazy_object(__init_pl)


def __init_dt():
    from sklearn.base import BaseEstimator, TransformerMixin
    
    class DebugTransformer(BaseEstimator, TransformerMixin):
        """ Transformer class for debugging the pipeline. """
        def transform(self, X):
            self.shape = X.shape
            return X
        
        def fit(self, X, y=None, **params):
            return self
    return DebugTransformer
DebugTransformer = lazy_object(__init_dt)

