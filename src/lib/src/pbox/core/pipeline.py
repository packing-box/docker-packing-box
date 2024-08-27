# -*- coding: UTF-8 -*-
import builtins as bi
from tinyscript import logging

lazy_load_module("sklearn.pipeline", alias="sklpl")


__all__ = ["make_pipeline", "DebugPipeline", "DebugTransformer", "Pipeline"]


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
bi.PREPROCESSORS = lazy_object(__init_pr)


# defined here for pickling when a new model gets joblib-serialized
def _df2npa(X):
    """ Dummy function for converting an input pandas.DataFrame to a numpy.array ; used for the FunctionTransformer that
         allows to implement an identity transformer (when no scaler or transformer is required in the pipeline). """
    from numpy import array
    return array(X)


def make_pipeline(pipeline, preprocessors, logger=null_logger):
    """ Make the ML pipeline by chaining the input preprocessors. """
    if len(preprocessors) == 0:  # create the Pipeline instance with the list of steps
        from sklearn.preprocessing import FunctionTransformer
        pipeline.append(("pandas.DataFrame -> numpy.array", FunctionTransformer(_df2npa)))
    for p in preprocessors:
        p, params = PREPROCESSORS.get(p, p), {}
        if isinstance(p, tuple):
            try:
                p, params = p
            except ValueError:
                logger.error(f"Bad preprocessor format: {p}")
                raise
            m = f"{p.__name__} with {', '.join('{}={}'.format(*i) for i in params.items())}"
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
            NB: verbosity is controlled via the logger, hence we output None so that it is not natively displayed. """
            if not getattr(Pipeline, "silent", False):
                name, _ = self.steps[step_idx]
                logger.info(f"[{step_idx+1}/{len(self.steps)}] {name}")
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

