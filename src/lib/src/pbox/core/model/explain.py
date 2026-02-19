# -*- coding: UTF-8 -*-
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

lazy_load_module("shap")
import logging
logger = logging.getLogger(__name__)

def _get_explainer(model_wrapper, X_background, logger=None):
    #? Predictor = full pipeline (scaler + classifier)
    #? inner_clf = only classifier (needed for tree explainer)
    if hasattr(model_wrapper, 'pipeline') and model_wrapper.pipeline is not None:
        predictor = model_wrapper.pipeline
    elif hasattr(model_wrapper, 'classifier') and model_wrapper.classifier is not None:
        predictor = model_wrapper.classifier
    else:
        #Default : model itself
        predictor = model_wrapper

    inner_clf = predictor
    if hasattr(predictor, 'steps') and len(predictor.steps) > 0:
        inner_clf = predictor.steps[-1][1]

    clf_name = inner_clf.__class__.__name__

    tree_models = {"RandomForestClassifier", "GradientBoostingClassifier",
                   "DecisionTreeClassifier", "ExtraTreesClassifier",
                   "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
                   "AdaBoostClassifier"}
    
    linear_models = {"LogisticRegression", "SGDClassifier",
                     "RidgeClassifier", "Perceptron"}
    
    if clf_name in tree_models:
        logger.debug(f"Using TreeExplainer for {clf_name}")
        return shap.TreeExplainer(inner_clf)
    
    if clf_name in linear_models:
        logger.debug(f"Using LinearExplainer for {clf_name}")
        return shap.LinearExplainer(inner_clf, X_background)
    
    # For everything else (SVC, MLP, ...), use Kernel -> slower
    logger.warning(f"Using KernelExplainer for {clf_name}")
    
    # Reduce background for Kernel speed
    # Source for 100 value : https://openreview.net/pdf?id=L38bbHmRKx
    background = shap.sample(X_background, min(35, len(X_background)))
    
    # ? When using KernelExplainer, we must pass the full sklearn pipeline (StandardScaler + Classifier)
    # ? instead of only classifier. Otherwise, KernelSHAP sends unscaled data to the
    # ? classifier, which expects scaled input, thus producing absurd SHAP values.
    predict_fn = predictor.predict_proba if hasattr(predictor, 'predict_proba') else predictor.predict
    
    return shap.KernelExplainer(lambda x: predict_fn(x)[:, 1], background)

def explain_model(model_wrapper, X_data, feature_names=None, max_samples=None, sample_indices=None, **kwargs):
    logger = kwargs.get('logger', logging.getLogger("null"))
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    if sample_indices is not None:
        X_subset = X_data.iloc[sample_indices] if hasattr(X_data, 'iloc') else X_data[sample_indices]
    else:
        #Linear time
        if max_samples is None and len(X_data) > 100:
            max_samples = 100
        
        if max_samples and len(X_data) > max_samples:
            logger.info(f"Subsampling to {max_samples} for explanation")
            indices = np.random.choice(len(X_data), max_samples, replace=False)
            X_subset = X_data.iloc[indices] if hasattr(X_data, 'iloc') else X_data[indices]
        else:
            X_subset = X_data

    explainer = _get_explainer(model_wrapper, X_data, logger=logger)
    
    # Compute SHAP values ONLY on the subset
    #TODO: rule of thumb = 2*n_features + 2048
    if isinstance(explainer, shap.KernelExplainer):
        nsamples = kwargs.get('nsamples', 1000)
        logger.info(f"Computing KernelSHAP on {len(X_subset)} samples with nsamples={nsamples}")
        shap_values = explainer.shap_values(X_subset, nsamples=nsamples)
    else:
        shap_values = explainer.shap_values(X_subset)

    #for tree explainer, shape_value = [array_classe_0, array_classe_1] so we just take value for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    #for kernel explainer, shape_value = (n_samples, n_features) 
    elif hasattr(shap_values, 'shape') and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)) and len(expected) == 2:
        expected = expected[1]

    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(shap_values), expected),
        data=X_subset.values if hasattr(X_subset, 'values') else X_subset,
        feature_names=feature_names
    )

    return {
        'explainer': explainer,
        'shap_explanation': shap_explanation,
        'shap_values': shap_values,
        'expected_value': expected,
        'data': X_subset,
        'feature_names': feature_names,
        'n_samples_explained': len(X_subset),
    }


def _base_value(exp):
    return exp['expected_value'] if np.isscalar(exp['expected_value']) else exp['expected_value'][1]

def _get_sample_idx(exp, packed=True):
    target = 1 if packed else 0
    y_test = exp.get('y_test')
    index_map = exp.get('_index_map')

    # Subset Mode
    if index_map is not None:
        for orig_idx, i in index_map.items():
            if y_test[orig_idx] == target:
                return i
        return 0

    indices = np.where(np.array(y_test) == target)[0]
    return indices[0] if len(indices) > 0 else 0


def _finalize_plot(output_path):
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def shap_summary(model, output_path=None, **kw):
    exp = model._explanation
    plt.figure(figsize=(14, 10))
    shap.plots.beeswarm(exp['shap_explanation'], show=False, max_display=kw.get('max_display', 20))
    _finalize_plot(output_path)


def shap_heatmap(model, output_path=None, max_display=10, **kw):
    exp = model._explanation
    plt.figure(figsize=(14, 10))
    shap.plots.heatmap(exp['shap_explanation'], max_display=max_display, show=False)
    _finalize_plot(output_path)

def shap_decision(model, output_path=None, max_samples=50, **kw):
    exp = model._explanation
    plt.figure(figsize=(14, 10))
    n = min(max_samples, len(exp['shap_values']))
    shap.decision_plot(
        exp['expected_value'],
        exp['shap_values'][:n],
        exp['data'][:n],
        feature_names=exp['feature_names'],
        show=False
    )
    _finalize_plot(output_path)


def _local_plot(model, output_path, packed, plot_type='waterfall', max_display=10):
    exp = model._explanation
    idx = _get_sample_idx(exp, packed=packed)
    
    row_explanation = exp['shap_explanation'][idx]
    
    if plot_type == 'waterfall':
        plt.figure(figsize=(14, 10))
        shap.plots.waterfall(row_explanation, max_display=max_display, show=False)
    elif plot_type == 'force':
        # Don't create plt.figure(), shap.force_plot creates its own
        shap.force_plot(
            row_explanation.base_values,
            row_explanation.values,
            row_explanation.data,
            feature_names=exp['feature_names'],
            matplotlib=True,
            show=False
        )
        
    _finalize_plot(output_path)

def shap_waterfall_packed(model, output_path=None, **kw):
    _local_plot(model, output_path, packed=True, plot_type='waterfall', **kw)

def shap_waterfall_not_packed(model, output_path=None, **kw):
    _local_plot(model, output_path, packed=False, plot_type='waterfall', **kw)

def shap_force_packed(model, output_path=None, **kw):
    _local_plot(model, output_path, packed=True, plot_type='force', **kw)

def shap_force_not_packed(model, output_path=None, **kw):
    _local_plot(model, output_path, packed=False, plot_type='force', **kw)

_EXPLANATIONS = {
    'summary':              shap_summary,
    'waterfall_packed':     shap_waterfall_packed,
    'waterfall_not_packed': shap_waterfall_not_packed,
    'heatmap':              shap_heatmap,
    'decision':             shap_decision,
    'force_packed':         shap_force_packed,
    'force_not_packed':     shap_force_not_packed,
}