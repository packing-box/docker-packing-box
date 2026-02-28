# -*- coding: UTF-8 -*-
import logging
import matplotlib
import warnings

from ...helpers import *

lazy_load_module("shap")
matplotlib.use('Agg')


__all__ = ["explain_model"]


def _get_explainer(model_wrapper, X_background, **kw):
    #? predictor = full pipeline (scaler + classifier)
    #? inner_clf = only classifier (needed for tree explainer)
    l = kw.get('logger', null_logger)
    predictor = getattr(model_wrapper, "pipeline", None) or getattr(model_wrapper, "classifier", None) or model_wrapper
    inner_clf = predictor.steps[-1][1] if len(getattr(predictor, "steps", [])) > 0 else predictor
    tree_models = {"RandomForestClassifier", "GradientBoostingClassifier", "DecisionTreeClassifier", 
                   "ExtraTreesClassifier", "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
                   "AdaBoostClassifier"}
    linear_models = {"LogisticRegression", "SGDClassifier", "RidgeClassifier", "Perceptron"}
    if (clf_name := inner_clf.__class__.__name__) in tree_models:
        l.debug(f"Using TreeExplainer for {clf_name}")
        return shap.TreeExplainer(inner_clf)
    if clf_name in linear_models:
        l.debug(f"Using LinearExplainer for {clf_name}")
        return shap.LinearExplainer(inner_clf, X_background)
    # for everything else (SVC, MLP, ...), use Kernel -> slower
    l.warning(f"Using KernelExplainer for {clf_name}")
    # reduce background for Kernel speed
    background = shap.sample(X_background, min(config['shap-kernel-explainer-samples'], len(X_background)))
    # when using KernelExplainer, we must pass the full sklearn pipeline (StandardScaler + Classifier) instead of only
    #  classifier. Otherwise, KernelSHAP sends unscaled data to the classifier, which expects scaled input, thus
    #  producing absurd SHAP values.
    if hasattr(predictor, "predict_proba"):
        return shap.KernelExplainer(lambda x: getattr(predictor, "predict_proba")(x)[:, 1], background)
    #FIXME: should be "predictor" instead of "inner_clf", but when using "predictor", we get error:
    #        [...]
    #        File "/home/user/.local/lib/python3.12/site-packages/shap/explainers/_kernel.py", line 95, in __init__
    #          self.model = convert_to_model(model, keep_index=self.keep_index)
    #                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #        File "/home/user/.local/lib/python3.12/site-packages/shap/utils/_legacy.py", line 129, in convert_to_model
    #          out.f.__self__.feature_names_in_ = None
    #          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #      AttributeError: property 'feature_names_in_' of 'Pipeline' object has no setter
    return shap.KernelExplainer(inner_clf.predict, background)


def _get_sample_idx(exp, packed=True):
    target = 1 if packed else 0
    y_test = exp.get('y_test')
    index_map = exp.get('_index_map')
    # subset mode
    if index_map is not None:
        for orig_idx, i in index_map.items():
            if y_test[orig_idx] == target:
                return i
        return 0
    indices = np.where(np.array(y_test) == target)[0]
    return indices[0] if len(indices) > 0 else 0


def _local_plot(model, packed, plot_type="waterfall", max_display=10):
    row_explanation = exp['shap_explanation'][_get_sample_idx(exp := model._explanation, packed=packed)]
    if plot_type == "waterfall":
        plt.figure(figsize=(14, 10))
        shap.plots.waterfall(row_explanation, max_display=max_display, show=False)
    elif plot_type == "force":
        # don't create plt.figure(), shap.force_plot creates its own
        shap.force_plot(
            row_explanation.base_values,
            row_explanation.values,
            row_explanation.data,
            feature_names=exp['feature_names'],
            matplotlib=True,
            show=False
        )
    plt.tight_layout()


def explain_model(model_wrapper, X_data, feature_names=None, max_samples=None, sample_indices=None, **kw):
    l = kw.get('logger', null_logger)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    if sample_indices is not None:
        X_subset = X_data.iloc[sample_indices] if hasattr(X_data, 'iloc') else X_data[sample_indices]
    else:
        # linear time
        if max_samples is None:
            max_samples = min(len(X_data), config['max-explanation-samples'])
        if max_samples and len(X_data) > max_samples:
            l.info(f"Subsampling to {max_samples} for explanation")
            indices = np.random.choice(len(X_data), max_samples, replace=False)
            X_subset = X_data.iloc[indices] if hasattr(X_data, 'iloc') else X_data[indices]
        else:
            X_subset = X_data
    explainer = _get_explainer(model_wrapper, X_data, logger=l)
    # compute SHAP values ONLY on the subset
    if isinstance(explainer, shap.KernelExplainer):
        nsamples = kw.get('nsamples', config['shap-values-number-samples']) # rule of thumb = 2*n_features + 2048
        l.info(f"Computing KernelSHAP on {len(X_subset)} samples with nsamples={nsamples}")
        shap_values = explainer.shap_values(X_subset, nsamples=nsamples)
    else:
        shap_values = explainer.shap_values(X_subset)
    # for tree explainer, shape_value = [array_classe_0, array_classe_1] so we just take value for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    # for kernel explainer, shape_value = (n_samples, n_features) 
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


@save_figure
def shap_decision(model, max_samples=50, **kw):
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
    plt.tight_layout()
    return f"{model.basename}_explained_shap-decision"


@save_figure
def shap_heatmap(model, max_display=10, **kw):
    exp = model._explanation
    plt.figure(figsize=(14, 10))
    shap.plots.heatmap(exp['shap_explanation'], max_display=max_display, show=False)
    plt.tight_layout()
    return f"{model.basename}_explained_shap-heatmap"


@save_figure
def shap_summary(model, **kw):
    exp = model._explanation
    plt.figure(figsize=(14, 10))
    shap.plots.beeswarm(exp['shap_explanation'], show=False, max_display=kw.get('max_display', 20))
    plt.tight_layout()
    return f"{model.basename}_explained_shap-summary"


@save_figure
def shap_waterfall_packed(model, **kw):
    _local_plot(model, output_path, packed=True, plot_type="waterfall", **kw)
    return f"{model.basename}_explained_shap-waterfall-packed"


@save_figure
def shap_waterfall_not_packed(model, **kw):
    _local_plot(model, output_path, packed=False, plot_type="waterfall", **kw)
    return f"{model.basename}_explained_shap-waterfall-not-packed"


@save_figure
def shap_force_packed(model, **kw):
    _local_plot(model, output_path, packed=True, plot_type="force", **kw)
    return f"{model.basename}_explained_shap-force-packed"


@save_figure
def shap_force_not_packed(model, **kw):
    _local_plot(model, output_path, packed=False, plot_type="force", **kw)
    return f"{model.basename}_explained_shap-force-not-packed"


_EXPLANATIONS = {
    'decision':             shap_decision,
    'force_not_packed':     shap_force_not_packed,
    'force_packed':         shap_force_packed,
    'heatmap':              shap_heatmap,
    'summary':              shap_summary,
    'waterfall_not_packed': shap_waterfall_not_packed,
    'waterfall_packed':     shap_waterfall_packed,
}

