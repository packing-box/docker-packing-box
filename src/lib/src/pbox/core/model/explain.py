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
        l.info(f"Using TreeExplainer for {clf_name}")
        return shap.TreeExplainer(inner_clf)
    if clf_name in linear_models:
        l.info(f"Using LinearExplainer for {clf_name}")
        if hasattr(predictor, 'steps') and len(predictor.steps) > 1:
            X_scaled = X_background.copy()
            for name, step in predictor.steps[:-1]:
                X_scaled = step.transform(X_scaled)
            return shap.LinearExplainer(inner_clf, X_scaled), predictor.steps[:-1]
        return shap.LinearExplainer(inner_clf, X_background), None
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
    target = int(packed)
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


def _local_plot(model, packed, sample_idx=None, plot_type="waterfall", max_display=10, **kw):
    exp = model._explanation
    if sample_idx is not None:
        row_explanation = exp['shap_explanation'][sample_idx]
    else:
        row_explanation = exp['shap_explanation'][_get_sample_idx(exp, packed=packed)]
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
    result = _get_explainer(model_wrapper, X_data, logger=l)
    if isinstance(result, tuple):
        explainer, preprocess_steps = result
    else:
        explainer, preprocess_steps = result, None
    X_explain = X_subset
    if preprocess_steps is not None:
        X_explain = X_subset.copy()
        for name, step in preprocess_steps:
            X_explain = step.transform(X_explain)
    # compute SHAP values ONLY on the subset
    if isinstance(explainer, shap.KernelExplainer):
        nsamples = kw.get('nsamples', config['shap-values-number-samples']) # rule of thumb = 2*n_features + 2048
        l.info(f"Computing KernelSHAP on {len(X_subset)} samples with nsamples={nsamples}")
        shap_values = explainer.shap_values(X_explain, nsamples=nsamples)
    else:
        shap_values = explainer.shap_values(X_explain)
    # for tree explainer, shap_value = [array_classe_0, array_classe_1] so we just take value for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    # for kernel explainer, shape_value = (n_samples, n_features) 
    elif hasattr(shap_values, 'shape') and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    shap_values = np.array(shap_values, dtype=np.float64)
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
def shap_decision(model, max_display=10, max_samples=50, **kw):
    plt.figure(figsize=(14, 10))
    exp = model._explanation
    n = min(max_samples, len(exp['shap_values']))
    sv = exp['shap_values'][:n]
    # select top features by mean absolute SHAP value, decision_plot doesn't have a max_display parameter
    importance = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(importance)[-max_display:]
    shap.decision_plot(
        exp['expected_value'],
        sv[:, top_idx],
        exp['data'].iloc[:n, top_idx] if hasattr(exp['data'], 'iloc') else exp['data'],
        feature_names=[exp['feature_names'][i] for i in top_idx],
        show=False
    )
    plt.tight_layout()
    return f"{model.basename}/explained_shap-decision"


@save_figure
def shap_heatmap(model, max_display=10, **kw):
    exp = model._explanation
    plt.figure(figsize=(14, 10))
    shap.plots.heatmap(exp['shap_explanation'], max_display=max_display, show=False)
    plt.tight_layout()
    return f"{model.basename}/explained_shap-heatmap"


@save_figure
def shap_summary(model, max_display=10, **kw):
    exp = model._explanation
    plt.figure(figsize=(14, 10))
    shap.plots.beeswarm(exp['shap_explanation'], show=False, max_display=max_display)
    plt.tight_layout()
    return f"{model.basename}/explained_shap-summary"

@save_figure
def shap_waterfall_packed(model, sample_idx=None, **kw):
    _local_plot(model, packed=True, sample_idx=sample_idx, plot_type="waterfall", **kw)
    if sample_idx is not None:
        return f"{model.basename}/explained_shap-waterfall-sample{sample_idx}"
    return f"{model.basename}/explained_shap-waterfall-packed"


@save_figure
def shap_waterfall_not_packed(model, sample_idx=None, **kw):
    _local_plot(model, packed=False, sample_idx=sample_idx, plot_type="waterfall", **kw)
    return f"{model.basename}/explained_shap-waterfall-not-packed"


@save_figure
def shap_force_packed(model, sample_idx=None, **kw):
    _local_plot(model, packed=True, sample_idx=sample_idx, plot_type="force", **kw)
    if sample_idx is not None:
        return f"{model.basename}/explained_shap-force-sample{sample_idx}"
    return f"{model.basename}/explained_shap-force-packed"


@save_figure
def shap_force_not_packed(model, sample_idx=None, **kw):
    _local_plot(model, packed=False, sample_idx=sample_idx, plot_type="force", **kw)
    return f"{model.basename}/explained_shap-force-not-packed"

@save_figure
def shap_taxonomy_donut(model, **kw):
    exp = model._explanation
    profile = exp.get('category_profile', {})
    if not profile:
        return None
    
    cats = list(profile.keys())
    vals = list(profile.values())
    
    filtered = [(c, v) for c, v in zip(cats, vals) if v >= 0.02]
    other = sum(v for _, v in zip(cats, vals) if v < 0.02)
    if other > 0:
        filtered.append(('Other', other))
    cats_f, vals_f = zip(*filtered)
    
    colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261',
              '#e76f51', '#606c38', '#dda15e', '#bc6c25', '#023047']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        vals_f, labels=cats_f, autopct='%1.0f%%',
        colors=colors[:len(vals_f)], pctdistance=0.8,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    ax.set_title(f'Model taxonomy of {model.basename}', size=13, fontweight='bold')
    plt.tight_layout()
    return f"{model.basename}/explained_shap-taxonomy"

_EXPLANATIONS = {
    'decision':             shap_decision,
    'force_not_packed':     shap_force_not_packed,
    'force_packed':         shap_force_packed,
    'heatmap':              shap_heatmap,
    'summary':              shap_summary,
    'taxonomy':             shap_taxonomy_donut,
    'waterfall_not_packed': shap_waterfall_not_packed,
    'waterfall_packed':     shap_waterfall_packed,
}

