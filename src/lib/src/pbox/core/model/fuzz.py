# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import re
from ...helpers import *

FEATURE_TYPE_PATTERNS = {
    r'^is_|^has_':                      (None, None, 'bool'),
    r'entropy':                         (0.0, 8.0,  'float'),
    r'^size_':                          (0, None,    'int'),
    r'^byte_\d+':                       (0, 255,     'int'),
    r'^ratio_':                         (0.0, None,  'float'),
    r'^number_|^count_|^nb_|_count$':   (0, None,    'int'),
}

def get_feature_constraints(name):
    for pattern, constraint in FEATURE_TYPE_PATTERNS.items():
        if re.search(pattern, name.lower()):
            return constraint
    return (None, None, None)

def is_boolean_feature(name):
    return get_feature_constraints(name)[2] == 'bool'

def _make_predict_proba_func(model):
    def func(data):
        return model.pipeline.predict_proba(data)[:, 1]
    return func

def _perturb_column(data, name, backup, delta_pct=None, n_sigma=None, sigma=None, sign=1):
    min_val, max_val, dtype = get_feature_constraints(name)
    if dtype == 'bool':
        data[name] = 1 - backup
        return
    col = backup.astype(float)
    if n_sigma is not None and sigma is not None:
        new = col + sign * n_sigma * sigma
    else:
        d = delta_pct or 0.1
        if dtype == 'int':
            delta = np.maximum(np.abs(col) * d, 1.0)
        else:
            delta = np.abs(col) * d
            delta = np.where(delta == 0, d, delta)
        new = col + sign * delta
    if min_val is not None or max_val is not None:
        new = np.clip(new, min_val, max_val)
    if dtype == 'int':
        new = np.round(new).astype(int)
    data[name] = new

def fuzz_single_feature(data, predict_fn, name, delta_pct=0.1, n_sigma=None):
    orig = predict_fn(data)
    backup = data[name].copy()
    if is_boolean_feature(name):
        data[name] = 1 - backup
        up = predict_fn(data)
        data[name] = backup
        return {'orig': orig, 'up': up, 'down': None,
                'feat_orig': backup.values, 'feat_up': (1 - backup).values, 'feat_down': None,
                'is_boolean': True}
    sigma = float(backup.astype(float).std()) if n_sigma else None
    if sigma is not None and sigma == 0:
        sigma = 1.0
    results = {'orig': orig, 'feat_orig': backup.values.copy(), 'is_boolean': False}
    results['mode'] = 'stddev' if n_sigma else 'delta'
    results['perturbation'] = n_sigma if n_sigma else delta_pct
    for direction, sign in [('up', 1), ('down', -1)]:
        _perturb_column(data, name, backup, delta_pct=delta_pct, n_sigma=n_sigma, sigma=sigma, sign=sign)
        results[direction] = predict_fn(data)
        results[f'feat_{direction}'] = data[name].values.copy()
    data[name] = backup
    return results

def compute_impact_per_class(fuzz_result):
    orig = fuzz_result['orig']
    breakdown = {}
    for label, mask in [('packed', orig >= 0.5), ('not_packed', orig < 0.5)]:
        n = int(mask.sum())
        if n == 0:
            breakdown[label] = {'n': 0, 'mean_up': 0.0, 'mean_down': 0.0}
            continue
        d_up = float(np.mean(fuzz_result['up'][mask] - orig[mask]))
        d_down = 0.0 if fuzz_result['is_boolean'] else float(np.mean(fuzz_result['down'][mask] - orig[mask]))
        breakdown[label] = {'n': n, 'mean_up': d_up, 'mean_down': d_down}
    return breakdown

def _max_abs_impact(entry):
    p, np = entry['packed'], entry['not_packed']
    return max(abs(p['mean_up']), abs(p['mean_down']), abs(np['mean_up']), abs(np['mean_down']))

@save_figure
def plot_fuzz_impact(model, fuzz_result, feature_name, delta_pct):
    orig, up, down = fuzz_result['orig'], fuzz_result['up'], fuzz_result['down']
    feat_orig, feat_up, feat_down = fuzz_result['feat_orig'], fuzz_result['feat_up'], fuzz_result['feat_down']
    sort_idx = np.argsort(orig)
    if fuzz_result['is_boolean']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(orig[sort_idx], 'k-', lw=1.5, label='Original')
        ax1.plot(up[sort_idx], 'r-', alpha=0.7, label='Flipped')
        ax1.set(xlabel='Sample (sorted)', ylabel='P(packed)', ylim=(-0.05, 1.05))
        ax1.legend()
        d = up - orig
        ax2.hist(d, bins=50, alpha=0.6, color='red', label=f'Flip: μ={np.mean(d):.4f}')
        ax2.axvline(0, color='k', ls='--')
        ax2.set(xlabel='ΔP(packed)', ylabel='Count')
        ax2.legend()
        plt.suptitle(f'Fuzzing: {feature_name} (boolean flip)', fontsize=14, fontweight='bold')
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        if fuzz_result.get('mode') == 'stddev':
            ds = f"{fuzz_result['perturbation']}σ"
        else:
            ds = f"{fuzz_result['perturbation']*100:.0f}%"
        ax1.plot(orig[sort_idx], 'k-', lw=1.5, label='Original')
        ax1.plot(up[sort_idx], 'r-', alpha=0.7, label=f'+{ds}')
        ax1.plot(down[sort_idx], 'b-', alpha=0.7, label=f'-{ds}')
        ax1.set(xlabel='Sample (sorted)', ylabel='P(packed)', ylim=(-0.05, 1.05))
        ax1.legend()
        for d, c, lb in [(up - orig, 'red', f'+{ds}'), (down - orig, 'blue', f'-{ds}')]:
            ax2.hist(d, bins=50, alpha=0.6, color=c, label=f'{lb}: μ={np.mean(d):.4f}')
        ax2.axvline(0, color='k', ls='--')
        ax2.set(xlabel='ΔP(packed)', ylabel='Count')
        ax2.legend()
        ax3.scatter(feat_orig, feat_up, alpha=0.3, s=10, c='red', label=f'+{ds}')
        ax3.scatter(feat_orig, feat_down, alpha=0.3, s=10, c='blue', label=f'-{ds}')
        vals = np.concatenate([feat_orig, feat_up, feat_down])
        ax3.plot([vals.min(), vals.max()], [vals.min(), vals.max()], 'k--', alpha=0.5)
        ax3.set(xlabel='Original Value', ylabel='Fuzzed Value')
        ax3.legend()
        plt.suptitle(f'Fuzzing: {feature_name} (δ={ds})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return f"{model.basename}_fuzz-impact_{feature_name}"

@save_figure
def plot_fuzz_summary(model, fuzz_results):
    top_n = config['fuzz-top-n']
    details = fuzz_results['details'][:top_n]
    names = [e['name'] for e in details]
    y = np.arange(len(names))
    h = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(names) * 0.4)), sharey=True)
    for ax, key, title in [(ax1, 'packed', 'Packed'), (ax2, 'not_packed', 'Not packed')]:
        ax.barh(y - h/2, [e[key]['mean_up'] for e in details], h, label='+δ', color='red', alpha=0.7)
        ax.barh(y + h/2, [e[key]['mean_down'] for e in details], h, label='-δ', color='blue', alpha=0.7)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel('Mean ΔP(packed)')
        ax.set_title(title)
        ax.legend()
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    plt.suptitle(f'Feature Sensitivity (Top {top_n})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return f"{model.basename}_fuzz-summary"

def fuzz_features(model, data, feature_names, delta_pct=0.1, top_n=20, export=True,
                  output_dir="fuzz_plots", use_stddev=False, n_sigma=1.0, logger=None):
    data = data[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0).copy()
    predict_fn = _make_predict_proba_func(model)
    if export:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info(f"Fuzzing {len(feature_names)} features with {'σ='+str(n_sigma) if use_stddev else 'δ='+str(int(delta_pct*100))+'%'}...")

    scores = []
    all_results = {}
    for i, name in enumerate(feature_names):
        if logger: logger.debug(f"  {i+1}/{len(feature_names)}: {name}")
        ns = n_sigma if use_stddev else None
        result = fuzz_single_feature(data, predict_fn, name, delta_pct=delta_pct, n_sigma=ns)
        all_results[name] = result
        bd = compute_impact_per_class(result)
        scores.append({'name': name, 'packed': bd['packed'], 'not_packed': bd['not_packed']})
        if export: plot_fuzz_impact(model, result, name, delta_pct)

    scores.sort(key=_max_abs_impact, reverse=True)
    return {'details': scores, 'results': all_results, 'delta_pct': delta_pct}

def multi_delta_stability(model, data, feature_names, deltas=(0.10, 0.25, 0.50, 1.00), logger=None):
    all_results = {}
    rankings = {}
    for d in deltas:
        if logger: logger.info(f"Running fuzzing at δ={d*100:.0f}%...")
        res = fuzz_features(model, data, feature_names, delta_pct=d, top_n=len(feature_names), export=False, logger=logger)
        all_results[d] = res
        rankings[d] = {e['name']: rank for rank, e in enumerate(res['details'], 1)}

    rank_df = pd.DataFrame(rankings)
    rank_df.columns = [f"δ={d*100:.0f}%" for d in deltas]
    stability = pd.DataFrame({
        'mean_rank': rank_df.mean(axis=1), 'rank_std': rank_df.std(axis=1),
        'min_rank': rank_df.min(axis=1), 'max_rank': rank_df.max(axis=1),
    })
    stability = stability.sort_values('mean_rank')
    return {'rankings': rank_df, 'stability': stability, 'all_results': all_results, 'deltas': deltas}

@save_figure
def plot_bump_chart(model, stability_result):
    top_n = min(config['fuzz-top-n'], 15)
    rank_df, stability = stability_result['rankings'], stability_result['stability']
    top = stability.head(top_n).index.tolist()
    df = rank_df.loc[top]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    colors = cm.tab20(np.linspace(0, 1, len(top)))
    x = np.arange(len(df.columns))
    for i, (feat, row) in enumerate(df.iterrows()):
        ax.plot(x, row.values, marker='o', ms=6, color=colors[i], ls='-', lw=2, alpha=1, label=feat)
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, fontsize=11)
    ax.set_ylabel('Rank')
    ax.set_xlabel('Perturbation magnitude')
    ax.set_title(f'Rank Stability (Top {top_n})', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_ylim(top_n + 1, 0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    return f"{model.basename}_fuzz-bump-chart"

def bootstrap_fuzz_ci(model, data, feature_names, n_bootstrap=100, delta_pct=0.1,
                      confidence=0.95, seed=42, logger=None):
    rng = np.random.RandomState(seed)
    n = len(data)
    impacts = np.zeros((n_bootstrap, len(feature_names)))
    for b in range(n_bootstrap):
        if logger and (b+1) % 10 == 0: logger.info(f"  Bootstrap {b+1}/{n_bootstrap}...")
        idx = rng.choice(n, size=n, replace=True)
        boot = data.iloc[idx].reset_index(drop=True)
        res = fuzz_features(model, boot, feature_names, delta_pct=delta_pct,
                            top_n=len(feature_names), export=False)
        by_name = {e['name']: _max_abs_impact(e) for e in res['details']}
        for j, name in enumerate(feature_names):
            impacts[b, j] = by_name.get(name, 0.0)

    alpha = (1 - confidence) / 2
    ci_df = pd.DataFrame({
        'feature': feature_names,
        'mean_impact': impacts.mean(axis=0),
        'ci_lower': np.percentile(impacts, alpha*100, axis=0),
        'ci_upper': np.percentile(impacts, (1-alpha)*100, axis=0),
    })
    ci_df = ci_df.sort_values('mean_impact', ascending=False).reset_index(drop=True)
    return {'ci_df': ci_df, 'bootstrap_impacts': impacts}

@save_figure
def plot_bootstrap_ci(model, ci_result):
    top_n = config['fuzz-top-n']
    df = ci_result['ci_df'].head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    y = np.arange(len(df))
    err = [np.clip(df['mean_impact'].values - df['ci_lower'].values, 0, None),
           np.clip(df['ci_upper'].values - df['mean_impact'].values, 0, None)]
    ax.barh(y, df['mean_impact'], xerr=err, alpha=0.7, capsize=3, ecolor='gray')
    ax.set_yticks(y)
    ax.set_yticklabels(df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Max Abs Impact')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_title(f'Feature Importance with 95% CI (Top {top_n})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return f"{model.basename}_fuzz-bootstrap-ci"

_DIR_COMBOS = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]

def run_interaction_analysis(model, data, feature_names, fuzz_results,
                             top_k=10, delta_pct=0.1, logger=None):
    top_names = [e['name'] for e in fuzz_results['details'][:top_k] if e['name'] in feature_names]
    k = len(top_names)
    data = data[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0).copy()
    predict_fn = _make_predict_proba_func(model)
    orig = predict_fn(data)

    if logger: logger.info(f"Computing pairwise interactions for top {k} features...")

    # Individual impacts (diagonal)
    indiv = {}
    for i, name in enumerate(top_names):
        for sign in (+1, -1):
            bak = data[name].copy()
            _perturb_column(data, name, bak, delta_pct=delta_pct, sign=sign)
            indiv[(i, sign)] = np.mean(np.abs(predict_fn(data) - orig))
            data[name] = bak
    diag = np.array([max(indiv[(i, +1)], indiv[(i, -1)]) for i in range(k)])

    # Pairwise impact 
    matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(i+1, k):
            bak_i, bak_j = data[top_names[i]].copy(), data[top_names[j]].copy()
            best = 0.0
            for si, sj in _DIR_COMBOS:
                _perturb_column(data, top_names[i], bak_i, delta_pct=delta_pct, sign=si)
                _perturb_column(data, top_names[j], bak_j, delta_pct=delta_pct, sign=sj)
                joint = np.mean(np.abs(predict_fn(data) - orig))
                best = max(best, abs(joint - indiv[(i, si)] - indiv[(j, sj)]))
                data[top_names[i]] = bak_i; data[top_names[j]] = bak_j
            matrix[i, j] = matrix[j, i] = best

    plot_interaction_heatmap(model, matrix, top_names, diag)
    return {'matrix': matrix, 'labels': top_names, 'individual_impacts': diag}

@save_figure
def plot_interaction_heatmap(model, matrix, labels, diag, **kw):
    k = len(labels)
    display = matrix.copy()
    np.fill_diagonal(display, diag)
    fig, ax = plt.subplots(figsize=(max(8, k * 0.7), max(7, k * 0.6)))
    im = ax.imshow(display, cmap='YlOrRd', aspect='equal')
    for i in range(k):
        for j in range(i + 1, k):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, fc='white', ec='white'))
    for i in range(k):
        for j in range(i + 1):
            v = display[i, j]
            if v > 0.001:
                tc = 'white' if v > np.max(display) * 0.6 else 'black'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=max(6, 10 - k//4), color=tc, fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Interaction strength')
    ax.set_title('Pairwise Feature Interactions\n(diagonal = individual)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return f"{model.basename}_fuzz-interactions"