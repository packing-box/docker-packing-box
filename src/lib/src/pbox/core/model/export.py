# -*- coding: UTF-8 -*-
""" Export of a trained model as a standalone tool: a repository holding the strictly necessary code (pbox.portable)
     for computing the features the model uses and for running the model, with no dependency.
"""

__all__ = ["serialize_pipeline", "UnsupportedModelError"]


class UnsupportedModelError(ValueError):
    """ The model cannot be exported (algorithm, preprocessor, feature or format not supported). """


def _l(a):
    """ numpy array (or scalar) to plain Python values. """
    return a.tolist() if hasattr(a, "tolist") else a


def _tree(t):
    if t.n_outputs != 1:
        raise UnsupportedModelError("multi-output trees are not supported")
    return {'left': _l(t.children_left), 'right': _l(t.children_right), 'feature': _l(t.feature),
            'threshold': _l(t.threshold), 'value': [v[0] for v in _l(t.value)]}


def _step(obj):
    """ Plain-data description of a fitted pipeline step (see pbox.portable.runtime). """
    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, \
                                      StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    if isinstance(obj, FunctionTransformer):
        if getattr(obj.func, "__name__", None) == "_df2npa":   # pbox's identity transformer (see pbox.core.pipeline)
            return {'type': "identity"}
        raise UnsupportedModelError(f"function transformer '{obj.func}' is not supported")
    if isinstance(obj, MinMaxScaler):
        return {'type': "minmax", 'scale': _l(obj.scale_), 'min': _l(obj.min_), 'clip': obj.clip,
                'feature_range': list(obj.feature_range)}
    if isinstance(obj, StandardScaler):
        return {'type': "standard", 'mean': _l(obj.mean_) if obj.with_mean else None,
                'scale': _l(obj.scale_) if obj.with_std else None}
    if isinstance(obj, MaxAbsScaler):
        return {'type': "maxabs", 'scale': _l(obj.scale_)}
    if isinstance(obj, RobustScaler):
        return {'type': "robust", 'center': _l(obj.center_) if obj.with_centering else None,
                'scale': _l(obj.scale_) if obj.with_scaling else None}
    if isinstance(obj, Normalizer):
        return {'type': "normalizer", 'norm': obj.norm}
    if isinstance(obj, PCA):
        return {'type': "pca", 'mean': _l(obj.mean_), 'components': _l(obj.components_), 'whiten': obj.whiten,
                'explained_variance': _l(obj.explained_variance_)}
    if isinstance(obj, DecisionTreeClassifier):
        return {'type': "tree", 'classes': _l(obj.classes_), 'tree': _tree(obj.tree_)}
    if isinstance(obj, (RandomForestClassifier, ExtraTreesClassifier)):
        return {'type': "forest", 'classes': _l(obj.classes_), 'trees': [_tree(e.tree_) for e in obj.estimators_]}
    if isinstance(obj, SVC):
        if obj.kernel not in ("linear", "poly", "rbf", "sigmoid"):
            raise UnsupportedModelError(f"SVC kernel '{obj.kernel}' is not supported")
        if getattr(obj, "break_ties", False) and len(obj.classes_) > 2:
            raise UnsupportedModelError("SVC with break_ties is not supported")
        return {'type': "svc", 'classes': _l(obj.classes_), 'kernel': obj.kernel, 'gamma': float(obj._gamma),
                'coef0': float(obj.coef0), 'degree': int(obj.degree), 'support_vectors': _l(obj.support_vectors_),
                'dual_coef': _l(obj._dual_coef_), 'rho': _l(-obj._intercept_), 'n_support': _l(obj._n_support),
                'probability': len(obj._probA) > 0, 'prob_a': _l(obj._probA), 'prob_b': _l(obj._probB)}
    if isinstance(obj, MLPClassifier):
        if obj.out_activation_ not in ("logistic", "softmax"):
            raise UnsupportedModelError("only single-label MLP classifiers are supported")
        return {'type': "mlp", 'classes': _l(obj.classes_), 'activation': obj.activation,
                'out_activation': obj.out_activation_, 'coefs': [_l(c) for c in obj.coefs_],
                'intercepts': [_l(i) for i in obj.intercepts_]}
    raise UnsupportedModelError(f"'{obj.__class__.__name__}' is not supported (supported: DecisionTreeClassifier, "
                                "RandomForestClassifier, ExtraTreesClassifier, SVC, MLPClassifier, with the MinMax, "
                                "Standard, MaxAbs and Robust scalers, Normalizer and PCA)")


def serialize_pipeline(pipeline):
    """ Plain-data description of a fitted scikit-learn pipeline, for pbox.portable.runtime.predict. """
    steps = getattr(pipeline, "steps", None) or [(None, pipeline)]
    return {'steps': [_step(obj) for _, obj in steps]}


# ---------------------------------------------------------------------------------------------------- tool export
import ast as _ast
import json as _json
import re as _re
import shutil as _shutil
from datetime import date as _date
from pathlib import Path as _Path
from string import Template as _Template

from . import export_templates as _T

__all__ += ["export_tool"]

SUPPORTED_FORMATS = (".NET", "PE32", "PE64")   # formats handled by pbox.portable (PE only, for now)
_PORTABLE = _Path(__file__).resolve().parents[2].joinpath("portable")
_FAILSAFE = _re.compile(r"([a-zA-Z][a-zA-Z0-9_-]*)\*\(")
_BUILTIN_NAMES = set(dir(__import__("builtins")))


def _free_names(expr):
    """ Names an expression reads, minus those bound by its lambdas and comprehensions. """
    tree, loaded, bound = _ast.parse(expr, mode="eval"), set(), set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name):
            (loaded if isinstance(node.ctx, _ast.Load) else bound).add(node.id)
        elif isinstance(node, _ast.arg):
            bound.add(node.arg)
    return loaded - bound


def _binary_items(expr, names):
    """ Items of the parsed binary an expression accesses (binary['x'], binary.x, pe['x'], ...). """
    items = set()
    for node in _ast.walk(_ast.parse(expr, mode="eval")):
        if isinstance(node, _ast.Subscript) and isinstance(node.value, _ast.Name) and node.value.id in names and \
           isinstance(node.slice, _ast.Constant) and isinstance(node.slice.value, str):
            items.add(node.slice.value)
        elif isinstance(node, _ast.Attribute) and isinstance(node.value, _ast.Name) and node.value.id in names:
            items.add(node.attr)
    return items


def _pefeats_indices(expr):
    """ Indices of pefeats' values an expression uses (constant indices, possibly computed, e.g. pefeats[0+48]). """
    r = set()
    for node in _ast.walk(_ast.parse(expr, mode="eval")):
        if isinstance(node, _ast.Subscript) and isinstance(node.value, _ast.Name) and node.value.id == "pefeats" and \
           all(isinstance(n, (_ast.Constant, _ast.BinOp, _ast.UnaryOp, _ast.operator, _ast.unaryop))
               for n in _ast.walk(node.slice)):
            try:
                v = eval(compile(_ast.Expression(node.slice), "<index>", "eval"), {'__builtins__': {}})
            except Exception:
                continue
            if isinstance(v, int):
                r.add(v)
    return r


def _pefeats_name(i):
    from ...portable.extractors.pefeats import FEATURES
    return f"{FEATURES[i]} ({i})" if 0 <= i < len(FEATURES) else str(i)


def _slug(name):
    s = _re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "tool"
    return s if s[0].isalpha() else "tool-" + s


def _resolve_features(names, formats, logger):
    """ Definitions of the features (with their dependencies) for each format, and what they need. Raises
         UnsupportedModelError with the list of what cannot be exported. """
    import builtins as bi
    from ..executable.extractors import Extractors
    from ..executable.features import Features
    from ...helpers.data import get_data
    from ...helpers.formats import get_format_group
    from ...portable.expressions import HELPERS
    from ...portable.parsers.pe import PE as PortablePE, Section as PortableSection
    Features()
    Extractors(None)
    problems, result = [], {}
    for fmt in formats:
        reg = Features.registry.get(fmt, {})
        raw_names = set(Extractors.registry.get(fmt, {}))
        group = get_format_group(fmt).lower()
        binary_names = {"binary", group}
        consts = {}
        for c in bi.FEATURE_CONSTANTS:
            v = getattr(bi, c)
            consts[c] = (v.get(fmt) or v.get(get_format_group(fmt)) or v.get('default') or v) if isinstance(v, dict) \
                        else v
        defs, todo, helpers, items, indices = {}, list(names), set(), set(), set()
        while todo:
            n = todo.pop()
            if n in defs:
                continue
            if n not in reg:
                problems.append(f"[{fmt}] feature '{n}' is not defined (check the features definitions of the workspace "
                                "in which the model was trained)")
                continue
            f = reg[n]
            expr = _FAILSAFE.sub(r"failsafe(\1)(", str(f.result))
            try:
                free = _free_names(expr)
            except SyntaxError as e:
                problems.append(f"[{fmt}] {n}: bad expression ({e})")
                continue
            deps = sorted(free & set(reg))
            other = free - set(reg) - raw_names - binary_names - set(consts) - set(HELPERS)
            if other:
                problems.append(f"[{fmt}] {n}: unsupported name(s) {', '.join(sorted(other))} in '{expr}'")
            if free & binary_names:
                for it in _binary_items(expr, free & binary_names):
                    if not hasattr(PortablePE, it):
                        problems.append(f"[{fmt}] {n}: '{it}' of the parsed binary is not available without LIEF "
                                        f"(pbox.portable) in '{expr}'")
                    items.add(it)
            if "pefeats" in free:
                indices |= _pefeats_indices(expr)
            helpers |= free & set(HELPERS)
            # first pass of pbox (raw data only) if the expression needs nothing else than raw data and helpers
            is_raw = not (free - raw_names - set(HELPERS) - (_BUILTIN_NAMES & free))
            defs[n] = {'expr': expr, 'boolean': bool(f.boolean), 'fail': f.fail, 'raw': is_raw and not deps,
                       'deps': deps}
            todo.extend(deps)
        all_free = set().union(*(_free_names(d['expr']) for d in defs.values())) if defs else set()
        data = get_data(fmt)
        result[fmt] = {
            'definitions': defs, 'helpers': sorted(helpers), 'group': group,
            'pefeats': "pefeats" in all_free, 'pefeats_indices': sorted(indices),
            'parser': bool(all_free & binary_names), 'binary_items': sorted(items),
            'constants': {c: v for c, v in consts.items() if c in all_free},
            'section_names': (list(data.get('STANDARD_SECTION_NAMES', [])),
                              list(data.get('COMMON_PACKER_SECTION_NAMES', []))),
        }
    if problems:
        # one line per problem, with the formats it concerns
        grouped = {}
        for p in problems:
            fmt, msg = p[1:].split("] ", 1)
            grouped.setdefault(msg, []).append(fmt)
        raise UnsupportedModelError("the features of this model cannot be exported:\n- " +
                                    "\n- ".join(f"{msg} [{', '.join(fmts)}]" for msg, fmts in grouped.items()))
    return result


def _copy_portable(dst, needs):
    """ Copy the strictly necessary modules of pbox.portable. """
    files = ["expressions.py", "formats.py", "runtime/__init__.py"] + [f"runtime/{m}.py" for m in needs['runtime']]
    if needs['parser'] or "ent" in needs['helpers']:
        files.append("primitives.py")
    if needs['parser']:
        files += ["parsers/__init__.py", "parsers/pe.py"]
    if needs['pefeats']:
        files += ["extractors/__init__.py", "extractors/pefeats.py"]
    for f in files:
        d = dst.joinpath(f)
        d.parent.mkdir(parents=True, exist_ok=True)
        _shutil.copy(_PORTABLE.joinpath(f), d)
    dst.joinpath("__init__.py").write_text('# -*- coding: UTF-8 -*-\n""" Code from pbox.portable (the Packing Box): '
                                           'feature extraction and model inference without dependency. """\n')
    return sorted(files)


def _runtime_modules(model):
    m = set()
    for s in model['steps']:
        t = s['type']
        m.add("preprocessing" if t in ("identity", "minmax", "standard", "maxabs", "robust", "normalizer", "pca") else
              "tree" if t in ("tree", "forest") else "svm" if t == "svc" else t)
    return sorted(m)


def _py(name, value):
    return f"{name} = {value!r}\n"


def _sample_files(dataset_path, samples, n):
    """ Executables to validate the feature extraction on: the given ones, or n files of the dataset. """
    import random
    files = []
    for s in samples or []:
        p = _Path(str(s))
        files.extend(sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p])
    if not files and dataset_path is not None and dataset_path.joinpath("files").is_dir():
        files = sorted(f for f in dataset_path.joinpath("files").iterdir() if f.is_file())
    random.Random(0).shuffle(files)
    return files[:n]


def export_tool(model_path, output=None, name=None, samples=None, n_samples=30, n_rows=200, force=False,
                dataset=None, logger=None):
    """ Export a trained model as a standalone tool repository.
    
    :param model_path: folder of the model (with dump.joblib, features.json, metadata.json)
    :param output:     folder of the repository to create (default: ./<name>)
    :param name:       name of the tool (default: derived from the model's name)
    :param samples:    executables (or folders) to validate the feature extraction on (default: files of the dataset)
    :param n_samples:  maximum number of executables to validate the feature extraction on
    :param n_rows:     maximum number of dataset rows to validate the model on
    :param force:      write into an existing non-empty folder, and keep the repository even if validation fails
    :param dataset:    folder of the training dataset (default: the dataset of the model in the workspace)
    :return:           path of the repository and validation report
    """
    import joblib
    import numpy as np
    import pandas as pd
    from ...helpers.formats import expand_formats
    log = logger or null_logger
    model_path = _Path(str(model_path))
    md = _json.loads(model_path.joinpath("metadata.json").read_text())
    descriptions = _json.loads(model_path.joinpath("features.json").read_text())
    pipeline = joblib.load(str(model_path.joinpath("dump.joblib")))
    features = sorted(descriptions)   # pbox orders the columns alphabetically
    first = pipeline.steps[0][1] if hasattr(pipeline, "steps") else pipeline
    if hasattr(first, "feature_names_in_") and list(first.feature_names_in_) != features:
        raise UnsupportedModelError("the features of the model do not match the columns it was fitted on")
    # model
    model = serialize_pipeline(pipeline)
    clf = model['steps'][-1]
    # formats
    formats = sorted(set(expand_formats(*md['dataset'].get('formats', []))))
    unsupported = [f for f in formats if f not in SUPPORTED_FORMATS]
    if unsupported:
        raise UnsupportedModelError(f"format(s) not supported yet: {', '.join(unsupported)} (supported: "
                                    f"{', '.join(SUPPORTED_FORMATS)})")
    log.info(f"Resolving the {len(features)} features for {', '.join(formats)}...")
    resolved = _resolve_features(features, formats, log)
    needs = {'runtime': _runtime_modules(model), 'parser': any(r['parser'] for r in resolved.values()),
             'pefeats': any(r['pefeats'] for r in resolved.values()),
             'helpers': set().union(*(r['helpers'] for r in resolved.values()))}
    # repository
    tool = _slug(name or model_path.name)
    package = tool.replace("-", "_")
    out = _Path(str(output or tool)).absolute()
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"'{out}' is not empty (use force to write into it)")
    src = out.joinpath("src", package)
    src.mkdir(parents=True, exist_ok=True)
    copied = _copy_portable(src.joinpath("portable"), needs)
    binary = set(map(str, clf['classes'])) <= {"0", "1"}
    labels = {"0": "not packed", "1": "packed"} if binary else \
             {str(c): {"-": "not packed", "?": "not labelled"}.get(str(c), str(c)) for c in clf['classes']}
    algo, ds = md.get('algorithm', {}), md.get('dataset', {})
    title = f"{tool} ({algo.get('description', algo.get('name', 'model'))})"
    description = f"Packing detection with a {algo.get('description', algo.get('name'))} model trained with the " \
                  f"Packing Box" + ("" if binary else " (packer identification)")
    subst = {'title': title, 'description': description, 'tool': tool, 'package': package, 'version': "1.0.0",
             'model': model_path.name, 'date': _date.today().isoformat(), 'n_features': len(features)}
    t = lambda tpl: _Template(tpl).substitute(subst)
    src.joinpath("__init__.py").write_text(t(_T.INIT))
    src.joinpath("__main__.py").write_text(t(_T.MAIN))
    # generated data: features and model
    defs = {fmt: r['definitions'] for fmt, r in resolved.items()}
    src.joinpath("features.py").write_text(
        '# -*- coding: UTF-8 -*-\n""" Features of the model, generated by the Packing Box from its features definitions '
        '(only those the model\n     needs, with their dependencies). """\n\n' +
        _py("FEATURES", features) + _py("DESCRIPTIONS", {n: descriptions[n] for n in features}) +
        _py("HELPERS", sorted(needs['helpers'])) +
        _py("GROUPS", {fmt: r['group'] for fmt, r in resolved.items()}) +
        _py("PARSER", {fmt: r['parser'] for fmt, r in resolved.items()}) +
        _py("PEFEATS", {fmt: r['pefeats'] for fmt, r in resolved.items()}) +
        _py("CONSTANTS", {fmt: r['constants'] for fmt, r in resolved.items()}) +
        _py("SECTION_NAMES", {fmt: r['section_names'] for fmt, r in resolved.items() if r['parser']}) +
        "DEFINITIONS = {\n" + "".join(f"    {fmt!r}: {{\n" + "".join(f"        {n!r}: {d!r},\n" for n, d in
                                      sorted(ds_.items())) + "    },\n" for fmt, ds_ in sorted(defs.items())) + "}\n")
    src.joinpath("model.py").write_text(
        '# -*- coding: UTF-8 -*-\n""" Model, generated by the Packing Box from the trained pipeline (scikit-learn), as '
        'plain data for\n     portable.runtime. """\n\n' + _py("LABELS", labels) + _py("MODEL", {**model,
        'features': features}))
    out.joinpath("run.py").write_text(t(_T.RUN))
    out.joinpath("pyproject.toml").write_text(t(_T.PYPROJECT))
    out.joinpath("Makefile").write_text(t(_T.MAKEFILE))
    out.joinpath(".gitignore").write_text(_T.GITIGNORE)
    out.joinpath(".github", "workflows").mkdir(parents=True, exist_ok=True)
    out.joinpath(".github", "workflows", "build.yml").write_text(t(_T.WORKFLOW))
    lic = _Path(__file__).resolve().parents[4].joinpath("LICENSE")
    if lic.is_file():
        _shutil.copy(lic, out.joinpath("LICENSE"))
    # validation
    report = _validate(out, package, pipeline, model, features, md, samples, n_samples, n_rows, dataset, log)
    out.joinpath("tests", "data").mkdir(parents=True, exist_ok=True)
    out.joinpath("tests", f"test_{package}.py").write_text(t(_T.TESTS))
    out.joinpath("tests", "data", "expected.json").write_text(_json.dumps(report.pop('expected'), indent=1) + "\n")
    # README (model card)
    _readme(out, subst, md, model_path, features, descriptions, resolved, labels, clf, report, copied)
    ok = report['model']['mismatches'] == 0 and report['files']['mismatches'] == 0
    if not ok and not force:
        bad = out.with_name(out.name + ".failed")
        if bad.exists():
            _shutil.rmtree(bad)
        out.rename(bad)
        raise UnsupportedModelError(f"validation failed (see {bad.joinpath('README.md')}):\n" +
                                    "\n".join(report['problems'][:20]))
    return out, report


def _validate(out, package, pipeline, model, features, md, samples, n_samples, n_rows, dataset, log):
    """ Compare the exported tool with scikit-learn (model) and pbox (features, format, prediction). """
    import importlib
    import sys
    import numpy as np
    import pandas as pd
    report = {'model': {'rows': 0, 'mismatches': 0, 'max_proba_diff': 0.}, 'problems': [],
              'files': {'checked': 0, 'mismatches': 0, 'skipped': 0}, 'expected': {'model': [], 'files': {}}}
    sys.path.insert(0, str(out.joinpath("src")))
    for m in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
        del sys.modules[m]
    dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # no __pycache__ in the repository
    tmp = __import__("tempfile").TemporaryDirectory(prefix="pbox-export-")
    try:
        tool = importlib.import_module(package)
        runtime = importlib.import_module(package + ".portable.runtime")
        classes = model['steps'][-1]['classes']
        has_proba = hasattr(pipeline, "predict_proba") and model['steps'][-1].get('probability', True)
        # 1. model, on rows of the training dataset (from its data.csv)
        ds_path = _Path(str(dataset)) if dataset else _Path(str(config['datasets'].joinpath(md['dataset']['name'])))
        csv = _Path(str(ds_path)).joinpath("data.csv")
        X = None
        if csv.is_file():
            df = pd.read_csv(csv, sep=";", low_memory=False)
            if all(f in df.columns for f in features):
                X = df[features].fillna(-1)
                if len(X) > n_rows:
                    X = X.sample(n_rows, random_state=0)
        if X is None:
            report['problems'].append("model not checked on dataset rows (dataset or features not found)")
            log.warning(report['problems'][-1])
        else:
            log.info(f"Checking the model on {len(X)} rows of the dataset...")
            pred = pipeline.predict(X)
            proba = pipeline.predict_proba(X) if has_proba else None
            for i, (_, row) in enumerate(X.iterrows()):
                x = [float(v) for v in row.tolist()]
                cls, p = runtime.predict(model, x)
                report['model']['rows'] += 1
                if str(cls) != str(pred[i]):
                    report['model']['mismatches'] += 1
                    report['problems'].append(f"row {row.name}: class {cls} instead of {pred[i]}")
                if proba is not None:
                    d = max(abs(p[c] - proba[i][k]) for k, c in enumerate(classes))
                    report['model']['max_proba_diff'] = max(report['model']['max_proba_diff'], float(d))
                    if d > 1e-9:
                        report['model']['mismatches'] += 1
                        report['problems'].append(f"row {row.name}: probabilities differ by {d:.2e}")
                report['expected']['model'].append({'id': str(row.name), 'x': x, 'class': _l(pred[i]),
                                                    'probability': None if proba is None else _l(proba[i])})
        # 2. features, format and prediction, on executables, against pbox with the portable parser
        files = _sample_files(_Path(str(ds_path)), samples, n_samples)
        if not files:
            report['problems'].append("feature extraction not checked (no executable: give samples or use a dataset "
                                      "with files)")
            log.warning(report['problems'][-1])
        else:
            log.info(f"Checking the feature extraction on {len(files)} executables...")
            from ..executable import Executable
            from ..executable.features import Features
            previous = config['pe_parser']
            config['pe_parser'] = "portable"   # in memory only ; same values as LIEF + pefeats (see the parity tests)
            try:
                for f in files:
                    data = f.read_bytes()
                    # work on a copy: an executable inside a dataset's folder would be bound to the dataset
                    copy = _Path(tmp.name).joinpath(f.name)
                    copy.write_bytes(data)
                    exe = Executable(str(copy))
                    try:
                        fmt = tool.portable.formats.detect(data)
                        if exe.format not in tool.FORMATS:
                            report['files']['skipped'] += 1
                            continue
                        ref = Features(exe, feature_names=features)
                        ref_x = tool.vector({n: ref.get(n) for n in features})
                    except Exception as e:
                        report['files']['skipped'] += 1
                        log.debug(f"{f.name}: pbox could not compute the features ({e})")
                        continue
                    report['files']['checked'] += 1
                    try:
                        r = tool.predict(data, with_features=True)
                        x = tool.vector(r['features'])
                    except Exception as e:
                        report['files']['mismatches'] += 1
                        report['problems'].append(f"{f.name}: the tool failed ({e})")
                        continue
                    diff = [n for n, a, b in zip(features, x, ref_x) if a != b]
                    expected_cls = pipeline.predict(pd.DataFrame([ref_x], columns=features))[0]
                    if fmt != exe.format or diff or str(r['class']) != str(expected_cls):
                        report['files']['mismatches'] += 1
                        report['problems'].append(f"{f.name}: format {fmt} (pbox: {exe.format}), class {r['class']} "
                                                  f"(pbox: {expected_cls}), features differing: {', '.join(diff)}")
                    report['expected']['files'][__import__("hashlib").sha256(data).hexdigest()] = \
                        {'format': fmt, 'x': x, 'class': _l(r['class'])}
            finally:
                config['pe_parser'] = previous
    finally:
        tmp.cleanup()
        sys.dont_write_bytecode = dont_write
        sys.path.remove(str(out.joinpath("src")))
        for m in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
            del sys.modules[m]
    return report


def _readme(out, subst, md, model_path, features, descriptions, resolved, labels, clf, report, copied):
    algo, ds = md.get('algorithm', {}), md.get('dataset', {})
    perf = ""
    p = model_path.joinpath("performance.csv")
    if p.is_file():
        import pandas as pd
        try:
            df = pd.read_csv(p, sep=";")
            if len(df) > 0:
                perf = "### Performance (as recorded by the Packing Box)\n\n" + \
                       df.to_markdown(index=False) + "\n\n"
        except Exception:
            pass
    rows = "\n".join(f"| `{n}` | {descriptions[n]} |" for n in features)
    src = []
    for fmt, r in sorted(resolved.items()):
        what = [x for x, b in (("the PE parser", r['parser']), ("pefeats (Python port)", r['pefeats'])) if b]
        src.append(f"- {fmt}: {len(r['definitions'])} definitions computed (with dependencies), using "
                   f"{' and '.join(what) or 'no extractor'}" +
                   (f" ; pefeats values (0-based index): {', '.join(_pefeats_name(i) for i in r['pefeats_indices'])}"
                    if r['pefeats_indices'] else "") +
                   (f" ; parsed binary: {', '.join(r['binary_items'])}" if r['binary_items'] else ""))
    sources = "### Embedded code\n\n" + "\n".join(src) + "\n- Modules of `pbox.portable` embedded: " + \
              ", ".join(f"`{c}`" for c in copied)
    m, f = report['model'], report['files']
    val = [f"- Model: {m['rows']} rows of the training dataset, {m['mismatches']} mismatch(es) with scikit-learn "
           f"(maximum difference of probabilities: {m['max_proba_diff']:.1e})" if m['rows'] else
           "- Model: not checked (dataset rows not available)",
           f"- Feature extraction: {f['checked']} executables, {f['mismatches']} mismatch(es) with the Packing Box "
           f"(format, features, prediction){f' ; {f['skipped']} skipped' if f['skipped'] else ''}" if f['checked'] else
           "- Feature extraction: not checked (no executable available)"]
    if report['problems']:
        val.append("\nProblems:\n\n" + "\n".join(f"- {x}" for x in report['problems'][:50]))
    params = ", ".join(f"`{k}={v}`" for k, v in sorted(algo.get('parameters', {}).items()))
    counts = ", ".join(f"{labels.get(str(k), {'-': 'not packed', '?': 'not labelled'}.get(str(k), k))}: {v}"
                       for k, v in ds.get('counts', {}).items())
    readme = _Template(_T.README).substitute(
        subst, algorithm=algo.get('description', algo.get('name')), parameters=params or "-",
        preprocessors=", ".join(algo.get('preprocessors', [])) or "none",
        classes=", ".join(f"`{c}` ({labels[str(c)]})" for c in clf['classes']), formats=", ".join(sorted(resolved)),
        dataset=ds.get('name', "?"), executables=ds.get('executables', "?"), counts=counts or "?",
        performance=perf, features_table="| Feature | Description |\n| --- | --- |\n" + rows, sources=sources,
        validation="\n".join(val), example_label=list(labels.values())[-1])
    out.joinpath("README.md").write_text(readme)
