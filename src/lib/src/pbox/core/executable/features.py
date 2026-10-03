# -*- coding: UTF-8 -*-
import builtins as bi
import yaml
from collections import deque
from tinyscript import ast, itertools, logging, re
from tinyscript.helpers import is_generator as is_gen, Path
from tinyscript.report import *
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_regression

from ...helpers import *


__all__ = ["Features"]


_NAME_REGEX = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9<>%]*(?:_[a-zA-Z0-9][a-zA-Z0-9<>%]*)*")
_SPLIT_REGEX = re.compile(r"[\s\.\,\-\+\[\]\(\)]")


# special class and method for a YAML representer used in the Features.filter method
class __FlowList(list):
    """List that YAML will render in flow style: [1, 3, 7, 9]"""
    @staticmethod
    def representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
yaml.add_representer(__FlowList, __FlowList.representer)


class Feature(dict2):
    def __init__(self, *args, **kwargs):
        super(Feature, self).__init__(*args, **kwargs)
        self['boolean'] = any(self['name'].startswith(p) for p in ["is_", "has_"])
        self.setdefault('alias', [])
        self.setdefault('keep', True)
        self.setdefault('references', [])
        # this parameter applies to dynamic features (that have a 'values' attribute)
        self.setdefault('select', "all")  # 'best': only first feature by importance selected
                                          # 'important': all important features get selected
                                          # 'all': keep all the features
        self.setdefault('significant', False)
        self.setdefault('tags', [])
    
    def __call__(self, data, *args, **kwargs):
        self._exe = data.get('executable')
        try:
            return super().__call__(data, *args, **kwargs)
        except ZeroDivisionError:  # i.e. when a ratio has its denominator set to 0 ;
            return                 #  in this case, feature's value is undefined
    
    @cached_property
    def dependencies(self):
        return list(set(x for x in _SPLIT_REGEX.split(self.result or "") if x in Features.names))
    
    @cached_property
    def fail(self):
        return self.get('fail', "error")
    
    # 'parser' parameter in the YAML config has precedence on the globally configured parser
    @cached_property
    def parser(self):
        try:
            p = self._exe.shortgroup
            delattr(self, "_exe")
        except AttributeError:
            p = "default"
        return self.get('parser', config[f'{p}_parser'])


class Features(dict, metaclass=MetaBase):
    """ This class parses the YAML definitions of features to be derived from the extracted ones.
    
    NB: On the contrary of abstractions (e.g. Packer, Detector), Features lazily computes its registry.
    """
    boolean_only = False
    names_map    = {}
    
    def __init__(self, exe=None, feature_names=None, benchmark=False, benchmark_threshold=0.):
        benchmark = benchmark or benchmark_threshold > 0.
        ft, l = Features, self.__class__.logger
        ft._load()
        if exe is not None:
            if exe.format not in ft.registry:
                raise BadFileFormat("Features extraction is not supported for this executable format")
            from .extractors import Extractors
            self._rawdata = Extractors(exe)
            todo, counts, reg = deque(), {}, ft.registry[exe.format]
            # compute features based on the extracted values first
            for name, feature in reg.items():
                # compute only if it has the keep flag not False ; otherwise, it will be lazily computed on need
                if (not ft.boolean_only or ft.boolean_only and feature.boolean) and \
                   (feature.keep and (feature_names is None or feature.name in feature_names)):
                    try:
                        v = feature(self._rawdata, True, benchmark=benchmark, benchmark_threshold=benchmark_threshold)
                        self[name] = bool(v) if feature.boolean else v
                    except NameError:
                        todo.append(feature)
                    except ForbiddenNodeError:  # already handled in dict2.__call__
                        continue
            # then lazily compute features until we converge in a state where all the required features are computed
            while len(todo) > 0:
                feature = todo.popleft()
                n = feature.name
                p = exe.parse(feature.parser, reset=False)
                # set 'binary' as the generic reference for the parsed binary but also for specific formats ('pe', ...)
                d = {'binary': p, exe.group.lower(): p}
                # add some constants
                c = {c: getattr(bi, c) for c in FEATURE_CONSTANTS}
                d.update({c: v.get(exe.format) or v.get(exe.group) or v.get('default') or v if isinstance(v, dict) \
                             else v for c, v in c.items()})
                # add raw extracted data
                d.update(self._rawdata)
                # add already computed features
                d.update(self)
                try:
                    v = feature(d, silent=feature.fail in ["continue", "warning"], benchmark=benchmark, 
                                benchmark_threshold=benchmark_threshold)
                    self[n] = bool(v) if feature.boolean else v
                except NameError:
                    bad = False
                    # every feature dependency has already been seen, but yet feature computation fails
                    if all(name2 in counts for name2 in feature.dependencies):
                        counts.setdefault(n, 0)
                        counts[n] += 1
                    else:
                        for name2 in feature.dependencies:
                            if name2 not in reg:
                                del reg[n]
                                if n in counts:
                                    del counts[n]
                                bad = True
                                break
                        if not bad:
                            for name2 in feature.dependencies:
                                # compute the dependency in priority
                                todo.appendleft(reg[name2])
                            counts.setdefault(name2, 0)
                    if counts.get(n, 0) > 10:
                        raise ValueError(f"Too much iterations of '{n}'")
                    todo.append(feature)
                except ForbiddenNodeError:  # already handled in dict2.__call__
                    continue
                except Exception as e:
                    if feature.fail == "error":
                        raise
                    elif feature.fail == "warning":
                        l.warn(f"{feature.name}: {e}")
                    elif feature.fail == "continue":
                        l.debug(f"{feature.name}: {e}")
                        self[n] = None
            # once converged, ensure that we did not leave a feature that should not be kept
            do_not_keep = []
            for name in self:
                if not reg[name].keep:
                    do_not_keep.append(name)
            for name in do_not_keep:
                del self[name]
    
    def __getitem__(self, name):
        value = super(Features, self).__getitem__(name)
        # if string, this may be a flat list/dictionary converted for working with pandas.DataFrame (cfr error:
        #  ValueError: Must have equal len keys and value when setting with an iterable)
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except ValueError:
                pass
        return value
    
    @classmethod
    def _load(cls, warn=False):
        """ Load the registry of features, that is, the Feature instances sorted per executable format. """
        ft, l = Features, cls.logger
        # parse YAML features definition once
        if ft.registry is None:
            src = ft.config  # WARNING! this line must appear BEFORE ft.registry={} because the first time that the
                             #           config attribute is called, it is initialized and the registry is reset to None
            l.debug(f"loading features from {src}...")
            ft._registry = {}
            # important note: the 'keep' parameter is not considered here as some features may be required for computing
            #                  others but not kept in the final data, hence required in the registry yet
            flist = [f for l in [["All"], [f for f in FORMATS.keys() if f != "All"], expand_formats("All")] for f in l]
            for name, params in load_yaml_config(src):
                r, values = params.pop('result', {}), params.pop('values', [])
                # allow to use 'result: ...' instead of 'result:\n  All: ...' to save space
                if not isinstance(r, dict):
                    r = {'All': r}
                # allow to use expressions in the 'values' field
                if isinstance(values, str):
                    values = list(dict2({'result': values})({'get_data': get_data}))
                # collect exclusions first
                excl = []
                for fmt in flist:
                    if fmt in r.keys() and r[fmt] is None:
                        excl.extend(expand_formats(fmt))
                # consider features for most specific formats first, then intermediate format classes and finally the
                #  collapsed format class "All"
                for fmt in flist:
                    if (expr := r.get(fmt)) is not None:
                        if len(values) > 0:
                            if not all(isinstance(x, (list, set, tuple, range, range2)) or is_gen(x) for x in values):
                                values = [values]
                            f = []
                            for val in itertools.product(*values):
                                p = {k: v for k, v in params.items()}
                                val = val[0] if isinstance(val, tuple) and len(val) == 1 else val
                                try:
                                    e = expr % val
                                except Exception as e:
                                    l.error(f"expression: {expr}")
                                    l.error(f"value:      {val}")
                                    raise
                                try:
                                    n = name % (val.lower() if isinstance(val, str) else val)
                                except TypeError:
                                    l.error(f"name:  {name}")
                                    l.error(f"value: {val}")
                                    raise
                                try:
                                    p['comment'] = p['comment'] % val
                                except (KeyError, TypeError):
                                    pass
                                try:
                                    p['description'] = (d := p['description']) % val
                                except TypeError:
                                    l.error(f"{field}: {d}")
                                    l.error(f"{'value:'.ljust(len(field))}  {val}")
                                    raise
                                f.append(Feature(p, name=n, result=e, logger=l))
                        else:
                            f = [Feature(params, name=name, result=expr, logger=l)]
                        for feat in f:
                            if feat.name != name:
                                ft.names_map[feat.name] = name
                            for subfmt in expand_formats(fmt):
                                if subfmt in excl:
                                    continue
                                ft.registry.setdefault(subfmt, {})
                                ft.registry[subfmt][feat.name] = feat
            l.debug(f"{len(Features.names)} features loaded")
        elif warn:
            l.warning(f"Features already loaded")
    
    @classproperty
    def descriptions(cls):
        if d := getattr(cls, "_descriptions", {}):
            return d
        for _, feat in cls.registry.items():
            for name, data in feat.items():
                if name not in d:
                    d[name] = data['description']
        cls._descriptions = d = {k: v for k, v in sorted(d.items())}
        return d
    
    @classmethod
    def filter(cls, dataset=None, output=None, var_threshold=.0, mi_top_k=100, scoring="matthews_corrcoef", cv=5,
               min_features=1, **kw):
        """ Filter the set of features based on multiple feature selection methods. """
        from ..dataset import Dataset
        l = cls.logger
        cls._load()
        if isinstance(dataset, str):
            dataset = Dataset(dataset)
        if dataset is None:
            raise ValueError("Dataset must be provided")
        fmt_names = {}
        for fmt, reg in cls.registry.items():
            for name, feat in reg.items():
                if feat.keep:
                    fmt_names[name] = feat
        names = sorted(fmt_names.keys())
        l.info(f"Feature set: {len(names)} features")
        X, y, skipped = [], [], 0
        if dataset._files:
            l.info("Computing features...")
        with progress_bar(target=dataset.basename) as p:
            for exe in p.track(dataset):
                try:
                    X.append([int(v) if isinstance(v := exe.data.get(n), bool) else \
                              -1. if v is None else v for n in names])
                    y.append(exe.num_label)
                except Exception as e:
                    skipped += 1
                    l.warning(f"Skipping {exe}: {e}")
        X, y = np.array(X, dtype=float), np.array(y)
        l.info(f"Computed: {len(X)} samples ({sum(y==0)} not-packed, {sum(y==1)} packed)"
                + (f", {skipped} skipped" if skipped else ""))
        if len(X) == 0:
            l.error("No samples could be processed")
            return []
        # start filtering here
        # (1) variance
        l.info(f"[1/4] Variance filter (threshold={var_threshold})")
        mask = np.var(X, axis=0) > var_threshold
        X, names = X[:, mask], [n for n, m in zip(names, mask) if m]
        l.info(f"  {len(names)} features retained (removed {sum(~mask)})")
        if len(names) == 0:
            l.error("All features removed by variance filter")
            return []
        # (2) Mutual Information (MI)
        l.info(f"[2/4] Mutual Information filter (top {mi_top_k})")
        mi_scores = mutual_info_classif(X, y, random_state=42)
        ranking = sorted(zip(names, mi_scores, range(len(names))), key=lambda x: -x[1])
        top = ranking[:min(mi_top_k, len(ranking))]
        for name, mi, _ in top[:10]:
            l.debug(f"    {name:50s} MI={mi:.4f}")
        X, names = X[:, [x[2] for x in top]], [x[0] for x in top]
        l.info(f"  {len(names)} features retained")
        # (3) Minimum Redundancy Maximum Relevance (mRMR)
        l.info("[3/4] mRMR redundancy filter")
        l.info("    Computing pairwise MI matrix...")
        n_features = X.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            mi_row = mutual_info_regression(X, X[:, i].ravel(), random_state=42)
            mi_matrix[:, i] = mi_row
        candidates = set(range(n_features)) - set(selected_idx := [int(np.argmax(mi_scores))])
        for _ in range(min(mi_top_k - 1, len(candidates))):
            best_score, best_idx = -np.inf, None
            for c in candidates:
                redundancy = np.mean([mi_matrix[c, s] for s in selected_idx])
                score = mi_scores[c] - redundancy
                if score > best_score:
                    best_score, best_idx = score, c
            if best_idx is None:
                break
            selected_idx.append(best_idx)
            candidates.remove(best_idx)
        names = [names[i] for i in selected_idx]
        l.info(f"  {len(names)} features after mRMR redundancy filter")
        X = X[:, selected_idx]
        # (4) Recursive Feature Elimination with Cross-Validation (RFECV)
        l.info(f"[4/4] RFECV (scoring={scoring}, cv={cv})")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), 
                      scoring=scoring, min_features_to_select=min_features, n_jobs=-1)
        rfecv.fit(X, y)
        mask = rfecv.support_
        selected = [n for n, m in zip(names, mask) if m]
        l.info(f"  Best CV score (MCC): {rfecv.cv_results_['mean_test_score'][rfecv.n_features_ - 1]:.4f}")
        rf_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_final.fit(X[:, mask], y)
        ranked = sorted(zip(selected, rf_final.feature_importances_), key=lambda x: -x[1])
        names = [x[0] for x in ranked]
        # finally, if relevant, only keep the first occurrence of parametrized features
        #  e.g. 
        #TODO
        # stop filtering here
        # report and save the filtered features set
        l.info(f"  {len(ranked)} features selected:")
        for i, (name, imp) in enumerate(ranked, 1):
            l.info(f"    {i:2d}. {name:50s} importance={imp:.4f}")
        if output:
            cls.select(output, names=names)
        return names
    
    @classmethod
    @file_or_folder_or_dataset
    def detect(cls, executable, **kwargs):
        """ Test features computation on a given executable, folder or dataset. """
        cls.logger.info(exe := Executable(executable))
        Features(exe)
    
    @classmethod
    def show(cls, **kw):
        """ Show an overview of the features. """
        from ...helpers.utils import pd
        Features()
        keys, ud = {'category': "category", 'ptime': "processing time", 'tcomplexity': "time complexity"}, "<undefined>"
        for k in keys.keys():
            descr = keys[k]
            cls.logger.debug(f"computing features overview per {k}...")
            formats = list(Features.registry.keys())
            # collect values
            values = []
            for fmt in formats:
                reg = pd.DataFrame.from_dict(Features.registry[fmt], orient="index")
                try:
                    for v in getattr(reg, k).unique():
                        if v not in values:
                            values.append(v)
                except AttributeError:  # no value defined for this key
                    continue
            if len(values) == 0:
                values = {ud}
            if k == 'ptime':
                values = sorted(values, key=lambda k: FEATURE_PTIME.index(k) if k in FEATURE_PTIME else -1)
            elif k == 'tcomplexity':
                values = sorted(values, key=lambda x: -1 if x == ud else \
                                                      eval(x.replace("log(n)", "2").replace("n", "100")))
            # now collect counts
            counts = {}
            for fmt in formats:
                reg = pd.DataFrame.from_dict(Features.registry[fmt], orient="index")
                for val in values:
                    counts.setdefault(val, [])
                    counts[val].append(len(reg if val == ud else reg.query(f"{k} == '{val}'")))
            render(Section(f"Counts per {descr}"), Table([[c] + v for c, v in counts.items()],
                                                         column_headers=[descr.title()] + formats))

