# -*- coding: UTF-8 -*-
from tinyscript import json, itertools
from tinyscript.helpers import human_readable_size, is_generator, Path, TempPath
from tinyscript.report import *

from .algorithm import *
from .algorithm import __all__ as _algo
from .metrics import *
from .metrics import __all__ as _metrics
from ..dataset import *
from ..executable import *
from ..pipeline import *
from ...helpers import *

lazy_load_module("joblib")


__all__ = ["DumpedModel", "Model"] + _algo + _metrics


FLOAT_FORMAT = "%.6f"


class BaseModel(Entity):
    """ Base class for a model. """
    DEFAULTS = {
        '_features': {},
        '_metadata': {},
        '_performance': lazy_object(lambda: pd.DataFrame()),
        'pipeline': lazy_object(lambda: DebugPipeline()),
    }
    
    def __len__(self):
        """ Get the length of model's pipeline. """
        return len(self.pipeline.steps)
    
    def _metrics(self, data, prediction, target=None, proba=None, metrics="classification", proctime=None,
                 ignore_labels=False):
        """ Metrics computation method. """
        l, mfunc = self.logger, f"{metrics}_metrics"
        if mfunc not in globals():
            l.error(f"Bad metrics type '{metrics}'")
            return
        m = globals()[mfunc]
        l.debug(f"Computing {metrics} metrics...")
        df2np = lambda df: df.to_numpy() if isinstance(df, pd.DataFrame) else df
        fix = lambda v: df2np(v) if getattr(df2np(v), "ndim", 1) == 1 else df2np(v).transpose()[0]
        values, headers = m(data, fix(prediction), y_true=fix(target), y_proba=fix(proba), proctime=proctime,
                            ignore_labels=ignore_labels, logger=self.logger)
        return [f(v) for v, f in zip(values, headers.values())], list(headers.keys())
    
    def _prepare(self, dataset=None, preprocessor=None, multiclass=False, labels=None, data_only=False,
                 unlabelled=False, mi_select=False, mi_kbest=None, true_class=None, **kw):
        """ Prepare the Model instance based on the given Dataset/FilelessDataset/CSV/other instance.
        NB: after preparation,
             (1) input data is prepared (NOT preprocessed yet as this is part of the pipeline), according to 4 use cases
                  - FilelessDataset (features already computed)
                  - Dataset (features to be computed yet)
                  - CSV file (assumes features from this model are present)
                  - Others (single executable or folder)
             (2) if training, the learning pipeline is created (with preprocessing steps only),
                 otherwise it will be loaded via self._load() from the model (including the selected model)
             (3) if training, model's metadata is populated AND training/test subsets are established
        IMPORTANT: data_only is necessary as self._load() occurs AFTER self._prepare(...) ; this is due to the fact that
                    self.name includes the number of features, which requires data preparation """
        from ast import literal_eval
        ds, l, labels = dataset, self.logger, labels or {}
        # if not only the data shall be prepared, then the only supported input format is a valid dataset ;
        #  i.e. when preparing data for the train() method
        if not data_only:
            if not getattr(ds, "is_valid", lambda: False)():
                l.error(f"{dataset} is not a valid input dataset")
                return False
            # copy relevant information from the input dataset (which is the reference one for the trained model)
            l.debug("Preparing dataset...")
            self._metadata['dataset'] = {'name': ds.path.stem, 'path': str(ds.path)}
            self._metadata['dataset'].update({k: v for k, v in ds._metadata.items()})
        # if using data only (thus not training), this could be for preprocessing and visualizing features ; then the
        #  model's reference dataset is to be used
        elif ds is None:
            ds = self._metadata['dataset']['name']
        # at this point, we may either have a valid dataset or another source format ; if string, this shall be a path
        if isinstance(ds, str):
            ds = Path(ds)
            if not ds.is_absolute():
                try:
                    ds = Dataset.load(ds)
                except ValueError:
                    pass
            if not ds.exists():
                l.error(f"Bad input dataset ({ds})")
                return False
        self._dataset = ds
        l.info(f"{kw.get('ds_type', ['Reference','Test'][data_only])} dataset:  {ds}")
        self._data, self._target = pd.DataFrame(), pd.DataFrame(columns=["label"])
        Features.boolean_only = self.algorithm.boolean
        # start input dataset parsing
        def __parse(exes, label=True):
            l.info("Computing features...")
            if not isinstance(exes, list) and not is_generator(exes):
                exes = [exes]
            exes, cnt = itertools.tee(exes)
            n = sum(1 for _ in cnt)
            with progress_bar(silent=n <= 1) as p:
                task = p.add_task("", total=n)
                for exe in exes:
                    if not isinstance(exe, Executable):
                        exe = Executable(str(exe))
                    if not data_only:
                        self._features.update(exe.features)
                    self._data = pd.concat([self._data, pd.DataFrame.from_records([exe.data])], ignore_index=True)
                    if label:
                        d = {'label': labels.get(exe.hash)}
                        self._target = pd.concat([self._target, pd.DataFrame.from_records([d])], ignore_index=True)
                    p.update(task, advance=1.)
        # case 1: fileless dataset (where features are already computed)
        if isinstance(ds, FilelessDataset):
            l.info("Loading features...")
            for exe in ds:  # take the very first sample
                break
            exe = Executable(exe)
            if not data_only:
                self._features.update(exe.features)
            self._data = ds._data[list(exe.features.keys())]
            self._target = ds._data.loc[:, ds._data.columns == "label"]
        # case 2: normal dataset (features shall still be computed)
        elif isinstance(ds, Dataset):
            __parse(ds.files.listdir(is_exe), False)
            self._target = ds._data.loc[:, ds._data.columns == "label"]
        # case 3: CSV file
        elif ds.extension == ".csv":
            l.info("Loading features...")
            try:
                d, n = pd.read_csv(str(ds), sep=kw.pop('sep', ",")), Path(ds).filename
                self._data, self._target = d.loc[:, d.columns != "label"], d.loc[:, d.columns == "label"]
                if not data_only:
                    self._features = {k: "" for k in self._data.columns}
            except AttributeError:  # 'DataFrame' object has no attribute 'label'
                l.error(d.columns)
                l.warning("This error may be caused by a bad CSV separator ; you can set it with --sep")
                raise
        # case 4: other data, either single executable or folder (only valid when not training)
        elif data_only and (is_exe(ds) or ds.is_dir()):
            self._data, self._target = pd.DataFrame(), pd.DataFrame()
            __parse([ds] if is_exe(ds) else ds.listdir(is_exe))
        # this shall not occur
        else:
            raise ValueError("Unsupported input format")
        if len(self._data) == 0:
            l.warning("No data")
            return False
        if len(self._features) == 0:
            l.warning("No selectable feature ; this may be due to a model unrelated to the input")
            return False
        # ensure features are sorted and data has its columns sorted too
        self._features = {k: v for k, v in sorted(self._features.items(), key=lambda x: x[0]) if v != ""}
        try:
            self._data = self._data[sorted(self._features.keys())]
        except KeyError as e:
            missing_cols = literal_eval(e.args[0].replace("not in index", ""))
            for col in missing_cols:
                self._features[col] = np.nan
            self._data = self._data.reindex(columns=sorted(self._features.keys()))
        # if data has to be unlabelled, force values for column 'label'
        if unlabelled:
            self._target['label'] = NOT_LABELLED
        # fill missing values
        self._data, self._target = self._data.fillna(-1), self._target.fillna(NOT_LABELLED)
        # convert to binary class
        if not multiclass:
            self._target = (self._target['label'] == true_class).astype('int') if true_class is not None else \
                           self._target.map(lambda x: LABELS_BACK_CONV.get(x, 1)).astype('int')
        # create the pipeline if it does not exist (i.e. while training)
        if not data_only:
            l.info("Making pipeline...")
            make_pipeline(self.pipeline, preprocessor, self.logger)
        # if only data is to be processed (i.e. while testing), stop here, the rest is for training the model
        else:
            return True
        # apply variance threshold of 0.0 to remove useless features and rectify the list of features
        from sklearn.feature_selection import VarianceThreshold
        l.debug("> remove 0-variance features")
        selector = VarianceThreshold()
        selector.fit(self._data)
        self._data = self._data[self._data.columns[selector.get_support(indices=True)]]
        removed = [f for f in self._features.keys() if f not in self._data]
        self._features = {k: v for k, v in self._features.items() if k in self._data.columns}
        if len(removed) > 0:
            self.logger.debug("> features removed:\n- %s" % "\n- ".join(sorted(removed)))
            self._metadata['dataset']['dropped-features'] = removed
        if mi_select:
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            l.debug("> apply mutual information feature selection")
            if mi_kbest >= 1:
                k = int(mi_kbest)
            elif mi_kbest > 0 and mi_kbest < 1:
                k = int(len(self._data.columns) * mi_kbest)
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            selector.fit(self._data, self._target)
            selected_features = self._data.columns[selector.get_support(indices=True)]
            removed = [f for f in self._features.keys() if f not in selected_features]
            self._data = self._data[selected_features]
            self._features = {k: v for k, v in self._features.items() if k in selected_features}
            if len(removed) > 0:
                self.logger.debug("> features removed:\n- %s" % "\n- ".join(sorted(removed)))
                self._metadata['dataset']['dropped-features'] = removed
        class Dummy: pass
        self._train, self._test = Dummy(), Dummy()
        ds.logger.debug("> split data and target vectors to train and test subsets")
        if self.algorithm.labelling == "none":
            self._train.data, self._train.target = self._data, self._target
            self._test.data, self._test.target = pd.DataFrame(), pd.DataFrame()
        else:  # use a default split of 80% training and 20% testing
            from sklearn.model_selection import train_test_split
            tsize = kw.get('split_size', .2)
            self._train.data, self._test.data, self._train.target, self._test.target = \
                train_test_split(self._data, self._target, test_size=tsize, random_state=42, stratify=self._target)
        #FIXME: from there, self._train.data has columns of types float64 and bool ;
        #        for MBKMeans, it gives "TypeError: No matching signature found"
        #pd.to_numeric(s)
        return True
    
    def test(self, executable, ignore_labels=False, **kw):
        """ Test a single executable or a set of executables and evaluate metrics. """
        l, ds = self.logger, executable
        if len(self.pipeline.steps) == 0:
            l.warning("Model shall be trained before testing")
            return
        kw['data_only'], kw['dataset'] = True, ds
        if not self._prepare(**kw):
            return
        cls = self._algorithm = Algorithm.get(self._metadata['algorithm']['name'])
        l.debug(f"Testing {self.name} on {ds}...")
        prediction, dt = benchmark(self.pipeline.predict)(self._data)
        try:
            proba, dt2 = benchmark(self.pipeline.predict_proba)(self._data)
            dt += dt2
            proba = proba[:, 1]
        except AttributeError:
            proba = None
        metrics = cls.metrics if isinstance(cls.metrics, (list, tuple)) else [cls.metrics]
        render(Section("Test results for: " + ds))
        for metric in metrics:
            try:
                m, h = self._metrics(self._data, prediction, self._target, proba, metric, dt, ignore_labels)
            except TypeError:  # when None is returned because of a bad metrics category OR ignore_labels is
                continue       #  True and labels were required for the metrics category
            for header in [[], ["Model"]][self.__class__ is DumpedModel] + ["Dataset"] + h:
                if header not in self._performance.columns:
                    self._performance[header] = np.nan
            render(Table([m], column_headers=h, title=f"{metric.capitalize()} metrics" if len(metrics) > 0 else None))
            if len(self._data) > 0:
                row = {'Model': self.name} if self.__class__ is DumpedModel else {}
                row['Dataset'] = str(ds) + ["", "(unlabelled)"][ignore_labels]
                for k, v in zip(h, m):
                    row[k] = v
                self._performance = pd.concat([self._performance, pd.DataFrame.from_records([row])], ignore_index=True)
        if len(self._data) > 0:
            self._save()
    
    @property
    def classifier(self):
        if self.pipeline is not None:
            return self.pipeline.steps[-1][1]
    
    @property
    def name(self):
        if not hasattr(self, "_name"):
            self._name = None
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = config.check(value)
        self.path = Path(config['models'].joinpath(self._name)).absolute()
        self._load()


class Model(BaseModel):
    """ Folder structure:
    
    [name]
      +-- dump.joblib                           # dump of the model in joblib format
      +-- features.json                         # dictionary of feature name/description pairs
      +-- metadata.json                         # useful information about the model
      +-- performance.csv                       # performance testing data
    """
    STRUCTURE = ["dump.joblib", "features.json", "metadata.json", "performance.csv"]
    
    def _load(self):
        """ Load model's associated files if relevant or create instance's attributes. """
        self.__read_only = False
        if not Model.check(self.path):  # NB: self.path is a property computed based on self.name
            return
        self.logger.debug(f"loading model {self.path}...")
        for n in ["dump", "features", "metadata", "performance"]:
            p = self.path.joinpath(n + (".joblib" if n == "dump" else ".csv" if n == "performance" else ".json"))
            self.logger.debug(f"> loading {p.basename}...")
            if n == "dump":
                self.pipeline.pipeline = joblib.load(str(p))
                self.__read_only = True
            elif n == "performance":
                try:
                    self._performance = pd.read_csv(str(p), sep=";")
                except pd.errors.EmptyDataError:
                    pass  # self._performance was already created in __init__
            else:
                with p.open() as f:
                    setattr(self, "_" + n, json.load(f))
    
    def _save(self):
        """ Save model's state to the related files. """
        l = self.logger
        if not self.__read_only and self.path.exists():
            l.warning("This model already exists !")
            return
        self.path.mkdir(exist_ok=True)
        l.debug(f"{['Saving','Updating'][self.__read_only]} model {self.path}...")
        if not self.__read_only:
            for n in ["dump", "features", "metadata"]:
                p = self.path.joinpath(n + (".joblib" if n == "dump" else ".json"))
                l.debug(f"> saving {p}...")
                if n == "dump":
                    joblib.dump(self.pipeline.pipeline, str(p))
                else:
                    with p.open('w+') as f:
                        json.dump(getattr(self, "_" + n), f, indent=2)
                p.chmod(0o444)
            self.__read_only = True
        p = self.path.joinpath("performance.csv")
        l.debug(f"> saving {p}...")
        self._performance.to_csv(str(p), sep=";", index=False, header=True, float_format=FLOAT_FORMAT)
    
    def browse(self, executable=None, query=None, **kw):
        """ Browse the data from a dataset, including its predictions. """
        l, ds = self.logger, executable or self._metadata['dataset']['name']
        if len(self.pipeline.steps) == 0:
            l.warning("Model shall be trained before browsing a dataset")
            return
        kw['data_only'], kw['dataset'] = True, ds
        if not self._prepare(ds_type="Target", **kw):
            return
        l.debug(f"Applying predictions with {self.name} on {ds}...")
        pred = self.pipeline.predict(self._data)
        try:
            data = Dataset.load(ds)._data
        except ValueError:
            data = self._data
        for f in self._metadata['dataset']['dropped-features']:
            try:
                del data[f]
            except KeyError:
                continue
            l.debug(f"> dropped feature: {f}")
        # if the prediction relates to a cluster, name it accordingly
        k = 'cluster' if 'n_clusters' in self.algorithm.parameters or \
                      any('n_clusters' in d for d in self.algorithm.parameters.values()) else 'prediction'
        data[k] = pred
        with data_to_temp_file(filter_data(data, query, logger=self.logger), prefix="model-data-") as tmp:
            edit_file(tmp, logger=self.logger)
    
    def compare(self, dataset=None, model=None, include=False, **kw):
        """ Compare the last performance of this model on the given dataset with other ones. """
        l, data, models = self.logger, [], [self]
        l.debug("Comparing models...")
        if isinstance(dataset, list):
            dataset = [d.path.stem for d in dataset if not isinstance(d, str)]
        if isinstance(model, (list, tuple)):
            models.extend(model)
        elif model is not None:
            models.append(model)
        if include:
            class Dummy:
                def __str__(self): return ""
            m = Dummy()
            m.name, csv = "", config['models'].joinpath(".performances.csv")
            m._performance = pd.read_csv(csv, sep=";")
            models.append(m)
        # compute performance data
        perf = pd.DataFrame()
        for m in models:
            data = []
            for h in m._performance.columns:
                if h not in perf.columns:
                    perf.columns[h] = np.nan
            for _, row in m._performance.iterrows():
                row = row.to_dict()
                d = row['Dataset']
                if "Model" not in row:
                    tmp, row = row, {'Model': m.name}
                    row.update(tmp)
                if dataset is None or d == dataset or isinstance(dataset, (list, tuple)) and d in dataset:
                    data.append(row)
            perf = pd.concat([perf, pd.DataFrame.from_records(data)])
        if len(perf) == 0:
            l.warning("No model selected" if dataset is None else \
                      f"{[dataset[0],'Datasets'][len(dataset) > 1]} not found for the given model")
            return
        # display performance data
        h = list(perf.columns)
        data = sorted(perf.values.tolist(), key=lambda row: (row[0], row[1]))
        render(Table(highlight_best(data, h, [0, 1, -1]), column_headers=h))
    
    def edit(self, **kw):
        """ Edit the performance log file. """
        self.logger.debug("editing %sperformances.csv..." % ["model's ", "workspace's ."][self.path is None])
        edit_file(self.path or config['models'].joinpath(".performances.csv"), logger=self.logger)
    
    def preprocess(self, executable=None, query=None, **kw):
        """ Preprocess an input dataset given selected features and display it with visidata for review. """
        kw['data_only'], kw['dataset'] = True, executable or self._metadata['dataset']['name']
        if not self._prepare(**kw):
            return
        ds = Dataset.load(kw['dataset'])
        result = pd.DataFrame()
        for col in ["hash"] + EXE_METADATA:
            result[col] = ds._data[col]
        df = pd.DataFrame(self.pipeline.preprocess(self._data), columns=self._data.columns)
        for col in self._data.columns:
            result[col] = df[col]
            col2 = "*" + col
            result[col2] = ds._data[col]
        result['label'] = ds._data['label']
        with data_to_temp_file(filter_data(result, query, logger=self.logger), prefix="model-preproc-") as tmp:
            edit_file(tmp, logger=self.logger)
    
    def show(self, **kw):
        """ Show an overview of the model. """
        a, ds, ps = self._metadata['algorithm'], self._metadata['dataset'], self.pipeline.steps
        last_step = ps[-1][1] if isinstance(ps[-1], tuple) else ps[-1]
        fi, fi_str = getattr(last_step, "feature_importances_", None), []
        if fi is not None:
            # compute the ranked list of features with non-null importances (including the importance value)
            best_feat = [(i, n) for i, n in \
                         sorted(zip(fi, sorted(self._features.keys())), key=lambda x: -x[0]) if i > 0.]
            if len(best_feat) > 0:
                l, nf = max(map(len, [x[1] for x in best_feat])), len(str(len(best_feat)))
                best_feat = [("{: <%s}: {} (%.3f)" % (l + nf - len(str(i+1)), p[0])) \
                             .format(p[1], self._features[p[1]]) for i, p in enumerate(best_feat)]
                fi_str = [f"**Features**:      {len(self._features)} ({len(best_feat)} with non-null importance)"
                          f"\n\n\t- {'\n\n\t- '.join(best_feat)}\n\n"]
        else:
            from tinyscript import re, string
            feat, cnt = [], None
            for f in string.sorted_natural(self._features.keys()):
                try:
                    prefix, i, suffix = re.split(r"(\d+)", f, 1)
                    j = int(i)
                    if cnt is None:
                        cnt, prev_prefix, prev_suffix = (j, None), prefix, suffix
                        continue
                    elif prev_prefix == prefix and prev_suffix == suffix and \
                         (cnt[1] is None and j == cnt[0] + 1 or j == cnt[1] + 1):
                        cnt = (cnt[0], j)
                        continue
                except ValueError:
                    pass
                if cnt is not None:
                    if isinstance(cnt[0], int) and cnt[1] is None:
                        feat.append(f"{prev_prefix}{cnt[0]}{prev_suffix}")
                    else:
                        feat.append(f"{prev_prefix}[{cnt[0]}-{cnt[1]}]{prev_suffix}")
                    cnt = None
                feat.append(f)
            fi_str = [f"**Features**:      {len(self._features)}\n\n\t- {'\n\n\t- '.join(feat)}\n\n"]
        params = a['parameters'].keys()
        l = max(map(len, params))
        params = [("{: <%s} = {}" % l).format(*p) for p in sorted(a['parameters'].items(), key=lambda x: x[0])]
        c = List([f"**Path**:          {self.path}",
                  f"**Size**:          {human_readable_size(self.path.joinpath('dump.joblib').size)}",
                  f"**Algorithm**:     {a['description']} ({a['name']})",
                  f"**Multiclass**:    {'NY'[a['multiclass']]}"] + fi_str + \
                 ["**Preprocessors**: \n\n\t- %s\n\n" % "\n\t- ".join(a['preprocessors']),
                  "**Parameters**: \n\n\t- %s\n\n" % "\n\t- ".join(params)])
        render(Section("Model characteristics"), c)
        ds_path = config['datasets'].joinpath(ds['name'])
        c = List([f"**Path**:         {ds_path}",
                  f"**Size**:         {human_readable_size(ds_path.size)}",
                  f"**#Executables**: {ds['executables']}",
                  f"**Formats**:      {', '.join(ds['formats'])}",
                  f"**Packers**:      {', '.join(get_counts(ds).keys())}"])
        render(Section("Reference dataset"), c)
    
    def train(self, algorithm=None, cv=5, n_jobs=None, param=None, reset=False, ignore_labels=False,
              wrapper_select=False, select_param=None, **kw):
        """ Training method handling cross-validation. """
        l, ds, multiclass = self.logger, kw['dataset'], kw.get('multiclass', False)
        try:
            cls = self._algorithm = Algorithm.get(algorithm)
        except KeyError:
            l.error(f"{algorithm} not available")
            return
        kw['preprocessor'] = kw.get('preprocessor') or getattr(cls, "preprocessors", [])
        algo = cls.__class__.__name__
        self._metadata.setdefault('algorithm', {})
        self._metadata['algorithm']['name'] = algo
        self._metadata['algorithm']['description'] = cls.description
        self._metadata['algorithm']['multiclass'] = multiclass
        self._metadata['algorithm']['preprocessors'] = kw['preprocessor']
        # check that, if the algorithm is supervised, it has full labels
        if cls.labelling == "full" and ds.labelling < 1.:
            l.error(f"'algo' won't work with a dataset that is not fully labelled")
            return
        # check that, if the algorithm is semi-supervised, it is not labelled at all ; if so, stop here (should be
        #  unsupervised, that is, cls.labelling == "none")
        if cls.labelling == "partial" and ds.labelling == 0.:
            l.error(f"'{algo}' won't work with a dataset that is not labelled")
            return
        l.info(f"Selected algorithm: {cls.description}")
        # prepare the dataset first, as we need to know the number of features for setting model's name
        if not self._prepare(**kw):
            return
        if self.name is None:
            c = sorted(collapse_formats(*ds._metadata['formats']))
            self.name = f"{ds.path.stem}_{'-'.join(map(lambda x: x.lower(), c)).replace('.','')}_" \
                        f"{ds._metadata['executables']}_{algo.lower().replace('.','')}_f{len(self._features)}"
        if reset:
            self.path.remove(error=False)
        self._load()
        if self.__read_only:
            l.warning("Cannot retrain a model")
            l.warning(f"You can remove it first with the following command: model purge {self.name}")
            return
        if not getattr(cls, "multiclass", True) and multiclass:
            l.error(f"'{algo}' does not support multiclass")
            return
        # get classifer and parameters
        params = cls.parameters.get('static', cls.parameters if cls.labelling == "none" else None)
        if cls.is_weka():
            #params['model'] = self
            params['feature_names'] = sorted(self._features.keys())
        param_grid = {k: list(v) if isinstance(v, range) else v for k, v in cls.parameters.get('cv', {}).items()}
        if cls.labelling == "none" and len(param_grid) > 0:
            l.error(f"'{algo}' does not support grid search (while CV parameters are specified)")
            return
        # apply user-defined parameters
        for n, v in param.items():
            params[n] = v
            try:
                del param_grid[n]
            except KeyError:
                pass
        # set particular parameters ;
        # - n_clusters
        if params.get('n_clusters') == "auto":
            params['n_clusters'] = n = 2 if ds.labelling == .0 or not multiclass or ignore_labels else \
                                   len(set(l for l in self._metadata['dataset']['counts'].keys() if l != NOT_LABELLED))
            l.debug(f"> parameter n_clusters=\"auto\" set to {n}{[' based on labels',''][ignore_labels]}")
        # use recursive feature elimination with cross-validation to select optimal features
        n_jobs = n_jobs or config['number_jobs']
        if wrapper_select:
            l.info("Finding optimal feature set...")
            # apply user-defined parameters
            select_params = {'cv': cv, 'n_jobs': n_jobs}
            if select_param:
                for sp in select_param[0]:
                    n, v = sp.split('=')
                    select_params[str(n)] = int(v) if v.isdigit() else v
            from sklearn.feature_selection import RFECV
            feature_selector = RFECV(estimator=cls.base(**params), **select_params)
            feature_selector.fit(self._data, self._target.values.ravel())
            self._data = self._data[self._data.columns[feature_selector.get_support(indices=True)]]
            removed = [f for f in self._features.keys() if f not in self._data]
            if len(removed) > 0:
                self._metadata['dataset']['dropped-features'] = removed
                best_feat = [(i, n) for i, n in sorted(zip(feature_selector.ranking_, self._features.keys())) if i == 1]
                ll, nf = max(map(len, [x[1] for x in best_feat])), len(str(len(best_feat)))
                best_feat = [("{: <%s}: {}" % (ll + nf - len(str(i+1)))) \
                .format(p[1], self._features[p[1]]) for i, p in enumerate(best_feat)]
                fi_str = ["**Optimal features**:      %d (selected from %d features)\n\n\t1. %s\n\n" % \
                (len(best_feat), len(self._features), "\n\n\t1. ".join(best_feat))]
                render(List(fi_str))
                self._features = {k: v for k, v in self._features.items() if k in self._data.columns}
                self._train.data = self._train.data[self._data.columns]
                self._test.data = self._test.data[self._data.columns]
        l.info("Training model...")
        self.pipeline.append((cls.description, cls.base(**params)))
        # if a param_grid is input, perform cross-validation and select the best classifier
        _convert = lambda d: {k.split("__", 1)[1]: v for k, v in d.items()}
        if len(param_grid) > 0:
            from sklearn.model_selection import GridSearchCV
            l.debug(f"> applying Grid Search (CV={cv})...")
            Pipeline.silent = True
            # as we use a pipeline, we need to rename all parameters to [estimator name]__[parameter]
            param_grid = {f"{self.pipeline.steps[-1][0]}__{k}": v for k, v in param_grid.items()}
            grid = GridSearchCV(self.pipeline.pipeline, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=n_jobs)
            grid.fit(self._data, self._target.values.ravel())
            results = '\n'.join(f"  {m:0.3f} (+/-{s*2:0.03f}) for {_convert(p)}" \
                                for m, s, p in zip(grid.cv_results_['mean_test_score'],
                                                   grid.cv_results_['std_test_score'],
                                                   grid.cv_results_['params']))
            # as we use a pipeline, we need to reconvert parameters back to their normal name
            best_params = _convert(grid.best_params_)
            l.debug("> grid scores:\n{}".format(results))
            l.debug("> best parameters found:\n  {}".format(best_params))
            params.update(best_params)
            #if cls.is_weka():
            #    params['model'] = self
            self.pipeline.pop()
            self.pipeline.append((cls.description, cls.base(**params)))
            Pipeline.silent = False
        # now fit the (best) classifier and predict labels
        l.debug("> fitting the classifier...")
        self.pipeline.fit(self._train.data, self._train.target.values.ravel())
        Pipeline.silent = True
        l.debug("> making predictions...")
        predict = self.pipeline.predict if hasattr(self.pipeline.steps[-1][1], "predict") else self.pipeline.fit_predict
        self._train.predict = predict(self._train.data)
        try:
            self._train.predict_proba = self.pipeline.predict_proba(self._train.data)[:, 1]
        except (AttributeError, KeyError):  # some algorithms do not support .predict_proba(...)
            self._train.predict_proba = None
        if len(self._test.data) > 0:
            self._test.predict = predict(self._test.data)
            try:
                self._test.predict_proba = self.pipeline.predict_proba(self._test.data)[:, 1]
            except (AttributeError, KeyError):  # some algorithms do not support .predict_proba(...)
                self._test.predict_proba = None
        metrics = cls.metrics if isinstance(cls.metrics, (list, tuple)) else [cls.metrics]
        render(Section(f"Name: {self.name}"))
        print("\n")
        for metric in metrics:
            d, h = [], []
            for dset in ["train", "test"]:
                s = getattr(self, "_" + dset)
                if len(s.data) > 0:
                    try:
                        m, h = self._metrics(s.data, s.predict, s.target, s.predict_proba, metric,
                                             ignore_labels=ignore_labels)
                    except TypeError:  # when None is returned because of a bad metrics category OR ignore_labels is
                        continue       #  True and labels were required for the metrics category
                    d.append([dset.capitalize()] + m)
                    h = ["."] + h
            if len(h) > 0:
                t = f"{metric.capitalize()} metrics" if len(metrics) > 0 else None
                render(Table(d, column_headers=h, title=t))
        l.info("Parameters:\n- %s" % "\n- ".join(f"{p} = {v}" for p, v in params.items()))
        self._metadata['algorithm']['parameters'] = params
        self._save()
    
    def visualize(self, export=False, output_dir=".", **kw):
        """ Plot the model for visualization. """
        if len(self.pipeline.steps) == 0:
            self.logger.warning("Model shall be trained before visualizing")
            return
        from .visualization import _VISUALIZATIONS
        a = self._metadata['algorithm']
        viz_dict = _VISUALIZATIONS.get(a['name'], {})
        viz_func = viz_dict.get(["text", "image"][export])
        if viz_func is None:
            self.logger.warning(f"Visualization not available for this algorithm{[' in text mode',''][export]}")
            return
        params = {'algo_name' : a['name'], 'algo_params': a['parameters'],
                  'feature_names': sorted(self._features.keys()), 'logger': self.logger}
        params.update(kw.pop('viz_params', {}))
        # if visualization requires the original data (e.g. kNN), use self._prepare to create self._data/self._target
        if viz_dict.get("data", False):
            kw['data_only'] = True
            if not self._prepare(**kw):
                return
            params['data'], params['format'] = self._data, self._dataset._data['format']
            params['labels'] =self._dataset._data['label']
            params['dataset_name'] = self._dataset.name
            params['extension'] = self._dataset._data['realpath'].apply(lambda p: Path(p).extension)
            if viz_dict.get("target", True):
                params['target'] = self._target
        c = self.classifier
        c.model = self
        if export:
            viz_func(c, **params)
        else:
            print(viz_func(c, **params))
    
    @property
    def algorithm(self):
        return getattr(self, "_algorithm", None) or Algorithm.get(self._metadata['algorithm']['name'])
    
    @property
    def dataset(self):
        return getattr(self, "_dataset", None) or Dataset.load(self._metadata['dataset']['path'])
    
    @classmethod
    def list(cls, algorithms=False, **kw):
        """ List all the models from the given path or all available algorithms. """
        if algorithms:
            d = [(a.name, a.description) for a in Algorithm.registry]
            r = [Section(f"Algorithms ({len(d)})"), Table(d, column_headers=["Name", "Description"])]
        else:
            d = []
            for model in cls.iteritems(False):
                with model.joinpath("metadata.json").open() as meta:
                    metadata = json.load(meta)
                alg, ds = metadata['algorithm'], metadata['dataset']
                d.append([
                    model.stem,
                    alg['name'].upper(),
                    alg['description'],
                    ds['name'],
                    str(ds['executables']),
                    ",".join(sorted(ds['formats'])),
                    shorten_str(",".join(f"{n}{{{c}}}" for n, c in sorted(get_counts(ds).items(),
                                                                          key=lambda x: (-x[1], x[0])))),
                ])
            if len(d) == 0:
                cls.logger.warning(f"No model found in the workspace ({config['models']})")
                return
            r = [Section(f"Models ({len(d)})"),
                 Table(d, column_headers=["Name", "Algorithm", "Description", "Dataset", "Size", "Formats", "Packers"])]
        render(*r)


class DumpedModel(BaseModel):
    STRUCTURE = "*.joblib"
    
    def _load(self):
        from sklearn.preprocessing import MinMaxScaler
        obj = joblib.load(str(self.path))
        cls = obj.__class__.__name__
        if "Pipeline" not in cls:
            self.pipeline.append(("MinMaxScaler", default_scaler or MinMaxScaler)())
            self.pipeline.append((cls, obj))
        else:
            for s in obj.steps:
                self.pipeline.append(s)
        self.__p = config['models'].joinpath(".performances.csv")
        try:
            self._performance = pd.read_csv(str(self.__p), sep=";")
        except FileNotFoundError:
            pass
    
    def _save(self):
        self.logger.debug(f"> saving metrics to {self.__p}...")
        p = self._performance
        k = p.columns
        p = p.loc[p.round(3).drop_duplicates(subset=k[:-1]).index]
        p.to_csv(str(self.__p), sep=";", columns=k, index=False, header=True, float_format=FLOAT_FORMAT)

