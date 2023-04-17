# -*- coding: UTF-8 -*-
import joblib
import multiprocessing as mp
from _pickle import UnpicklingError
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tinyscript import ast, json, itertools, logging, subprocess
from tinyscript.helpers import human_readable_size, is_generator, Path, TempPath
from tinyscript.report import *

from .algorithm import Algorithm, WekaClassifier
from .dataset import *
from .executable import Executable
from .features import Features
from .metrics import *
from .pipeline import *
from .visualization import *
from ..common.config import *
from ..common.rendering import *
from ..common.utils import *


__all__ = ["open_model", "DumpedModel", "Model", "N_JOBS"]


FLOAT_FORMAT = "%.6f"
N_JOBS = mp.cpu_count() // 2


def open_model(item):
    """ Open the target model with the right class. """
    p = config['models'].joinpath(item)
    if Model.check(item):
        return Model(item)
    try:
        return DumpedModel(item)
    except FileNotFoundError:
        raise ValueError("%s does not exist" % item)
    except UnpicklingError:
        raise ValueError("%s is not a valid model" % item)


class Model:
    """ Folder structure:
    
    [name]
      +-- dump.joblib                           # dump of the model in joblib format
      +-- features.json                         # dictionary of feature name/description pairs
      +-- metadata.json                         # useful information about the model
      +-- performance.csv                       # performance testing data
    """
    @logging.bindLogger
    def __init__(self, name=None, load=True, **kw):
        self.__read_only = False
        self._features, self._metadata = {}, {}
        self._performance = pd.DataFrame()
        self.name = name.stem if isinstance(name, Path) else name
        self.pipeline = DebugPipeline()
        if load:
            self._load()
    
    def _load(self):
        """ Load model's associated files if relevant or create instance's attributes. """
        #self.path = config['models'].joinpath(self.name)
        if not Model.check(self.path):
            return
        self.logger.debug("loading model %s..." % self.path)
        for n in ["dump", "features", "metadata", "performance"]:
            p = self.path.joinpath(n + (".joblib" if n == "dump" else ".csv" if n == "performance" else ".json"))
            self.logger.debug("> loading %s..." % p.basename)
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
    
    def _metrics(self, data, prediction, target=None, proba=None, metrics="classification", proctime=None,
                 ignore_labels=False):
        """ Metrics computation method. """
        l, mfunc = self.logger, "%s_metrics" % metrics
        if mfunc not in globals():
            l.error("Bad metrics type '%s'" % metrics)
            return
        m = globals()[mfunc]
        l.debug("Computing %s metrics..." % metrics)
        df2np = lambda df: df.to_numpy() if isinstance(df, pd.DataFrame) else df
        fix = lambda v: df2np(v) if getattr(df2np(v), "ndim", 1) == 1 else df2np(v).transpose()[0]
        values, headers = m(data, fix(prediction), y_true=fix(target), y_proba=fix(proba), proctime=proctime,
                            ignore_labels=ignore_labels, logger=self.logger)
        return [f(v) for v, f in zip(values, headers.values())], list(headers.keys())
    
    def _prepare(self, dataset=None, preprocessor=None, multiclass=False, labels=None, feature=None, data_only=False,
                 unlabelled=False, **kw):
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
        ds, l, labels = dataset, self.logger, labels or {}
        # if not only the data shall be prepared, then the only supported input format is a valid dataset ;
        #  i.e. when preparing data for the train() method
        if not data_only:
            if not getattr(ds, "is_valid", lambda: False)():
                l.error("%s is not a valid input dataset" % dataset)
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
                    ds = open_dataset(ds)
                except ValueError:
                    pass
            if not ds.exists():
                l.error("Bad input dataset (%s)" % ds)
                return False
        self._dataset = ds
        l.info("%s dataset:  %s" % (kw.get('ds_type', ["Reference", "Test"][data_only]), ds))
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
                    self._data = self._data.append(exe.data, ignore_index=True)
                    if label:
                        self._target = self._target.append({'label': labels.get(exe.hash)}, ignore_index=True)
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
            missing_cols = ast.literal_eval(e.args[0].replace("not in index", ""))
            for col in missing_cols:
                self._features[col] = np.nan
            self._data = self._data.reindex(columns=sorted(self._features.keys()))
        self._target = self._target.replace(LABELS_BACK_CONV)
        if unlabelled:
            self._target['label'] = NOT_LABELLED
        self._data, self._target = self._data.fillna(-1), self._target.fillna(NOT_LABELLED)
        if not multiclass:  # convert to binary class
            self._target.loc[self._target.label == "", "label"] = 0
            self._target.loc[self._target.label != 0, "label"] = 1
            self._target = self._target.astype('int')
        # create the pipeline if it does not exist (i.e. while training)
        if not data_only:
            self.pipeline = DebugPipeline()
            l.info("Making pipeline...")
            make_pipeline(self.pipeline, preprocessor, self.logger)
        # if only data is to be processed (i.e. while testing), stop here, the rest is for training the model
        else:
            return True
        # apply variance threshold of 0.0 to remove useless features and rectify the list of features
        l.debug("> remove 0-variance features")
        selector = VarianceThreshold()
        selector.fit(self._data)
        self._data = self._data[self._data.columns[selector.get_support(indices=True)]]
        removed = [f for f in self._features.keys() if f not in self._data]
        self._features = {k: v for k, v in self._features.items() if k in self._data.columns}
        if len(removed) > 0:
            self.logger.debug("> features removed:\n- %s" % "\n- ".join(sorted(removed)))
            self._metadata['dataset']['dropped-features'] = removed
        # prepare for training and testing sets
        class Dummy: pass
        self._train, self._test = Dummy(), Dummy()
        ds.logger.debug("> split data and target vectors to train and test subsets")
        if self.algorithm.labelling == "none":
            self._train.data, self._train.target = self._data, self._target
            self._test.data, self._test.target = np.array([]), np.array([])
        else:  # use a default split of 80% training and 20% testing
            tsize = kw.get('split_size', .2)
            self._train.data, self._test.data, self._train.target, self._test.target = \
                train_test_split(self._data, self._target, test_size=tsize, random_state=42, stratify=self._target)
        return True
    
    def _save(self):
        """ Save model's state to the related files. """
        l = self.logger
        if not self.__read_only and self.path.exists():
            l.warning("This model already exists !")
            return
        self.path.mkdir(exist_ok=True)
        l.debug("%s model %s..." % (["Saving", "Updating"][self.__read_only], str(self.path)))
        if not self.__read_only:
            for n in ["dump", "features", "metadata"]:
                p = self.path.joinpath(n + (".joblib" if n == "dump" else ".json"))
                l.debug("> saving %s..." % str(p))
                if n == "dump":
                    joblib.dump(self.pipeline.pipeline, str(p))
                else:
                    with p.open('w+') as f:
                        json.dump(getattr(self, "_" + n), f, indent=2)
                p.chmod(0o444)
            self.__read_only = True
        p = self.path.joinpath("performance.csv")
        l.debug("> saving %s..." % str(p))
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
        l.debug("Applying predictions with %s on %s..." % (self.name, ds))
        pred = self.pipeline.predict(self._data)
        try:
            data = open_dataset(ds)._data
        except ValueError:
            data = self._data
        for f in self._metadata['dataset']['dropped-features']:
            try:
                del data[f]
            except KeyError:
                continue
            l.debug("> dropped feature: %s" % f)
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
            l.warning("No model selected" if dataset is None else "%s not found for the given model" % \
                      [dataset[0], "Datasets"][len(dataset) > 1])
            return
        # display performance data
        h = list(perf.columns)
        data = sorted(perf.values.tolist(), key=lambda row: (row[0], row[1]))
        render(Table(highlight_best(data, h, [0, 1, -1]), column_headers=h))
    
    def edit(self, **kw):
        """ Edit the performance log file. """
        self.logger.debug("editing %sperformances.csv..." % ["model's ", "workspace's ."][self.path is None])
        edit_file(self.path or config['models'].joinpath(".performances.csv"), logger=self.logger)
    
    def list(self, algorithms=False, **kw):
        """ List all the models from the given path or all available algorithms. """
        if algorithms:
            d = [(a.name, a.description) for a in Algorithm.registry]
            r = [Section("Algorithms (%d)" % len(d)), Table(d, column_headers=["Name", "Description"])]
        else:
            d = []
            for model in self.iteritems():
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
                    shorten_str(",".join("%s{%d}" % i for i in sorted(get_counts(ds).items(),
                                                                      key=lambda x: (-x[1], x[0])))),
                ])
            if len(d) == 0:
                self.logger.warning("No model found in the workspace (%s)" % config['models'])
                return
            r = [Section("Models (%d)" % len(d)),
                 Table(d, column_headers=["Name", "Algorithm", "Description", "Dataset", "Size", "Formats",
                                          "Packers"])]
        render(*r)
    
    def preprocess(self, executable=None, query=None, **kw):
        """ Preprocess an input dataset given selected features and display it with visidata for review. """
        kw['data_only'], kw['dataset'] = True, executable or self._metadata['dataset']['name']
        if not self._prepare(**kw):
            return
        ds = open_dataset(kw['dataset'])
        result = pd.DataFrame()
        for col in ["hash"] + Executable.FIELDS:
            result[col] = ds._data[col]
        df = pd.DataFrame(self.pipeline.preprocess(self._data), columns=self._data.columns)
        for col in self._data.columns:
            result[col] = df[col]
            col2 = "*" + col
            result[col2] = ds._data[col]
        result['label'] = ds._data['label']
        with data_to_temp_file(filter_data(result, query, logger=self.logger), prefix="model-preproc-") as tmp:
            edit_file(tmp, logger=self.logger)
    
    def purge(self, **kw):
        """ Purge the current model. """
        self.logger.debug("purging model...")
        self.path.remove(error=False)
    
    def rename(self, name2=None, **kw):
        """ Rename the current model. """
        self.logger.debug("renaming model...")
        self.path = p = config['models'].joinpath(name2)
        if p.exists():
            self.logger.warning("%s already exists" % p)
    
    def show(self, **kw):
        """ Show an overview of the model. """
        a, ds, ps = self._metadata['algorithm'], self._metadata['dataset'], self.pipeline.steps
        last_step = ps[-1][1] if isinstance(ps[-1], tuple) else ps[-1]
        fi, fi_str = getattr(last_step, "feature_importances_", None), []
        if fi is not None:
            # compute the ranked list of features with non-null importances (including the importance value)
            best_feat = [(i, n) for i, n in \
                         sorted(zip(fi, sorted(self._features.keys())), key=lambda x: -x[0]) if i > 0.]
            l, nf = max(map(len, [x[1] for x in best_feat])), len(str(len(best_feat)))
            best_feat = [("{: <%s}: {} (%.3f)" % (l + nf - len(str(i+1)), p[0])) \
                         .format(p[1], self._features[p[1]]) for i, p in enumerate(best_feat)]
            fi_str = ["**Features**:      %d (%d with non-null importance)\n\n\t1. %s\n\n" % \
                      (len(self._features), len(best_feat), "\n\n\t1. ".join(best_feat))]
        params = a['parameters'].keys()
        l = max(map(len, params))
        params = [("{: <%s} = {}" % l).format(*p) for p in sorted(a['parameters'].items(), key=lambda x: x[0])]
        c = List(["**Path**:          %s" % self.path,
                  "**Size**:          %s" % human_readable_size(self.path.joinpath("dump.joblib").size),
                  "**Algorithm**:     %s (%s)" % (a['description'], a['name']),
                  "**Multiclass**:    %s" % "NY"[a['multiclass']]] + fi_str + \
                 ["**Preprocessors**: \n\n\t- %s\n\n" % "\n\t- ".join(a['preprocessors']),
                  "**Parameters**: \n\n\t- %s\n\n" % "\n\t- ".join(params)])
        render(Section("Model characteristics"), c)
        ds_path = config['datasets'].joinpath(ds['name'])
        c = List(["**Path**:         %s" % ds_path,
                  "**Size**:         %s" % human_readable_size(ds_path.size),
                  "**#Executables**: %d" % ds['executables'],
                  "**Formats**:      %s" % ", ".join(ds['formats']),
                  "**Packers**:      %s" % ", ".join(get_counts(ds).keys())])
        render(Section("Reference dataset"), c)
    
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
        l.debug("Testing %s on %s..." % (self.name, ds))
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
            render(Table([m], column_headers=h, title="%s metrics" % metric.capitalize() if len(metrics) > 0 else None))
            if len(self._data) > 0:
                row = {'Model': self.name} if self.__class__ is DumpedModel else {}
                row['Dataset'] = str(ds) + ["", "(unlabelled)"][ignore_labels]
                for k, v in zip(h, m):
                    row[k] = v
                self._performance = self._performance.append(row, ignore_index=True)
        if len(self._data) > 0:
            self._save()
    
    def train(self, algorithm=None, cv=5, n_jobs=N_JOBS, param=None, reset=False, ignore_labels=False, **kw):
        """ Training method handling cross-validation. """
        l, n_cpu, ds, multiclass = self.logger, mp.cpu_count(), kw['dataset'], kw.get('multiclass', False)
        try:
            cls = self._algorithm = Algorithm.get(algorithm)
        except KeyError:
            l.error("%s not available" % algorithm)
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
            l.error("'%s' won't work with a dataset that is not fully labelled" % algo)
            return
        # check that, if the algorithm is semi-supervised, it is not labelled at all ; if so, stop here (should be
        #  unsupervised, that is, cls.labelling == "none")
        if cls.labelling == "partial" and ds.labelling == 0.:
            l.error("'%s' won't work with a dataset that is not labelled" % algo)
            return
        l.info("Selected algorithm: %s" % cls.description)
        if n_jobs > n_cpu:
            l.warning("Maximum n_jobs is %d" % n_cpu)
            n_jobs = n_cpu
        # prepare the dataset first, as we need to know the number of features for setting model's name
        if not self._prepare(**kw):
            return
        if self.name is None:
            c = sorted(collapse_formats(*ds._metadata['formats']))
            self.name = "%s_%s_%d_%s_f%d" % (ds.path.stem, "-".join(map(lambda x: x.lower(), c)).replace(".", ""),
                                             ds._metadata['executables'],
                                             algo.lower().replace(".", ""), len(self._features))
        if reset:
            self.path.remove(error=False)
        self._load()
        if self.__read_only:
            l.warning("Cannot retrain a model")
            l.warning("You can remove it first with the following command: model purge %s" % self.name)
            return
        if not getattr(cls, "multiclass", True) and multiclass:
            l.error("'%s' does not support multiclass" % algo)
            return
        # get classifer and parameters
        params = cls.parameters.get('static', cls.parameters if cls.labelling == "none" else None)
        if isinstance(cls, WekaClassifier):
            params['model'] = self
        param_grid = {k: list(v) if isinstance(v, range) else v for k, v in cls.parameters.get('cv', {}).items()}
        if cls.labelling == "none" and len(param_grid) > 0:
            l.error("'%s' does not support grid search (while CV parameters are specified)" % algo)
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
            l.debug("> parameter n_clusters=\"auto\" set to %d%s" % (n, [" based on labels", ""][ignore_labels]))
        l.info("Training model...")
        self.pipeline.append((cls.name, cls.base(**params)))
        # if a param_grid is input, perform cross-validation and select the best classifier
        _convert = lambda d: {k.split("__", 1)[1]: v for k, v in d.items()}
        if len(param_grid) > 0:
            l.debug("> applying Grid Search (CV=%d)..." % cv)
            Pipeline.silent = True
            # as we use a pipeline, we need to rename all parameters to [estimator name]__[parameter]
            param_grid = {"%s__%s" % (self.pipeline.steps[-1][0], k): v for k, v in param_grid.items()}
            grid = GridSearchCV(self.pipeline.pipeline, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=n_jobs)
            grid.fit(self._data, self._target.values.ravel())
            results = '\n'.join("  %0.3f (+/-%0.03f) for %r" % (m, s * 2, _convert(p)) \
                                for m, s, p in zip(grid.cv_results_['mean_test_score'],
                                                   grid.cv_results_['std_test_score'],
                                                   grid.cv_results_['params']))
            # as we use a pipeline, we need to reconvert parameters back to their normal name
            best_params = _convert(grid.best_params_)
            l.debug("> grid scores:\n{}".format(results))
            l.debug("> best parameters found:\n  {}".format(best_params))
            params.update(best_params)
            if isinstance(cls, WekaClassifier):
                params['model'] = self
            self.pipeline.pop()
            self.pipeline.append((cls.name, cls.base(**params)))
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
        except AttributeError:  # some algorithms do not support .predict_proba(...)
            self._train.predict_proba = None
        if len(self._test.data) > 0:
            self._test.predict = predict(self._test.data)
            try:
                self._test.predict_proba = self.pipeline.predict_proba(self._test.data)[:, 1]
            except AttributeError:  # some algorithms do not support .predict_proba(...)
                self._test.predict_proba = None
        metrics = cls.metrics if isinstance(cls.metrics, (list, tuple)) else [cls.metrics]
        render(Section("Name: %s" % self.name))
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
                t = "%s metrics" % metric.capitalize() if len(metrics) > 0 else None
                render(Table(d, column_headers=h, title=t))
        l.info("Parameters:\n- %s" % "\n- ".join("%s = %s" % p for p in params.items()))
        self._metadata['algorithm']['parameters'] = params
        self._save()
    
    def visualize(self, export=False, output_dir=".", **kw):
        """ Plot the model for visualization. """
        if len(self.pipeline.steps) == 0:
            self.logger.warning("Model shall be trained before visualizing")
            return
        a = self._metadata['algorithm']
        viz_dict = VISUALIZATIONS.get(a['name'], {})
        viz_func = viz_dict.get(["text", "image"][export])
        if viz_func is None:
            self.logger.warning("Visualization not available for this algorithm%s" % [" in text mode", ""][export])
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
        if export:
            fig = viz_func(self.classifier, **params)
            dst = str(Path(output_dir).joinpath("%s%s.png" % (self.name, getattr(fig, "dst_suffix", ""))))
            fig.savefig(dst, format="png", bbox_inches="tight")
            self.logger.warning("Visualization saved to %s" % dst)
        else:
            print(viz_func(self.classifier, **params))
    
    @property
    def algorithm(self):
        return getattr(self, "_algorithm", None) or Algorithm.get(self._metadata['algorithm']['name'])
    
    @property
    def classifier(self):
        if self.pipeline is not None:
            return self.pipeline.steps[-1][1]
    
    @property
    def dataset(self):
        return getattr(self, "_dataset", None) or open_dataset(self._metadata['dataset']['path'])
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = check_name(value)
    
    @property
    def path(self):
        if self.name:
            if not hasattr(self, "_path"):
                self._path = Path(config['models'].joinpath(self.name)).absolute()
            return self._path
    
    @path.setter
    def path(self, value):
        if not isinstance(value, Path):
            value = Path(value).absolute()
        self._path = value
    
    @staticmethod
    def check(folder):
        try:
            Model.validate(folder)
            return True
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def count():
        return sum(1 for _ in Path(config['models']).listdir(Model.check))
    
    @staticmethod
    def iteritems(instantiate=False):
        for model in Path(config['models']).listdir(Model.check):
            yield Model(model) if instantiate else model
    
    @staticmethod
    def validate(folder):
        f = config['models'].joinpath(folder)
        if not f.exists():
            raise ValueError("Folder does not exist")
        if not f.is_dir():
            raise ValueError("Input is not a folder")
        for fn in ["dump.joblib", "features.json", "metadata.json", "performance.csv"]:
            if not f.joinpath(fn).exists():
                raise ValueError("Folder does not have %s" % fn)
        return f


class DumpedModel:
    @logging.bindLogger
    def __init__(self, name=None, is_pipeline=False, default_scaler=StandardScaler, **kw):
        global logger
        logger = self.logger
        obj, self.pipeline = joblib.load(str(name)), DebugPipeline()
        if not is_pipeline:
            self.pipeline.append(default_scaler())
            self.pipeline.append(obj)
        else:
            for s in obj.steps:
                self.pipeline.append(s)
        self.name      = Path(name).stem
        self._features = {}
        self._metadata = {}
        self.__p = config['models'].joinpath(".performances.csv")
        try:
            self._performance = pd.read_csv(str(self.__p), sep=";")
        except FileNotFoundError:
            self._performance = pd.DataFrame()
    
    def _save(self):
        self.logger.debug("> saving metrics to %s..." % str(self.__p))
        p = self._performance
        k = p.columns
        p = p.loc[p.round(3).drop_duplicates(subset=k[:-1]).index]
        p.to_csv(str(self.__p), sep=";", columns=k, index=False, header=True, float_format=FLOAT_FORMAT)
    
    _metrics    = Model._metrics
    _prepare    = Model._prepare
    classifier  = Model.classifier
    test        = Model.test

