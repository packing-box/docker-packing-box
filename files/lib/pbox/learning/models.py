# -*- coding: UTF-8 -*-
import joblib
import multiprocessing as mp
import pandas as pd
from sklearn.base import is_classifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import *
from tinyscript import ast, json, logging, subprocess
from tinyscript.helpers import human_readable_size, is_generator, Path
from tinyscript.report import *

from .algorithms import Algorithm, WekaClassifier
from .dataset import *
from .executable import Executable
from ..common.config import config
from ..common.utils import *


__all__ = ["DumpedModel", "Model", "PREPROCESSORS"]


FLOAT_FORMAT = "%.6f"
N_JOBS = mp.cpu_count() // 2
PERF_HEADERS = {
    'Dataset':         lambda x: x,
    'Accuracy':        lambda x: "%.2f%%" % (x * 100),
    'Precision':       lambda x: "%.2f%%" % (x * 100),
    'Recall':          lambda x: "%.2f%%" % (x * 100),
    'F-Measure':       lambda x: "%.2f%%" % (x * 100),
    'MCC':             lambda x: "%.2f%%" % (x * 100),
    'AUC':             lambda x: "%.2f%%" % (x * 100),
    'Processing Time': lambda x: "%.3fms" % (x * 1000),
}
PREPROCESSORS = {
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
    'Std':        StandardScaler,
}


def _shorten(string, l=80):
    i = 0
    if len(string) <= l:
        return string
    s = ",".join(string.split(",")[:-1])
    if len(s) == 0:
        return string[:l-3] + "..."
    while 1:
        t = s.split(",")
        if len(t) > 1:
            s = ",".join(t[:-1])
            if len(s) < l-3:
                return s + "..."
        else:
            return s[:l-3] + "..."
    return s + "..."


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
        self.name = name.stem if isinstance(name, Path) else name
        self.classifier = None
        if load:
            self._load()
    
    def _load(self):
        """ Load model's associated files if relevant or create instance's attributes. """
        if self.name is None:
            self._features = {}
            self._metadata = {}
            self._performance = pd.DataFrame(columns=PERF_HEADERS.keys())
        else:
            self.path = Path(config['models'].joinpath(self.name)).absolute()
            if not Model.check(self.path):
                return
            for n in ["dump", "features", "metadata", "performance"]:
                p = self.path.joinpath(n + (".joblib" if n == "dump" else ".csv" if n == "performance" else ".json"))
                self.logger.debug("Loading %s..." % str(p))
                if n == "dump":
                    self.classifier = joblib.load(str(p))
                    self.__read_only = True
                elif n == "performance":
                    self._performance = pd.read_csv(str(p), sep=";")
                else:
                    with p.open() as f:
                        setattr(self, "_" + n, json.load(f))
    
    def _metrics(self, target, prediction, proba):
        """ Metrics computation method. """
        self.logger.debug("> Computing metrics...")
        # compute indicators
        tn, fp, fn, tp = confusion_matrix(target, prediction).ravel()
        # compute evaluation metrics:
        accuracy  = float(tp + tn) / (tp + tn + fp + fn)
        precision = float(tp) / (tp + fp)
        recall    = float(tp) / (tp + fn)  # or also sensitivity
        f_measure = float(2 * precision * recall) / (precision + recall)
        mcc       = matthews_corrcoef(target, prediction)
        try:
            auc = roc_auc_score(target.label, proba)
        except ValueError:
            auc = -1
        # return metrics for further display
        return [accuracy, precision, recall, f_measure, mcc, auc]
    
    def _prepare(self, dataset, preprocessor=None, multiclass=False, feature=None, pattern=None, data_only=False, **kw):
        """ Prepare the Model instance based on the given Dataset instance. """
        ds, l, labels = dataset, self.logger, kw.get('labels', {})
        if not data_only:
            if not getattr(ds, "is_valid", lambda: False)():
                l.error("Not a valid input dataset")
                return False
            # copy relevant information from the input dataset
            l.debug("Preparing the dataset...")
            self._metadata['dataset'] = {k: v for k, v in ds._metadata.items()}
            self._metadata['dataset']['path'] = str(ds.path)
            self._metadata['dataset']['name'] = ds.path.stem
        self._data, self._target = pd.DataFrame(), pd.DataFrame(columns=["label"])
        # case 1: CSV file
        try:
            d, n = pd.read_csv(str(ds), sep=kw.pop('sep', ",")), Path(ds).filename
            self._data, self._target = d.loc[:, d.columns != "label"], d.loc[:, d.columns == "label"]
            self._features = {k: "" for k in self._data.columns}
        except AttributeError:  # 'DataFrame' object has no attribute 'label'
            l.error(d.columns)
            l.warning("This error may be caused by a bad CSV separator ; you can set it with --sep")
            raise
        except (KeyError, FileNotFoundError, TypeError):
            def __parse(exes):
                if not isinstance(exes, list) and not is_generator(exes):
                    exes = [exes]
                for exe in exes:
                    if not isinstance(exe, Executable):
                        exe = Executable(str(exe))
                    exe.selection = feature or pattern
                    self._features.update(exe.features)
                    self._data.append(exe.data, ignore_index=True)
                    self._target.append(labels.get(exe.hash, exe.label))
            # handle individual executables, folders of executables or both types of dataset instances
            if config['datasets'].joinpath(str(ds)).exists():
                ds = open_dataset(ds)
            # case 2: handle a fileless dataset (where features are already computed)
            if isinstance(ds, FilelessDataset):
                for exe in ds:
                    break
                exe = Executable(exe)
                exe.selection = feature or pattern
                self._features.update(exe.features)
                self._data = ds._data[list(exe.features.keys())]
                self._target = [x for x in ds._data.label.values]
            # case 3: handle a normal dataset (features shall still be computed)
            elif isinstance(ds, Dataset):
                __parse(ds.files.listdir(is_executable))
            # case 4: handle a single executable
            elif is_executable(str(ds)):
                __parse(ds)
            # case 5: handle the executables within a folder
            elif ds.is_folder():
                __parse(ds.listdir(is_executable))
            self._target = pd.DataFrame(self._target, columns=["label"])
        if len(self._data) == 0:
            l.warning("No data")
            return
        if len(self._features) == 0:
            l.warning("No selectable feature ; this may be due to a model unrelated to the input")
            raise ValueError("No feature to extract")
        n = getattr(ds, "name", str(ds))
        if not multiclass:  # convert to binary class
            self._target = self._target.fillna(0)
            self._target.loc[self._target.label != 0, "label"] = 1
        else:
            self._target = self._target.fillna("")
        self.labels = set(self._target.label.values)
        if multiclass is None:
            multiclass = len(self.labels) > 2
        self._taget = self._target.values.ravel()
        self._preprocess(*(preprocessor or ()))
        if data_only:
            return True
        # prepare for training and testing sets
        class Dummy: pass
        self._train, self._test = Dummy(), Dummy()
        ds.logger.debug("> Split data and target vectors to train and test subnets")
        self._train.data, self._test.data, self._train.target, self._test.target = \
            train_test_split(self._data, self._target, test_size=.2, random_state=42)
        # prepare for Weka
        l.debug("> Create ARFF train and test files (for Weka)")
        WekaClassifier.to_arff(self)
        return True
    
    def _preprocess(self, *preprocessors):
        """ Preprocess the bound _data with an input list of preprocessors. """
        c = False
        for p in preprocessors:
            p, params = PREPROCESSORS[p], {}
            if isinstance(p, tuple):
                p, params = p
                m = "%s with %s" % (p.__name__, ", ".join("{}={}".format(*i) for i in params.items()))
            else:
                m = p.__name__
            n = p.__name__
            v = "Transform" if n.endswith("Transformer") else "Encode" if n.endswith("Encoder") else \
                "Standardize" if n.endswith("Scaler") else "Normalize" if n.endswith("Normalizer") else \
                "Discretize" if n.endswith("Discretizer") else "Preprocess"
            self.logger.debug("> %s the data (%s)" % (v, m))
            preprocessed = p(**params).fit_transform(self._data)
            c = True
        if c:
            self._data = pd.DataFrame(preprocessed, index=self._data.index, columns=self._data.columns)
    
    def _save(self):
        """ Save model's state to the related files. """
        l = self.logger
        self.path = Path(config['models'].joinpath(self.name)).absolute()
        self.path.mkdir(exist_ok=True)
        l.debug("%s model %s..." % (["Saving", "Updating"][self.__read_only], str(self.path)))
        p = self.path.joinpath("performance.csv")
        l.debug("> %s" % str(p))
        self._performance.to_csv(str(p), sep=";", columns=PERF_HEADERS.keys(), index=False, header=True,
                                 float_format=FLOAT_FORMAT)
        if not self.__read_only:
            for n in ["dump", "features", "metadata"]:
                p = self.path.joinpath(n + (".joblib" if n == "dump" else ".json"))
                l.debug("> %s" % str(p))
                if n == "dump":
                    joblib.dump(self.classifier, str(p))
                else:
                    with p.open('w+') as f:
                        json.dump(getattr(self, "_" + n), f, indent=2)
                p.chmod(0o444)
            self.__read_only = True
    
    def compare(self, dataset=None, model=None, include=False, **kw):
        """ Compare the last performance of this model on the given dataset with other ones. """
        data = []
        models = [self]
        if isinstance(model, (list, tuple)):
            models.extend(model)
        elif model is not None:
            models.append(model)
        if include:
            class Dummy:
                def __str__(self): return ""
            m = Dummy()
            m._performance = pd.read_csv(config['models'].joinpath(".performances.csv"), sep=";")
            models.append(m)
        for m in models:
            p, l = m._performance, len(data)
            for i in range(p.shape[0] - 1, -1, -1):
                r = p.iloc[i]
                d = r['Dataset']
                try:
                    r['Model']
                    row = list(r.values)
                except KeyError:
                    row = [getattr(m, "name", str(m))] + list(r.values)
                    print(row)
                if dataset is None:
                    data.insert(l - 1, row)
                elif d in dataset and isinstance(dataset, (list, tuple)) or dataset == d:
                    data.append(row)
                    break
        if len(data) == 0:
            self.logger.warning("No model selected" if dataset is None else "%s not found" % dataset)
            return
        ph = PERF_HEADERS
        h = ["Model"] + list(ph.keys())
        data = sorted(data, key=lambda row: (row[0], row[1]))
        print(mdv.main(Table(highlight_best(data, h, [0, 1, -1], ph), column_headers=h).md()))
    
    def edit(self, **kw):
        """ Edit the performance log file. """
        p = config['models'].joinpath(".performances.csv") if self.name is None else \
            config['models'].joinpath(self.name, "performance.csv")
        subprocess.call(["vd", str(p.absolute()), "--csv-delimiter", ";"], stderr=subprocess.PIPE)
    
    def list(self, algorithms=False, **kw):
        """ List all the models from the given path or all available algorithms. """
        if algorithms:
            d = [(a.name, a.description) for a in Algorithm.registry]
            r = [Section("Algorithms (%d)" % len(d)), Table(d, column_headers=["Name", "Description"])]
        else:
            d = []
            for model in Path(config['models']).listdir(Model.check):
                with model.joinpath("metadata.json").open() as meta:
                    metadata = json.load(meta)
                alg, ds = metadata['algorithm'], metadata['dataset']
                d.append([
                    model.stem,
                    alg['name'].upper(),
                    alg['description'],
                    ds['name'],
                    str(ds['executables']),
                    ",".join(sorted(ds['categories'])),
                    _shorten(",".join("%s{%d}" % i for i in sorted(ds['counts'].items(), key=lambda x: (-x[1],x[0])))),
                ])
            if len(d) == 0:
                self.logger.warning("No model found in workspace (%s)" % config['models'])
                return
            r = [Section("Models (%d)" % len(d)),
                 Table(d, column_headers=["Name", "Algorithm", "Description", "Dataset", "Size", "Categories",
                                          "Packers"])]
        print(mdv.main(Report(*r).md()))
    
    def remove(self, **kw):
        """ Remove the current model. """
        Path(config['models'].joinpath(self.name)).absolute().remove(error=False)
    
    def rename(self, name2=None, **kw):
        """ Rename the current model. """
        self.path = p = config['models'].joinpath(name2)
        if p.exists():
            self.logger.warning("%s already exists" % p)
    
    def show(self, **kw):
        """ Show an overview of the model. """
        a, ds = self._metadata['algorithm'], self._metadata['dataset']
        best_feat = [n for i, n in sorted(zip(self.classifier.feature_importances_, self._features.keys()),
                                          key=lambda x: -x[0]) if i > 0.]
        l = max(map(len, best_feat))
        best_feat = [("{: <%s}: {}" % ((l + 1) if l >= 10 and i < 9 else l)).format(n, self._features[n]) \
                     for i, n in enumerate(best_feat)]
        params = a['parameters'].keys()
        l = max(map(len, params))
        params = [("{: <%s} = {}" % l).format(*p) for p in a['parameters'].items()]
        c = List(["**Path**:         %s" % self.path,
                  "**Size**:         %s" % human_readable_size(self.path.joinpath("dump.joblib").size),
                  "**Algorithm**:    %s (%s)" % (a['description'], a['name']),
                  "**Features**:     %d \n\n\t1. %s\n\n" % (len(self._features), "\n\n\t1. ".join(best_feat)),
                  "**Parameters**: \n\n\t- %s\n\n" % "\n\t- ".join(params)])
        print(mdv.main(Report(Section("Model characteristics"), c).md()))
        ds_path = config['datasets'].joinpath(ds['name'])
        c = List(["**Path**:         %s" % ds_path,
                  "**Size**:         %s" % human_readable_size(ds_path.size),
                  "**#Executables**: %d" % ds['executables'],
                  "**Categories**:   %s" % ", ".join(ds['categories']),
                  "**Packers**:      %s" % ", ".join(ds['counts'].keys())])
        print(mdv.main(Report(Section("Reference dataset"), c).md()))
    
    def test(self, executable, labels=None, preprocessor=None, multiclass=False, feature=None, pattern=None, **kw):
        """ Test a single executable or a set of executables and evaluate metrics if labels are provided. """
        l, n, labels = self.logger, executable, labels or {}
        if self.classifier is None:
            l.warning("Model shall be trained before testing")
            return
        if not self._prepare(executable, preprocessor, multiclass, feature, pattern, True, **kw):
            return
        l.debug("Testing %s on %s..." % (self.name, n))
        prediction, dt = benchmark(self.classifier.predict)(self._data)
        dt = 1000 * dt
        try:
            proba = self.classifier.predict_proba(self._data)[:, 1]
        except AttributeError:
            proba = None
        ph = PERF_HEADERS
        h, m = list(ph.keys())[1:], self._metrics(self._target, prediction, proba)
        m2 = [ph[k](v) if v >= 0 else "-" for k, v in zip(list(ph.keys())[1:-1], m)]
        r = Section("Test results for: " + n), Table([m2 + [ph['Processing Time'](dt)]], column_headers=h)
        print(mdv.main(Report(*r).md()))
        if len(self._data) > 0:
            row = {'Model': self.name} if self.__class__ is DumpedModel else {}
            row['Dataset'] = n
            for k, v in zip(h, m):
                row[k] = v
            row['Processing Time'] = dt
            self._performance = self._performance.append(row, ignore_index=True)
            self._save()
    
    def train(self, dataset=None, algorithm=None, cv=5, n_jobs=N_JOBS, multiclass=False, feature=None, pattern=None,
              param=None, preprocessor=None, reset=False, **kw):
        """ Training method handling cross-validation. """
        l, n_cpu = self.logger, mp.cpu_count()
        try:
            cls = self.algorithm = Algorithm.get(algorithm)
            algo = cls.__name__
        except KeyError:
            l.error("%s not available" % algorithm)
            return
        if n_jobs > n_cpu:
            self.logger.warning("Maximum n_jobs is %d" % n_cpu)
            n_jobs = n_cpu
        if not self._prepare(dataset, preprocessor, multiclass, feature, pattern, **kw):
            return
        if self.name is None:
            c = sorted(collapse_categories(*self._metadata['dataset']['categories']))
            self.name = "%s_%s_%d_%s_f%d" % (self._metadata['dataset']['name'], "-".join(map(lambda x: x.lower(), c)),
                                             self._metadata['dataset']['executables'],
                                             algo.lower().replace(".", ""), len(self._features))
        if reset:
            Path(config['models'].joinpath(self.name)).absolute().remove(error=False)
        self._load()
        if self.__read_only:
            l.warning("Cannot retrain a model")
            return
        if not getattr(cls, "multiclass", True) and multiclass:
            l.error("%s does not support multiclass" % algo)
            return
        # get classifer and grid search parameters
        params = cls.parameters.get('static')
        if isinstance(cls, WekaClassifier):
            params['model'] = self
        param_grid = {k: list(v) if isinstance(v, range) else v for k, v in cls.parameters.get('cv', {}).items()}
        # apply user-defined parameters
        if param is not None:
            for p in param:
                n, v = p.split("=")
                try:
                    v = ast.literal_eval(v)
                except ValueError:
                    pass
                params[n] = v
                try:
                    del param_grid[n]
                except KeyError:
                    pass
        l.debug("Training %s on %s..." % (algo, dataset.name))
        c = cls.base(**params)
        # if a param_grid is input, perform cross-validation and select the best classifier
        if len(param_grid) > 0:
            l.debug("> Applying Grid Search (CV=%d)..." % cv)
            grid = GridSearchCV(c, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
            grid.fit(self._data, self._target.values.ravel())
            results = '\n'.join("  %0.3f (+/-%0.03f) for %r" % (m, s * 2, p) \
                for m, s, p in zip(grid.cv_results_['mean_test_score'],
                                   grid.cv_results_['std_test_score'],
                                   grid.cv_results_['params']))
            l.debug("> Grid scores:\n{}".format(results))
            l.debug("> Best parameters found:\n  {}".format(grid.best_params_))
            params.update(grid.best_params_)
            if isinstance(cls.base, WekaClassifier):
                params['model'] = self
            c = cls.base(**params)
        # now fit the (best) classifier and predict labels
        l.debug("> Fitting the classifier...")
        c.fit(self._train.data, self._train.target)
        l.debug("> Making predictions...")
        d = []
        try:
            prob1 = c.predict_proba(self._train.data)[:, 1]
            prob2 = c.predict_proba(self._test.data)[:, 1]
        except AttributeError:
            prob1 = prob2 = None
        ph = PERF_HEADERS
        h = list(ph.keys())[1:-1]
        m1 = self._metrics(self._train.target, c.predict(self._train.data), prob1)
        m1 = [ph[k](v) if v >= 0 else "-" for k, v in zip(h, m1)]
        d.append(["Train"] + m1)
        m2 = self._metrics(self._test.target, c.predict(self._test.data), prob2)
        m2 = [ph[k](v) if v >= 0 else "-" for k, v in zip(h, m2)]
        d.append(["Test"] + m2)
        h = ["."] + h
        print(mdv.main(Table(d, column_headers=h).md()))
        self.classifier = c
        self._metadata.setdefault('algorithm', {})
        self._metadata['algorithm']['name'] = algo
        self._metadata['algorithm']['description'] = cls.description
        self._metadata['algorithm']['parameters'] = params
        self._metadata['algorithm']['multiclass'] = multiclass
        self._metadata['algorithm']['preprocessors'] = list(preprocessor or [])
        self._save()
    
    def visualize(self, export=False, output_dir=".", **kw):
        """ Plot the model for visualization. """
        if self.classifier is None:
            self.logger.warning("Model shall be trained before visualizing")
            return
        f = getattr(self, "visualizations", {}).get(self._metadata['algorithm']['name'], {}) \
                                               .get(["text", "export"][export])
        if f is None:
            self.logger.warning("Visualization not available for this algorithm")
            return
        params = {'feature_names': sorted(self._features.keys())}
        if export: #FIXME
            params['?'] = str(Path(output_dir).joinpath("%s.png" % self.name))
        print(f(self.classifier, **params))
    
    @staticmethod
    def check(folder):
        try:
            Model.validate(folder)
            return True
        except ValueError as e:
            return False
    
    @staticmethod
    def validate(folder):
        f = Path(folder)
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
    def __init__(self, name=None, **kw):
        self.classifier   = joblib.load(str(name))
        self.name         = Path(name).stem
        self._features    = {}
        self._metadata    = {}
        self.__p = config['models'].joinpath(".performances.csv")    
        try:
            self._performance = pd.read_csv(str(self.__p), sep=";")
        except FileNotFoundError:
            self._performance = pd.DataFrame(columns=["Model"] + list(PERF_HEADERS.keys()))
    
    def _save(self):
        self.logger.debug("> Saving metrics to %s..." % str(self.__p))
        p = self._performance
        k = p.columns
        p = p.loc[p.round(3).drop_duplicates(subset=k[:-1]).index]
        p.to_csv(str(self.__p), sep=";", columns=k, index=False, header=True, float_format=FLOAT_FORMAT)
    
    _metrics    = Model._metrics
    _prepare    = Model._prepare
    _preprocess = Model._preprocess
    test        = Model.test

