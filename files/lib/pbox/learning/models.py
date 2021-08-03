# -*- coding: UTF-8 -*-
import joblib
import pandas as pd
from sklearn.base import is_classifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import *
from tinyscript import ast, json, logging
from tinyscript.helpers import human_readable_size, Path
from tinyscript.report import *

from .algorithms import Algorithm, WekaClassifier
from .dataset import *
from .executable import Executable
from ..common.config import config
from ..common.utils import *


__all__ = ["DumpedModel", "Model", "PREPROCESSORS"]


PERF_HEADERS = ["Dataset", "Accuracy", "Precision", "Recall", "F-Measure", "MCC", "AUC", "Processing Time"]
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
    
    @file_or_folder_or_dataset
    def _iter(self, executable, **kw):
        """ Iterate over executables using the file_or_folder_or_dataset decorator applied to an input dataset. """
        return Executable(executable)
    
    def _load(self):
        """ Load model's associated files if relevant or create instance's attributes. """
        if self.name is None:
            self._features = {}
            self._metadata = {}
            self._performance = pd.DataFrame(columns=PERF_HEADERS)
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
        accuracy = float(tp + tn) / (tp + tn + fp + fn)
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)  # or also sensitivity
        f_measure = float(2 * precision * recall) / (precision + recall)
        mcc = matthews_corrcoef(target, prediction)
        auc = roc_auc_score(target.label, proba)
        # return metrics for further display
        return [accuracy, precision, recall, f_measure, mcc, auc]
    
    def _prepare(self, dataset, preprocessor=None, multiclass=False, feature=None, pattern=None, sort=False, **kw):
        """ Prepare the Model instance based on the given Dataset instance. """
        ds, l = dataset, self.logger
        if not getattr(ds, "is_valid", lambda: False)():
            l.error("Not a valid input dataset")
            return False
        # copy relevant information from the input dataset
        l.debug("Preparing the dataset...")
        self._metadata['dataset'] = {k: v for k, v in ds._metadata.items()}
        self._metadata['dataset']['path'] = str(ds.path)
        self._metadata['dataset']['name'] = ds.path.stem
        # first case: Dataset (with files) ; features must be computed
        try:
            ds.files  # this triggers an exception if dataset is fileless
            self._data = pd.DataFrame()
            for exe in ds:
                e = Executable(ds.files.joinpath(exe.hash))
                d = {'label': exe.label}
                d.update(e.data)
                self._data = self._data.append(d, ignore_index=True)
                self._features.update(e.features)
        # second case: FilelessDataset ; features are retrieved from dataset._data
        except AttributeError:
            self._data = ds._data.loc[:, ~ds._data.columns.isin(["hash"] + Executable.FIELDS)]
            self._features.update(ds._features)
        # compute the list of selected features
        self._features_vector = list(self._features.keys()) if feature is None and pattern is None else \
                                feature if isinstance(feature, (tuple, list)) else \
                                [x for x in self._features.keys() if re.search(pattern, x)]
        if sort:
            self._features_vector.sort()
        self._features = {k: self._features[k] for k in self._features_vector if k in self._features}
        if len(self._features) == 0:
            self.logger.warning("No feature selected")
            return False
        self._target = self._data.loc[:, self._data.columns == "label"]
        self._data = self._data.loc[:, self._data.columns.isin(self._features_vector)]
        # compute and attach sets from dataset._data and bind them to the instance.
        ds.logger.debug("> Split dataset to data and target vectors")
        if not multiclass:  # convert to binary class
            self._target = self._target.fillna(0).where(pd.isnull(self._target), 1).astype('int')
        else:
            self._target = self._target.fillna("")
        self.labels = set(self._target.label.values)
        self._data = self._data.reindex(self._features_vector, axis=1)
        self._preprocess(*preprocessor)
        # prepare for sklearn
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
                "Scale" if n.endswith("Scaler") else "Normalize" if n.endswith("Normalizer") else \
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
        self._performance.to_csv(str(p), sep=";", columns=PERF_HEADERS, index=False, header=True)
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
                    ",".join("%s{%d}" % i for i in sorted(ds['counts'].items(), key=lambda x: -x[1])),
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
        self.path.remove(error=False)
    
    def rename(self, path2=None, **kw):
        """ Rename the current model. """
        if not Path(path2).exists():
            self.path = self.path.rename(path2)
        else:
            self.logger.warning("%s already exists" % path2)
    
    def show(self, **kw):
        """ Show an overview of the model. """
        a, ds = self._metadata['algorithm'], self._metadata['dataset']
        best_feat = [n for i, n in sorted(zip(self.classifier.feature_importances_, self._features.keys()),
                                          key=lambda x: -x[0]) if i > 0.]
        l = max(map(len, best_feat))
        best_feat = [("{: <%s}: {}" % (l + 1) if l >= 10 and n < 10 else l).format(n, self._features[n]) \
                     for n in best_feat]
        params = a['parameters'].keys()
        l = max(map(len, params))
        params = [("{: <%s} = {}" % l).format(*p) for p in a['parameters'].items()]
        c = List(["**Path**:         %s" % self.path,
                  "**Algorithm**:    %s (%s)" % (a['description'], a['name']),
                  "**Features**:     %d \n\n\t1. %s\n\n" % (len(self._features), "\n\n\t1. ".join(best_feat)),
                  "**Parameters**: \n\n\t- %s\n\n" % "\n\t- ".join(params),
                  "**Size**:         %s" % human_readable_size(self.path.joinpath("dump.joblib").size)])
        print(mdv.main(Report(Section("Model characteristics"), c).md()))
        c = List(["**Path**:         %s" % config['datasets'].joinpath(ds['name']),
                  "**#Executables**: %d" % ds['executables'],
                  "**Categories**:   %s" % ", ".join(ds['categories']),
                  "**Packers**:      %s" % ", ".join(ds['counts'].keys())])
        print(mdv.main(Report(Section("Reference dataset"), c).md()))
    
    def test(self, executable, labels=None, preprocessor=None, feature=None, pattern=None, sep=";", **kw):
        """ Test a single executable or a set of executables and evaluate metrics if labels are provided. """
        l, labels = self.logger, labels or {}
        if self.classifier is None:
            l.warning("Model shall be trained before testing")
            return
        e, i = executable, -1
        try:
            d, n, i = pd.read_csv(str(e), sep=sep), Path(e).filename, 1
            self._data, target = d.loc[:, d.columns != "label"], d.loc[:, d.columns == "label"]
        except AttributeError:  # 'DataFrame' object has no attribute 'label'
            l.error(d.columns)
            l.warning("This error may be caused by a bad CSV separator ; you can set it with --sep")
            raise
        except (KeyError, OSError):
            if config['datasets'].joinpath(e).exists():
                e = open_dataset(e)
                labels.update({x.hash: x.label for x in e._data.itertuples()})
            self._data, target = pd.DataFrame(columns=["label"]), []
            for i, exe in enumerate(self._iter(e)):
                exe.selection = feature or pattern
                try:
                    t = labels[exe.hash]
                    if not self._metadata['algorithm']['multiclass']:
                        t = int(str(t) not in ["", "nan", "None"])
                except KeyError:
                    l.warning("%s not found in input labels" % exe.hash)
                    continue
                if i == 0:
                    feature_names = exe.features.keys()
                    if len(feature_names) == 0:
                        l.warning("No selectable feature ; this may be due to a model unrelated to the input")
                        return
                    if sort:
                        feature_names.sort()
                self._data = self._data.append(exe.data, ignore_index=True)
                target.append(t)
            if i < 0:
                l.warning("No data")
                return
            self._data = self._data.reindex(feature_names, axis=1)
            target = pd.DataFrame(target, columns=["label"])
            n = getattr(e, "name", str(exe))
        l.debug("Testing %s on %s..." % (self.name, n))
        self._preprocess(*(preprocessor or self._metadata.get('algorithm', {}).get('preprocessors', [])))
        prediction, dt = benchmark(self.classifier.predict)(self._data)
        dt = 1000 * dt
        proba = self.classifier.predict_proba(self._data)[:, 1]
        h, m = PERF_HEADERS[1:], self._metrics(target, prediction, proba)
        r = Section("Test results for: " + n), Table([m + ["%.3fms" % dt]], column_headers=h, flt_fmt="%.3f")
        print(mdv.main(Report(*r).md()))
        if i > 0:
            row = {k: v for k, v in zip(h, m)}
            row['Dataset'] = n
            row['Processing Time'] = dt
            self._performance = self._performance.append(row, ignore_index=True)
            self._save()
    
    def train(self, dataset=None, algorithm=None, cv=5, n_jobs=4, multiclass=False, feature=None, pattern=None,
              param=None, preprocessor=None, reset=False, **kw):
        """ Training method handling cross-validation. """
        if reset:
            self._remove()
        if not self._prepare(dataset, preprocessor, multiclass, feature, pattern):
            return
        if self.name is None:
            c = sorted(collapse_categories(*self._metadata['dataset']['categories']))
            self.name = "%s_%s_%d_%s_f%d" % (self._metadata['dataset']['name'], "-".join(map(lambda x: x.lower(), c)),
                                             self._metadata['dataset']['executables'],
                                             algorithm.lower().replace(".", ""), len(self._features))
        self._load()
        l = self.logger
        if self.__read_only:
            l.warning("Cannot retrain a model")
            return
        try:
            cls = self.algorithm = Algorithm.get(algorithm)
        except KeyError:
            l.error("%s not available" % algorithm)
            return
        if not getattr(cls, "multiclass", True) and multiclass:
            l.error("%s does not support multiclass" % algorithm)
            return
        # get classifer and grid search parameters
        params = cls.parameters.get('static')
        if isinstance(cls, WekaClassifier):
            params['model'] = self
        param_grid = {k: list(v) if isinstance(v, range) else v for k, v in cls.parameters.get('cv').items()}
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
        l.debug("Training %s on %s..." % (algorithm, dataset.name))
        c = cls.base(**params)
        # if a param_grid is input, perform cross-validation and select the best classifier
        if len(param_grid) > 0:
            l.debug("> Applying Grid Search (CV=%d)..." % cv)
            grid = GridSearchCV(c, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
            grid.fit(self._data, self._target)
            results = '\n'.join("  %0.3f (+/-%0.03f) for %r" % (m, s * 2, p) \
                for m, s, p in zip(grid.cv_results_['mean_test_score'],
                                   grid.cv_results_['std_test_score'],
                                   grid.cv_results_['params']))
            l.debug(" > Grid scores:\n{}".format(results))
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
        d.append(["Train"] + self._metrics(self._train.target,
                                           c.predict(self._train.data),
                                           c.predict_proba(self._train.data)[:, 1]))
        d.append(["Test"] + self._metrics(self._test.target,
                                          c.predict(self._test.data),
                                          c.predict_proba(self._test.data)[:, 1]))
        h = ["."] + PERF_HEADERS[1:-1]
        print(mdv.main(Table(d, column_headers=h, flt_fmt="%.3f").md()))
        self.classifier = c
        self._metadata.setdefault('algorithm', {})
        self._metadata['algorithm']['name'] = algorithm
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
        except OSError:
            self._performance = pd.DataFrame(columns=PERF_HEADERS)
    
    def _save(self):
        self.logger.debug("> Saving metrics to %s..." % str(self.__p))
        p = self._performance
        p = p.loc[p.round(3).drop_duplicates(subset=PERF_HEADERS[:-2]).index]
        p.to_csv(str(self.__p), sep=";", columns=PERF_HEADERS, index=False, header=True, float_format="%.3f")
    
    _metrics    = Model._metrics
    _preprocess = Model._preprocess
    test        = Model.test

