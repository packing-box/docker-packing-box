# -*- coding: UTF-8 -*-
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import *
from tinyscript import json, logging
from tinyscript.helpers import human_readable_size, Path
from tinyscript.report import *

from .algorithms import Algorithm, WekaClassifier
from .executable import Executable
from .metrics import *
from ..common.config import config
from ..common.utils import *


__all__ = ["Model", "SCALERS"]


PERF_HEADERS = ["Accuracy", "Precision", "Recall", "F-Measure", "MCC", "AUC", "Processing Time"]
SCALERS = {
    'MA':         MaxAbsScaler,
    'MM':         MinMaxScaler,
    'No':         None,
    'PT-bc':      (PowerTransformer, {'method': "box-box"}),
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
        if load:
            self._load()
    
    @file_or_folder_or_dataset
    def _iter(self, executable, **kw):
        """ Iterate over executables using the special decorator file_or_folder_or_dataset. """
        exe = Executable(executable)
        exe.dataset = executable.dataset
        return exe
    
    def _load(self):
        """ Load model's associated files if relevant or create instance's attributes. """
        if self.name is None:
            self.classifier = None
            self._features = {}
            self._metadata = {}
            self._performance = pd.DataFrame(columns=PERF_HEADERS)
        else:
            self.path = Path(config['models'].joinpath(self.name)).absolute()
            Model.validate(self.path)
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
        return self
    
    @file_or_folder_or_dataset
    def _predict(self, executable, **kw):
        """ Predict the label of an executable or executables from a folder or for a complete dataset. """
        exe = Executable(executable)
        return exe, self.classifier.predict(pd.DataFrame(exe.features, index=[0]))
    
    def _prepare(self, dataset, scaler="MM", multiclass=False, feature=None, pattern=None, **kw):
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
        self._features_vector = sorted(self._features.keys()) if feature is None and pattern is None else \
                                sorted(feature) if isinstance(feature, (tuple, list)) else \
                                sorted(x for x in self._features.keys() if re.search(pattern, x))
        self._features = {k: {'description': self._features[k]} for k in self._features_vector if k in self._features}
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
        # replace string features to numerical values
        ds.logger.debug("> Fit the data to numerical values")
        for column in self._data:
            values = set(self._data[column].values)
            # remap strings with custom values
            if any(isinstance(v, str) for v in values):
                self._data = self._data.replace({column: {v: float(i) for i, v in enumerate(values)}})
                self._features[column]['values'] = list(values)
        # scale the data
        scaler = SCALERS[scaler]
        if scaler is not None:
            n, params = scaler.__name__, {}
            if isinstance(scaler, tuple):
                scaler, params = scaler
                n = "%s with %s" % (scaler.__name__, ", ".join("{}={}".format(*i) for i in params.items()))
            ds.logger.debug("> Scale the data (%s)" % n)
            self._data = scaler(**params).fit_transform(self._data)
        # prepare for sklearn
        class Dummy: pass
        self._train, self._test = Dummy(), Dummy()
        ds.logger.debug("> Split data and target vectors to train and test subnets")
        self._train.data, self._test.data, self._train.target, self._test.target = train_test_split(self._data,
                                                                                                    self._target)
        # prepare for Weka
        l.debug("> Create ARFF train and test files (for Weka)")
        WekaClassifier.to_arff(self)
        return True
    
    def _save(self):
        """ Save model's state to the related files. """
        l = self.logger
        if self.name is None:
            c = "-".join(map(lambda x: x.lower(), collapse_categories(*self._metadata['dataset']['categories'])))
            self.name = "%s_%s_%d_%s" % (self._metadata['dataset']['name'], c, self._metadata['dataset']['executables'],
                                         self.algorithm.name)
        self.path = Path(config['models'].joinpath(self.name)).absolute()
        self.path.mkdir(exist_ok=True)
        l.debug("Saving model %s..." % str(self.path))
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
        if not ts.Path(path2).exists():
            self.path = self.path.rename(path2)
        else:
            self.logger.warning("%s already exists" % path2)
    
    def show(self, **kw):
        """ Show an overview of the model. """
        a, ds = self._metadata['algorithm'], self._metadata['dataset']
        c = List(["**Path**:                  %s" % self.path,
                  "**Algorithm**:             %s (%s)" % (a['description'], a['name']),
                  "**Parameters**: \n\n\t- %s\n\n" % "\n\t- ".join("%s = %s" % p for p in a['parameters'].items()),
                  "**Size**:                  %s" % human_readable_size(self.path.joinpath("dump.joblib").size)])
        print(mdv.main(Report(Section("Model characteristics"), c).md()))
        c = List(["**Path**:                  %s" % config['datasets'].joinpath(ds['name']),
                  "**Number of executables**: %d" % ds['executables'],
                  "**Categories**:            %s" % ", ".join(ds['categories']),
                  "**Packers**:               %s" % ", ".join(ds['counts'].keys())])
        print(mdv.main(Report(Section("Reference dataset"), c).md()))
    
    def test(self, executable, labels=None, feature=None, pattern=None, **kw):
        """ Test a single executable or a set of executables and evaluate metrics if labels are provided. """
        if self.classifier is None:
            self.logger.warning("Model shall be trained before testing")
            return
        i, target = -1, []
        for i, exe in enumerate(self._iter(executable)):
            try:
                t = labels[e.hash]
                if self._metadata['algorithm']['multiclass']:
                    t = int(t is not None)
            except KeyError:
                self.logger.warning("%s not found in input labels" % e.hash)
                continue
            if i == 0:
                feature_names = [_ for _ in feature_names if _ in e.features]
                if len(feature_names) == 0:
                    self.logger.warning("No selectable feature ; this may be due to a model unrelated to the input")
                    return
                data = pd.DataFrame(column=feature_names)
            data.append({k: e.features[k] for k in feature_names}, ignore_index=True)
            target.append(t)
        if i < 0:
            self.logger.warning("No data")
            return
        data = pd.DataFrame(data, columns=feature_names)
        target = pd.DataFrame(target, colums=["label"])
        prediction, dt = benchmark(self.classifier.predict(data))
        proba = self.classifier.predict_proba(data)[:, 1]
        d = [["Accuracy", "Precision", "Recall", "F-Measure", "MCC", "AUC"]]
        m = metrics(target, prediction, proba, logger=self.logger)
        print(Table(d + [m]).table)
        if i > 0:
            row = {k: v for k, v in zip(d[0], m)}
            row['Dataset'] = exe.dataset or str(exe)
            row['Processing Time'] = dt
            self._performance.update(pd.DataFrame(row, index=[0]))
            self._save()
    
    def train(self, dataset=None, algorithm=None, cv=5, n_jobs=4, multiclass=False, feature=None, pattern=None, **kw):
        """ Training method handling cross-validation. """
        if not self._prepare(dataset, **kw):
            return
        l = self.logger
        try:
            cls = self.algorithm = Algorithm.get(algorithm)
            if isinstance(cls, WekaClassifier):
                params['model'] = self
        except KeyError:
            l.error("%s not available" % algorithm)
            self.classifier = None
            return
        # get classifer and grid search parameters
        params = cls.parameters.get('static')
        param_grid = {k: list(v) if isinstance(v, range) else v for k, v in cls.parameters.get('cv').items()}
        l.debug("Processing %s..." % algorithm)
        c = cls.base(**params)
        # if a param_grid is input, perform cross-validation and select the best classifier
        if len(param_grid) > 0:
            l.debug("> Applying Grid Search (CV=%d)..." % cv)
            grid = GridSearchCV(c, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
            grid.fit(self._data, self._target)
            l.debug("> Best parameters found:\n  {}".format(grid.best_params_))
            results = '\n'.join("  %0.3f (+/-%0.03f) for %r" % (m, s * 2, p) \
                for m, s, p in zip(grid.cv_results_['mean_test_score'],
                                   grid.cv_results_['std_test_score'],
                                   grid.cv_results_['params']))
            l.debug(" > Grid scores:\n{}".format(results))
            params.update(grid.best_params_)
            if isinstance(cls.base, WekaClassifier):
                params['model'] = self
            c = cls.base(**params)
        # now fit the (best) classifier and predict labels
        l.debug("> Fitting the classifier...")
        c.fit(self._train.data, self._train.target)
        l.debug("> Making predictions...")
        self._test.prediction = c.predict(self._test.data)
        self._test.proba = c.predict_proba(self._test.data)[:, 1]
        self.classifier = c
        self._metadata.setdefault('algorithm', {})
        self._metadata['algorithm']['name'] = algorithm
        self._metadata['algorithm']['description'] = cls.description
        self._metadata['algorithm']['parameters'] = params
        self._metadata['algorithm']['multiclass'] = multiclass
        self._save()
    
    def visualize(self, export=False, output_dir=".", **kw):
        """ Plot the model for visualization. """
        if self.classifier is None:
            self.logger.warning("Model shall be trained before visualizing")
            return
        f = getattr(cls, "visualizations", {}).get(self._metadata['algorithm']['name'], {}) \
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

