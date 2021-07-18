# -*- coding: UTF-8 -*-
import joblib
from sklearn.model_selection import GridSearchCV
from tinyscript import logging, ts
from tinyscript.report import *

from .algorithms import *
from .metrics import *
from ..utils import *


__all__ = ["Model"]


PERF_HEADERS = ["Time", "Dataset", "Accuracy", "Processing Time"]


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
        p = config['workspace'].joinpath("models")
        p.mkdir(exist_ok=True)
        self.__read_only = False
        self.name = name
        if load:
            self._load()
    
    def _load(self):
        """ Load model's associated files if relevant or create instance's attributes. """
        if self.name is None:
            self.classifier = None
            self._features = {}
            self._metadata = {}
            self._performance = pd.DataFrame()
        else:
            self.path = ts.Path(config.path.joinpath("models", self.name)).absolute()
            Model.validate(self.path)
            for n in ["dump", "features", "metadata", "performance"]:
                p = self.path.joinpath(n + (".joblib" if n == "dump" else ".csv" if n == "performance" else ".json"))
                l.debug("Loading %s..." % str(p))
                if n == "dump":
                    self.classifier = joblib.load(str(p))
                    self.__read_only = True
                elif n == "performance":
                    self._performance = pd.read_csv(str(p), sep=";")
                else:
                    with p.open() as f:
                        setattr(self, "_" + n, json.load(f))
        return self
    
    def _prepare(self, dataset):
        """ Prepare the Model instance based on the given Dataset instance. """
        ds, l = dataset, self.logger
        # copy relevant information from the input dataset
        l.debug("Preparing model...")
        self._metadata['dataset'] = {k: v for k, v in dataset._metadata.items()}
        self._metadata['dataset']['path'] = str(dataset.path)
        self._metadata['dataset']['name'] = dataset.path.stem
        self._features = {k: v for k, v in dataset._features.items()}
        # compute and attach sets from dataset._data and bind them to the instance.
        ds.logger.debug("> Train and test subsets")
        data = ds._data.loc[:, ds._data.columns not in ds.FIELDS]
        target = ds._data.loc[:, ds._data.columns == "label"]
        # scale the data
        data = MinMaxScaler().fit_transform(data)
        # prepare for sklearn
        class Dummy: pass
        self.train, self.test = Dummy(), Dummy()
        self.train.data, self.test.data, self.train.target, self.test.target = train_test_split(data, target)
        # prepare for Weka
        l.debug("> ARFF train and test files...")
        WekaClassifier.to_arff(self)
    
    def _save(self):
        """ Save model's state to the related files. """
        l = self.logger
        if self.name is None:
            c = "-".join(map(lambda x: x.lower(), collapse_categories(self._metadata['dataset']['categories'])))
            self.name = "%s_%s_%d" % (self._metadata['dataset']['name'], c, self._metadata['dataset']['executables'])
        self.path = ts.Path(config.path.joinpath("models", self.name)).absolute()
        l.debug("Saving model %s..." % str(self.path))
        p = self.path.joinpath("performance.csv")
        l.debug("> %s" % str(p))
        self._performances.to_csv(str(p), sep=";", columns=PERF_HEADERS, index=False, header=True)
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
            d = [list(row) for row in CLASSIFIERS['descriptions'].items()]
            r = [Section("Algorithms (%d)" % len(d)), Table(d, column_headers=["Name", "Description"])]
        else:
            d = []
            for model in ts.Path(config['workspace'].joinpath("models")).listdir(Model.check):
                with model.joinpath("metadata.json").open() as meta:
                    metadata = json.load(meta)
                alg, ds = metadata['algorithm'], metadata['dataset']
                d.append([
                    alg['name'],
                    alg['description'],
                    ds['name'],
                    str(ds['executables']),
                    ",".join(sorted(ds['categories'])),
                    ",".join("%s{%d}" % i for i in sorted(ds['counts'].items(), key=lambda x: -x[1])),
                ])
            if len(d) == 0:
                return
            r = [Section("Models (%d)" % len(d)),
                 Table(d, column_headers=["Algorithm", "Description", "Dataset", "Size", "Categories", "Packers"])]
        print(mdv.main(Report(*r).md()))
    
    @file_or_folder_or_dataset
    def test(self, executable, **kwargs):
        """ Test method. """
        metrics(executable, logger=self.logger)
        data = [["", "Model", "Accuracy", "Precision", "Recall", "F-Measure", "MCC", "AUC"]]
        table = Table(highlight_best(table_data), "Results ([CRESCI] - {})".format(c))
        print(table.table)
        return self.classifier.predict(data)
    
    def train(self, name, cv=3, n_jobs=4, **kw):
        """ Training method handling the cross-validation. """
        l = self.logger
        try:
            cls = CLASSIFIERS['classes'][name]
            if isinstance(cls, WekaClassifier):
                params['model'] = self
        except KeyError:
            l.error("%s not available" % name)
            self.classifier = None
            return
        # get classifer and grid search parameters
        try:
            params = STATIC_PARAM[name]
        except KeyError:
            params = {}
        try:
            param_grid = CV_PARAM[name]
        except KeyError:
            param_grid = None
        c = cls(**params)
        c.labels = dataset.labels
        # if a param_grid is input, perform cross-validation and select the best classifier
        l.debug("Processing %s..." % name)
        if param_grid is not None:
            l.debug("> Applying Grid Search (CV=%d)..." % cv)
            grid = GridSearchCV(c, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
            grid.fit(dataset.data, dataset.target)
            l.debug("> Best parameters found:\n  {}".format(grid.best_params_))
            results = '\n'.join("  %0.3f (+/-%0.03f) for %r" % (m, s * 2, p) \
                for m, s, p in zip(grid.cv_results_['mean_test_score'],
                                   grid.cv_results_['std_test_score'],
                                   grid.cv_results_['params']))
            l.debug(" > Grid scores:\n{}".format(results))
            params.update(grid.best_params_)
            if isinstance(cls, WekaClassifier):
                params['model'] = self
            c = cls(**params)
        # now fit the (best) classifier and predict labels
        l.debug("> Fitting the classifier...")
        c.fit(self.train.data, self.train.target)
        l.debug("> Making predictions...")
        self.test.prediction = c.predict(dataset.test.data)
        self.test.proba = c.predict_proba(dataset.test.data)[:, 1]
        self.classifier = c
        setdefault(self._metadata['algorithm'], {})
        self._metadata['algorithm']['name'] = name
        self._metadata['algorithm']['name'] = CLASSIFIERS['descriptions'][name]
        self._metadata['algorithm']['parameters'] = params
    
    @staticmethod
    def check(folder):
        try:
            Model.validate(folder)
            return True
        except ValueError as e:
            return False
    
    @staticmethod
    def validate(folder):
        f = ts.Path(folder)
        if not f.exists():
            raise ValueError("Folder does not exist")
        if not f.is_dir():
            raise ValueError("Input is not a folder")
        for fn in ["dump.joblib", "features.json", "metadata.json", "performance.csv"]:
            if not f.joinpath(fn).exists():
                raise ValueError("Folder does not have %s" % fn)
        return f

