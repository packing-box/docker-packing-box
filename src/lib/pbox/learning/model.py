# -*- coding: UTF-8 -*-
import joblib
import multiprocessing as mp
import numpy as np
import pandas as pd
from _pickle import UnpicklingError
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from tinyscript import ast, json, logging, subprocess
from tinyscript.helpers import human_readable_size, is_generator, Path, TempPath
from tinyscript.report import *

from .algorithm import Algorithm, WekaClassifier
from .dataset import *
from .executable import Executable
from .features import Features
from .visualization import *
from ..common.config import *
from ..common.utils import *


__all__ = ["open_model", "DumpedModel", "Model", "N_JOBS", "PREPROCESSORS"]


FLOAT_FORMAT = "%.6f"
N_JOBS = mp.cpu_count() // 2
# FIXME: parametrize all preprocessors, knowing that scalers fail within pipelines (no method '(.*_)?predict')
PREPROCESSORS = {
    'MA':         MaxAbsScaler,
    'MM':         MinMaxScaler,
    'Norm':       Normalizer,
#    'OneHot':     (OneHotEncoder, {'drop': "if_binary", 'handle_unknown': "ignore"}),
#    'Ord':        OrdinalEncoder,
#    'PT-bc':      (PowerTransformer, {'method': "box-cox"}),
#    'PT-yj':      (PowerTransformer, {'method': "yeo-johnson"}),
#    'QT-normal':  (QuantileTransformer, {'output_distribution': "normal"}),
#    'QT-uniform': (QuantileTransformer, {'output_distribution': "uniform"}),
#    'Rob':        (RobustScaler, {'quantile_range': (25, 75)}),
    'PCA':        (PCA, {'n_components': 2}),
    'Std':        StandardScaler,
}


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


class DebugPipeline:
    """ Proxy class for attaching a logger ; NOT subclassed as logger cannot be pickled (cfr Model._save). """
    def __init__(self, *args, **kwargs):
        self.pipeline = None
        kwargs['verbose'] = False  # ensure no verbose, this is managed by self.logger with self._log_message
        self._args = (args, kwargs)
    
    def __getattribute__(self, name):
        if name in ["_args", "_log_message", "append", "logger", "pipeline", "pop", "serialize"]:
            return object.__getattribute__(self, name)
        return object.__getattribute__(object.__getattribute__(self, "pipeline"), name)
    
    def _log_message(self, step_idx):
        """ Overload original method to display messages with our own logger.
        NB: verbosity is controlled via the logger, therefore we output None so that it is not natively displayed.  """
        if not getattr(Pipeline, "silent", False):
            name, _ = self.steps[step_idx]
            logger.info("(step %d/%d) Processing %s" % (step_idx + 1, len(self.steps), name))
    Pipeline._log_message = _log_message
    
    def append(self, step):
        if not isinstance(step, tuple):
            step = (step.__class__.__name__, step)
        if self.pipeline is None:
            self.pipeline = Pipeline([step], *self._args[0], **self._args[1])
        else:
            self.pipeline.steps.append(step)
    
    def pop(self, idx=None):
        self.pipeline.steps.pop() if idx is None else self.pipeline.steps.pop(idx)


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
        global logger
        logger = self.logger
        self.__read_only = False
        self._features, self._metadata = {}, {}
        self._performance = pd.DataFrame(columns=PERF_HEADERS.keys())
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
                self._performance = pd.read_csv(str(p), sep=";")
            else:
                with p.open() as f:
                    setattr(self, "_" + n, json.load(f))
    
    def _metrics(self, target, prediction, proba):
        """ Metrics computation method. """
        l = self.logger
        if self.dataset.labelling < 1.:
            l.warning("cannot compute metrics ; data is not completely labelled")
            return [-1] * 6
        l.debug("computing metrics...")
        kw = {} if self._metadata['algorithm']['multiclass'] else {'labels': [0, 1]}
        cmatrix = confusion_matrix(target, prediction, **kw)
        l.debug("TN: %d ; FP: %d ; FN: %d ; TP: %d" % (cmatrix[0][0], cmatrix[0][1], cmatrix[1][0], cmatrix[1][1]))
        accuracy, precision, recall, f_measure = metrics(*cmatrix.ravel())
        mcc = matthews_corrcoef(target, prediction)
        try:
            auc = roc_auc_score(target, proba)
        except ValueError:
            auc = -1
        except:
            l.info(target)
            l.info(proba)
            raise
        return [accuracy, precision, recall, f_measure, mcc, auc]
    
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
            l.debug("preparing dataset...")
            self._metadata['dataset'] = {k: v for k, v in ds._metadata.items()}
            self._metadata['dataset']['path'] = str(ds.path)
            self._metadata['dataset']['name'] = ds.path.stem
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
        l.info("%s dataset:  %s" % (["Reference", "Test"][data_only], ds))
        self._data, self._target = pd.DataFrame(), pd.DataFrame(columns=["label"])
        Features.boolean_only = self.algorithm.boolean
        # start input dataset parsing
        def __parse(exes, label=True):
            l.info("Computing features...")
            if not isinstance(exes, list) and not is_generator(exes):
                exes = [exes]
            for exe in exes:
                if not isinstance(exe, Executable):
                    exe = Executable(str(exe))
                exe.selection = feature
                if not data_only:
                    self._features.update(exe.features)
                self._data = self._data.append(exe.data, ignore_index=True)
                if label:
                    self._target = self._target.append({'label': labels.get(exe.hash)}, ignore_index=True)
        # case 1: handle a fileless dataset (where features are already computed)
        if isinstance(ds, FilelessDataset):
            l.info("Loading features...")
            for exe in ds:
                break
            exe = Executable(exe)
            exe.selection = feature
            if not data_only:
                self._features.update(exe.features)
            self._data = ds._data[list(exe.features.keys())]
            self._target = ds._data.loc[:, ds._data.columns == "label"]
        # case 2: handle a normal dataset (features shall still be computed)
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
        self._target = self._target.replace(LABELS_BACK_CONV)
        # ensure features are sorted and data has its columns sorted too
        self._features = {k: v for k, v in sorted(self._features.items(), key=lambda x: x[0]) if v != ""}
        try:
            self._data = self._data[sorted(self._features.keys())]
        except KeyError as e:
            missing_cols = ast.literal_eval(e.args[0].replace("not in index", ""))
            for col in missing_cols:
                self._features[col] = ""
            self._data = self._data.reindex(columns=sorted(self._features.keys()))
        if unlabelled:
            self._target['label'] = np.nan
        self._data, self._target = self._data.fillna(-1), self._target.fillna("")
        if not multiclass:  # convert to binary class
            self._target.loc[self._target.label == "", "label"] = 0
            self._target.loc[self._target.label != 0, "label"] = 1
            self._target = self._target.astype('int')
        # create the pipeline if it does not exist (i.e. while training)
        if not data_only:
            self.pipeline = DebugPipeline()
            l.info("Making pipeline...")
            for p in preprocessor:
                p, params = PREPROCESSORS.get(p, p), {}
                if isinstance(p, tuple):
                    try:
                        p, params = p
                    except ValueError:
                        l.error("Bad preprocessor format: %s" % p)
                        raise
                    m = "%s with %s" % (p.__name__, ", ".join("{}={}".format(*i) for i in params.items()))
                else:
                    m = p.__name__
                n = p.__name__
                v = "transform" if n.endswith("Transformer") else "encode" if n.endswith("Encoder") else \
                    "standardize" if n.endswith("Scaler") else "normalize" if n.endswith("Normalizer") or n == "PCA" \
                     else "discretize" if n.endswith("Discretizer") else "preprocess"
                self.pipeline.append(("%s (%s)" % (v, m), p(**params)))
        # if only data is to be processed (i.e. while testing), stop here, the rest is for training the model
        else:
            return True
        # apply variance threshold of 0.0 to remove useless features and rectify the list of features
        l.debug("> remove 0-variance features")
        selector = VarianceThreshold()
        selector.fit(self._data)
        self._data = self._data[self._data.columns[selector.get_support(indices=True)]]
        self._features = {k: v for k, v in self._features.items() if k in self._data.columns}
        # prepare for training and testing sets
        class Dummy: pass
        self._train, self._test = Dummy(), Dummy()
        ds.logger.debug("> split data and target vectors to train and test subsets")
        self._train.data, self._test.data, self._train.target, self._test.target = \
            train_test_split(self._data, self._target, test_size=.2, random_state=42, stratify=self._target)
        return True
    
    def _save(self):
        """ Save model's state to the related files. """
        l = self.logger
        if not self.__read_only and self.path.exists():
            l.warning("This model already exists !")
            return
        self.path.mkdir(exist_ok=True)
        l.debug("%s model %s..." % (["saving", "updating"][self.__read_only], str(self.path)))
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
        l.debug("> saving %s..." % p.basename)
        self._performance.to_csv(str(p), sep=";", columns=PERF_HEADERS.keys(), index=False, header=True,
                                 float_format=FLOAT_FORMAT)
    
    def compare(self, dataset=None, model=None, include=False, **kw):
        """ Compare the last performance of this model on the given dataset with other ones. """
        l, data, models = self.logger, [], [self]
        l.debug("comparing models...")
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
            m.name = ""
            m._performance = pd.read_csv(config['models'].joinpath(".performances.csv"), sep=";")
            models.append(m)
        # compute performance data
        for m in models:
            p, dlen = m._performance, len(data)
            for i in range(p.shape[0]-1, -1, -1):
                r = p.iloc[i]
                d = r['Dataset']
                try:
                    r['Model']
                    row = list(r.values)
                except KeyError:
                    row = [getattr(m, "name", str(m))] + list(r.values)
                # if all performance metrics are 100% AND the dataset name matches this included in the model's name,
                #  assume it is the reference dataset and discard these metrics
                if all(x == 1.0 for x in row[2:7]) and m.name.split("_", 1)[0] == d:
                    continue
                row = [[v, "-"][v < 0.] if isinstance(v, (int, float)) else v for v in row]
                if dataset is None:
                    data.insert(dlen-1, row)
                elif d in dataset and isinstance(dataset, (list, tuple)) or dataset == d:
                    data.append(row)
                    break
        if len(data) == 0:
            l.warning("No model selected" if dataset is None else "%s not found for the given model" % \
                      [dataset[0], "Datasets"][len(dataset) > 1])
            return
        # display performance data
        ph = PERF_HEADERS
        h = ["Model"] + list(ph.keys())
        data = sorted(data, key=lambda row: (row[0], row[1]))
        print(mdv.main(Table(highlight_best(data, h, [0, 1, -1], ph), column_headers=h).md()))
    
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
                counts = {k: v for k, v in ds['counts'].items() if k != NOT_PACKED}
                d.append([
                    model.stem,
                    alg['name'].upper(),
                    alg['description'],
                    ds['name'],
                    str(ds['executables']),
                    ",".join(sorted(ds['formats'])),
                    shorten_str(",".join("%s{%d}" % i for i in sorted(counts.items(), key=lambda x: (-x[1], x[0])))),
                ])
            if len(d) == 0:
                self.logger.warning("No model found in the workspace (%s)" % config['models'])
                return
            r = [Section("Models (%d)" % len(d)),
                 Table(d, column_headers=["Name", "Algorithm", "Description", "Dataset", "Size", "Formats",
                                          "Packers"])]
        print(mdv.main(Report(*r).md()))
    
    def preprocess(self, **kw):
        """ Preprocess an input dataset given selected features and display it with visidata for review. """
        kw['data_only'] = True
        if not self._prepare(**kw):
            self.logger.debug("could not prepare dataset")
            return
        self.pipeline.fit(self._data, self._target)  # fit_transform with all the transformers, not with the estimator
        tmp_p = TempPath(prefix="model-preprocess-", length=8)
        tmp_f = tmp_p.tempfile("data.csv")
        self._data.to_csv(str(tmp_f), sep=";", index=False, header=True)
        edit_file(tmp_f, logger=self.logger)
        tmp_p.remove()
    
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
            l = max(map(len, best_feat))
            best_feat = [("{: <%s}: {} (%.3f)" % ((l + 1) if l >= 10 and i < 9 else l, p[0])) \
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
        print(mdv.main(Report(Section("Model characteristics"), c).md()))
        ds_path = config['datasets'].joinpath(ds['name'])
        c = List(["**Path**:         %s" % ds_path,
                  "**Size**:         %s" % human_readable_size(ds_path.size),
                  "**#Executables**: %d" % ds['executables'],
                  "**Formats**:      %s" % ", ".join(ds['formats']),
                  "**Packers**:      %s" % ", ".join(x for x in ds['counts'].keys() if x != NOT_PACKED)])
        print(mdv.main(Report(Section("Reference dataset"), c).md()))
    
    def test(self, executable, **kw):
        """ Test a single executable or a set of executables and evaluate metrics. """
        l, ds, a, unlab = self.logger, executable, self._metadata['algorithm'], kw.get('unlabelled', False)
        if len(self.pipeline.steps) == 0:
            l.warning("Model shall be trained before testing")
            return
        kw['data_only'], kw['dataset'] = True, ds
        if not self._prepare(**kw):
            return
        l.debug("Testing %s on %s..." % (self.name, ds))
        prediction, dt = benchmark(self.pipeline.predict)(self._data)
        dt = 1000 * dt
        try:
            proba, dt2 = benchmark(self.pipeline.predict_proba)(self._data)
            dt += 1000 * dt2
            proba = proba[:, 1]
        except AttributeError:
            proba = None
        ph = PERF_HEADERS
        h, m = list(ph.keys())[1:], self._metrics(self._target, prediction, proba)
        m2 = [ph[k](v) if v >= 0 else "-" for k, v in zip(list(ph.keys())[1:-1], m)]
        r = Section("Test results for: " + ds), Table([m2 + [ph['Processing Time'](dt)]], column_headers=h)
        print(mdv.main(Report(*r).md()))
        if len(self._data) > 0:
            row = {'Model': self.name} if self.__class__ is DumpedModel else {}
            row['Dataset'] = str(ds) + ["", "(unlabelled)"][unlab]
            for k, v in zip(h, m):
                row[k] = v
            row['Processing Time'] = dt
            self._performance = self._performance.append(row, ignore_index=True)
            self._save()
    
    def train(self, algorithm=None, cv=5, n_jobs=N_JOBS, param=None, reset=False, **kw):
        """ Training method handling cross-validation. """
        l, n_cpu, ds = self.logger, mp.cpu_count(), kw['dataset']
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
        self._metadata['algorithm']['multiclass'] = kw.get('multiclass', False)
        self._metadata['algorithm']['preprocessors'] = kw['preprocessor']
        # check that, if the algorithm is supervised, it has full labels
        if cls.labelling == "full" and ds.labelling < 1.:
            l.error("'%s' won't work with a dataset that is not fully labelled" % algo)
            return
        # check that, if the algorithm is semi-supervised, it is not labelled at all
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
            self.name = "%s_%s_%d_%s_f%d" % (ds.path.stem, "-".join(map(lambda x: x.lower(), c)),
                                             ds._metadata['executables'],
                                             algo.lower().replace(".", ""), len(self._features))
        if reset:
            self.path.remove(error=False)
        self._load()
        if self.__read_only:
            l.warning("Cannot retrain a model")
            l.warning("You can remove it first with the following command: model purge %s" % self.name)
            return
        if not getattr(cls, "multiclass", True) and kw.get('multiclass', False):
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
        l.info("Training model...")
        self.pipeline.append((cls.name, cls.base(**params)))
        # if a param_grid is input, perform cross-validation and select the best classifier
        _convert = lambda d: {k.split("__", 1)[1]: v for k, v in d.items()}
        if len(param_grid) > 0:
            l.debug("> applying Grid Search (CV=%d)..." % cv)
            Pipeline.silent = True
            # as we use a pipeline, we need to rename all parameters to [estimator name]__[parameter]
            param_grid = {"%s__%s" % (self.pipeline.steps[-1][0], k): v for k, v in param_grid.items()}
            grid = GridSearchCV(self.pipeline.pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
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
        if cls.labelling != "none":
            l.debug("> making predictions...")
            d = []
            ph = PERF_HEADERS
            h = list(ph.keys())[1:-1]
            m1 = self._metrics(self._train.target,
                               self.pipeline.predict(self._train.data),
                               self.pipeline.predict_proba(self._train.data)[:, 1])
            m1 = [ph[k](v) if v >= 0 else "-" for k, v in zip(h, m1)]
            d.append(["Train"] + m1)
            m2 = self._metrics(self._test.target,
                               self.pipeline.predict(self._test.data),
                               self.pipeline.predict_proba(self._test.data)[:, 1])
            m2 = [ph[k](v) if v >= 0 else "-" for k, v in zip(h, m2)]
            d.append(["Test"] + m2)
            h = ["."] + h
            print(mdv.main(Report(Title("Name: %s" % self.name), Table(d, column_headers=h)).md()))
        l.info("Parameters:\n- %s" % "\n- ".join("%s = %s" % p for p in params.items()))
        self._metadata['algorithm']['parameters'] = params
        self._save()
    
    def visualize(self, export=False, output_dir=".", **kw):
        """ Plot the model for visualization. """
        if len(self.pipeline.steps) == 0:
            self.logger.warning("Model shall be trained before visualizing")
            return
        a = self._metadata['algorithm']
        viz = VISUALIZATIONS.get(a['name']).get(["text", "image"][export])
        if viz is None:
            self.logger.warning("Visualization not available for this algorithm%s" % [" in text mode", ""][export])
            return
        params = {'feature_names': sorted(self._features.keys()), 'logger': self.logger}
        # if visualization requires the original data (e.g. kNN), use self._prepare to create self._data/self._target
        if VISUALIZATIONS.get(a['name']).get("data", False):
            kw['data_only'] = True
            if not self._prepare(**kw):
                return
            params.update({'data': self._data, 'target': self._target, 'algo_params': a['parameters']})
        if export:
            destination = str(Path(output_dir).joinpath("%s.png" % self.name))
            viz(self.classifier, **params).savefig(destination, format="png", bbox_inches="tight")
            self.logger.warning("Visualization saved to %s" % destination)
        else:
            print(viz(self.classifier, **params))
    
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
    
    @classmethod
    def count(cls):
        return sum(1 for _ in Path(config['models']).listdir(Model.check))
    
    @classmethod
    def iteritems(cls, instantiate=False):
        for model in Path(config['models']).listdir(Model.check):
            yield Model(model) if instantiate else model
    
    @staticmethod
    def check(folder):
        try:
            Model.validate(folder)
            return True
        except (TypeError, ValueError):
            return False
    
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
    def __init__(self, name=None, is_pipeline=False, **kw):
        global logger
        logger = self.logger
        obj, self.pipeline = joblib.load(str(name)), DebugPipeline()
        if not is_pipeline:
            self.pipeline.append(StandardScaler())
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
            self._performance = pd.DataFrame(columns=["Model"] + list(PERF_HEADERS.keys()))
    
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

