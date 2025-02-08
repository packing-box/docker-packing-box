# -*- coding: UTF-8 -*-
from tinyscript import functools, inspect, re
from tinyscript.helpers.data.types import *

from .files import Locator
from .formats import expand_formats


__all__ = ["add_argument", "characteristic_identifier", "expand_parameters", "figure_options", "filter_args",
           "item_exists", "legend_location", "percentage", "scenario_identifier", "set_yaml", "yaml_config"]


def _fix_args(f):
    @functools.wraps(f)
    def _wrapper(self, value=None):
        if value is None:
            value = self
        return f(self, value)
    return _wrapper


def add_argument(parser, *names, **kwargs):
    """ Set a standard argument for the given parser. """
    params = {k: kwargs[k] for k in ["nargs", "note"] if kwargs.get(k) is not None}
    for name in names:
        if name == "aggregate":
            parser.add_argument("-a", "--aggregate", help="pattern to aggregate some of the features together",
                                default="byte_[0-9]+_after_ep")
        elif name == "alteration":
            parser.add_argument("-A", "--alteration", action="extend", nargs="*", type=alteration_identifier,
                                help="alteration identifiers")
        elif name == "alterations-set":
            parser.add_argument("-a", "--alterations-set", metavar="YAML", default=str(config['alterations']),
                                type=file_exists, help="alterations set's YAML definition")
        elif name == "binary":
            parser.add_argument("-b", "--binary", dest="multiclass", action="store_false",
                                help="process features using binary classification (1:True/0:False/-1:Unlabelled)")
        elif name == "detect":
            parser.add_argument("-d", "--detect", action="store_true", help="detect used packer with the superdetector",
                                note="type '?' to see installed detectors that vote as a superdetector")
        elif name == "datasets":
            parser.add_argument("-d", "--datasets", action="extend", nargs="*", help="datasets to compare features to",
                                required=True)
        elif name == "dsname":
            parser.add_argument(kwargs.get('argname', "name"), type=dataset_exists(kwargs.get('force', False)),
                                help=kwargs.get('help', "name of the dataset"), **params)
        elif name == "dsname2":
            parser.add_argument("name2", type=folder_does_not_exist, help="name of the new dataset")
        elif name == "executable":
            if kwargs.get('single', False):
                parser.add_argument("executable", help="executable file", **params)
            else:
                parser.add_argument("executable", help="executable or folder containing executables or dataset or data"
                                                       " CSV file", **params)
        elif name == "exeformat":
            parser.add_argument("--format", default="PE32", type=exe_format, help="executable format to be considered")
        elif name == "expformat":
            parser.add_argument("-e", "--export", dest="output", default="csv", choices=EXPORT_FORMATS.keys(),
                                help="file format the data should be exported to")
        elif name == "feature":
            parser.add_argument("feature", action="extend", nargs="*", type=feature_identifier,
                                help="feature identifiers")
        elif name == "features-set":
            parser.add_argument("-f", "--features-set", metavar="YAML", type=yaml_config,
                                default=str(config['features']), help="features set's YAML definition", **params)
        elif name == "folder":
            parser.add_argument("filename", help="binary to be represented ; format is regex")
            parser.add_argument("folder", type=folder_exists, help="target folder")
            if kwargs.get('alias', False):
                parser.add_argument("-a", "--alias", type=json_config, help="input label alias in JSON format")
            if kwargs.get('fmt', False):
                parser.add_argument("-f", "--img-format", default="png", choices=IMG_FORMATS, help="image format")
            if kwargs.get('extended', True):
                parser.add_argument("-l", "--label", nargs="*", action="extend",
                                    help="select specific label (keeps order)")
                parser.add_argument("-m", "--max-not-matching", type=pos_int,
                                    help="maximum number of labels not matching")
        elif name == "ignore-labels":
            parser.add_argument("--ignore-labels", action="store_true",
               help="while computing metrics, only consider those not requiring labels")
        elif name == "labels":
            parser.add_argument("-l", "--labels", type=file_exists,
                                help="set labels from a JSON file or a CSV data file")
        elif name == "max-features":
            parser.add_argument("-n", "--max-features", default=0, type=pos_int,
               help=f"plot n features with {kwargs['max_feats_with']}", note="0 means no limit")
        elif name == "mdname":
            a = ("-n", "--name", ) if kwargs.get('optional', False) else ("name", )
            kw = {'type': model_exists(kwargs.get('force', False)), 'help': kwargs.get('help', "name of the model")}
            parser.add_argument(*a, **kw)
        elif name == "mi-select":
            parser.add_argument("-M", "--mi-select", action="store_true",
                                help="apply mutual information feature selection")
        elif name == "mi-kbest":
            parser.add_argument("-k", "--mi-kbest", type=pos_float, default=0.7,
                                help="threshold for mutual information feature selection",
                                note="if mi_kbest >= 1, the mi_kbest features with highest MI will be kept ; if within"
                                     " (0.0, 1.0), the mi_kbest percent of features will be kept")
        elif name == "multiclass":
            parser.add_argument("-m", "--multiclass", action="store_true", help="process features using true labels",
                                note="if False, means binary classification (1:True/0:False/-1:Unlabelled)")
        elif name == "n-jobs":
            parser.add_argument("--n-jobs", type=lambda x: pos_int(x, False), help="number of parallel jobs")
        elif name == "number":
            parser.add_argument("-n", "--number", dest=kwargs.get('dest', "limit"), type=pos_int, default=0,
                                help="limit number of executables for the output dataset", note="0 means all")
        elif name == "ncomponents":
            parser.add_argument("-n", "--n-components", metavar="N", default=20, type=int,
                                help="number of components for dimensionality reduction")
        elif name == "perplexity":
            parser.add_argument("-p", "--perplexity", metavar="P", default=30, type=int,
                                help="t-SNE perplexity for dimensionality reduction")
        elif name == "query":
            parser.add_argument("-q", "--query", default=kwargs.get('default', "all"),
                                help="query for filtering records to be selected",
                                note="see <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html>")
        elif name == "reduction-algorithm":
            parser.add_argument("-a", "--reduction-algorithm", metavar="ALGO", default="PCA", choices=("ICA", "PCA"),
                                help="dimensionality reduction algorithm")
        elif name == "title":
            parser.add_argument("-t", "--title", help="title for the figure", note="if not specified, it is generated")
        elif name == "true-class":
            parser.add_argument("-T", "--true-class", metavar="CLASS", help="class to be considered as True")
        elif name == "xpname":
            parser.add_argument("name", type=experiment_exists(kwargs.get('force', False)),
                                help=kwargs.get('help', "name of the experiment"), **params)
        else:
            raise ValueError(f"Argument '{name}' not defined")
    return parser


def alteration_identifier(name):
    from pbox.core.executable import Alterations
    Alterations(None)
    if name not in Alterations.names():
        raise ValueError("Not an alteration identifier")
    return name


def characteristic_identifier(name):
    if name not in ["format", "label", "signature"]:
        try:
            return feature_identifier(name)
        except ValueError:
            raise ValueError("Not a valid characteristic")
    return name


def dataset_exists(force=False):
    def _wrapper(string):
        from pbox.core.dataset import Dataset, FilelessDataset
        if string == "all" or "*" in string or Dataset.check(string) or FilelessDataset.check(string) or force:
            return string
        raise ValueError("Invalid dataset")
    _wrapper.__name__ = "dataset"
    return _wrapper


def exe_format(name):
    if name not in expand_formats("All"):
        raise ValueError("Invalid executable format")
    return name


def expand_parameters(*strings, **kw):
    """ This simple helper expands a [sep]-separated string of keyword-arguments defined with [key]=[value]. """
    from ast import literal_eval
    sep = kw.get('sep', ",")
    d = {}
    for s in strings:
        for p in s.split(sep):
            k, v = p.split("=", 1)
            try:
                v = literal_eval(v)
            except ValueError:
                pass
            d[k] = v
    return d


def experiment_exists(force=False):
    def _wrapper(string):
        from pbox.core.experiment import Experiment
        if string == "all" or "*" in string or Experiment.check(string) or force:
            return string
        raise ValueError("Invalid experiment")
    _wrapper.__name__ = "experiment"
    return _wrapper


def feature_identifier(name):
    from pbox.core.executable import Features
    Features(None)
    names = Features.names()
    if name not in names:
        regex = re.compile(name.replace(".*", "*").replace("*", ".*"))
        for n in names:
            if regex.search(n):
                return name
        raise ValueError("Not a valid feature")
    return name


def figure_options(parser, add_no_title=False):
    group = parser.add_argument_group("figure options", before="extra arguments")
    for option in sorted(list(config._defaults['visualization'].keys()) + [[], ["no-title"]][add_no_title]):
        params = config._defaults['visualization'].get(option)
        if option == "no-title":
            group.add_argument("--no-title", action="store_true", help="do not display the title (default: False)")
            continue
        kw = {}
        if params[1] != "BOOL":
            kw['metavar'] = params[1]
            if len(params) > 3:
                kw['type'] = _fix_args(params[3])
        else:
            kw['action'] = "store_true"
        group.add_argument(f"--{option.replace('_', '-')}", default=config[option], help=params[2], **kw)
    return group


def filter_args(params, target):
    valid = list(inspect.signature(target).parameters)
    return {k: v for k, v in params.items() if k in valid}


def item_exists(string):
    for k in ['workspace', 'datasets', 'models']:
        p = config[k].joinpath(string)
        if p.exists():
            return p
    raise ValueError("Path does not exist")


def legend_location(string):
    ver, hor = string.split("-", 1)
    if ver not in ["lower", "center", "upper"] or hor not in ["left", "center", "right"]:
        raise ValueError(string)
    return string.replace("-", " ")
legend_location.__name__ = "legend location"


def model_exists(force=False):
    def _wrapper(string):
        from pbox.core.experiment import DumpedModel, Model
        check = lambda s: Model.check(s) or DumpedModel.check(s)
        if string == "all" or "*" in string or check(string) or force:
            return string
        raise ValueError("Invalid model")
    _wrapper.__name__ = "model"
    return _wrapper


def percentage(p):
    p = float(p)
    if 0. <= p <= 100.:
        return p / 100.
    raise ValueError


def scenario_identifier(name):
    from pbox.core.experiment import Scenario
    name = name.replace("_", "-")
    if name not in Scenario.names:
        raise ValueError("Not a valid scenario")
    return name


def set_yaml(namespace):
    """ Set the 'source' attribute of the YAML definitions attached to the given namespace. """
    from inspect import currentframe
    glob = currentframe().f_back.f_globals
    for k, v in namespace._get_kwargs():
        if k.endswith("_set"):
            name = k[:-4].capitalize()
            if name in glob:  # e.g. Features
                setattr(glob[name], "source", Locator(f"conf://{v}"))
            else:
                name = name[:-1]  # strip 's' at the end ; e.g. Algorithms => Algorithm
                if name in glob:
                    setattr(glob[name], "source", Locator(f"conf://{v}"))
                else:
                    glob['logger'].warning(f"Could not find a class matching '{k}'")


def yaml_config(path):
    from .items import load_yaml_config
    try:
        load_yaml_config(path)
    except Exception as e:
        from tinyscript import logging
        logging.getLogger("main").error(e)
        raise ValueError("bad YAML")
    return path
yaml_config.__name__ = "YAML configuration"

