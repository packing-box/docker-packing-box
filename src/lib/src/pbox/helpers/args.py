# -*- coding: UTF-8 -*-
from tinyscript import functools
from tinyscript.helpers.data.types import file_exists, folder_does_not_exist, folder_exists, json_config, pos_int


__all__ = ["add_argument", "characteristic_identifier", "expand_parameters", "figure_options", "item_exists",
           "legend_location", "percentage", "scenario_identifier", "set_yaml", "yaml_file"]


def _fix_args(f):
    @functools.wraps(f)
    def _wrapper(self, value=None):
        if value is None:
            value = self
        return f(self, value)
    return _wrapper


def add_argument(parser, *names, **kwargs):
    """ Set a standard argument for the given parser. """
    opt = kwargs.get('optional', False)
    params = {k: kwargs[k] for k in ["nargs", "note"] if kwargs.get(k) is not None}
    for name in names:
        if name == "aggregate":
            parser.add_argument("-a", "--aggregate", help="pattern to aggregate some of the features together",
                                default="byte_[0-9]+_after_ep")
        elif name == "alteration":
            a = ("-a", "--alteration", ) if opt else ("alteration", )
            kw = {'action': "extend", 'nargs': "*", 'type': alteration_identifier, 'help': "alteration identifiers"}
            parser.add_argument(*a, **kw)
        elif name == "alterations-set":
            parser.add_argument("-a", "--alterations-set", metavar="YAML", default=str(config['alterations']),
                                type=file_exists, help="alterations set's YAML definition")
        elif name == "binary":
            parser.add_argument("-b", "--binary", dest="multiclass", action="store_false",
                                help="process features using binary classification (1:True/0:False/-1:Unlabelled)")
        elif name == "detect":
            parser.add_argument("-d", "--detect", action="store_true",
                                help="detect used packer with the superdetector",
                                note="type '?' to see installed detectors that vote as a superdetector")
        elif name == "datasets":
            parser.add_argument("-d", "--datasets", action="extend", nargs="*", help="datasets to compare features to",
                                required=True)
        elif name == "dsname":
            parser.add_argument(kwargs.get('argname', "name"), type=dataset_exists(kwargs.get('force', False)),
                                help=kwargs.get('help', "name of the dataset"), **params)
        elif name == "dsname2":
            parser.add_argument("name2", type=folder_does_not_exist, help="new name of the dataset")
        elif name == "executable":
            if kwargs.get('single', False):
                parser.add_argument("executable", help="executable file", **params)
            else:
                parser.add_argument("executable", help="executable or folder containing executables or dataset or data"
                                                       " CSV file", **params)
        elif name == "feature":
            parser.add_argument("feature", action="extend", nargs="*", type=feature_identifier,
                                help="feature identifiers")
        elif name == "features-set":
            parser.add_argument("-f", "--features-set", metavar="YAML", type=yaml_file,
                                default=str(config['features']), help="features set's YAML definition", **params)
        elif name == "folder":
            parser.add_argument("filename", help="binary to be represented ; format is regex")
            parser.add_argument("folder", type=folder_exists, help="target folder")
            if kwargs.get('alias', False):
                parser.add_argument("-a", "--alias", type=json_config, help="input label alias in JSON format")
            if kwargs.get('fmt', False):
                parser.add_argument("-f", "--format", default="png", choices=IMG_FORMATS, help="image format")
            if kwargs.get('extended', True):
                parser.add_argument("-l", "--label", nargs="*", action="extend",
                                    help="select specific label (keeps order)")
                parser.add_argument("-m", "--max-not-matching", type=pos_int,
                                    help="maximum number of labels not matching")
        elif name == "ignore-labels":
            parser.add_argument("--ignore-labels", action="store_true",
                                help="while computing metrics, only consider those not requiring labels")
        elif name == "labels":
            parser.add_argument("-l", "--labels", type=json_config, help="set labels from a JSON file")
        elif name == "max-features":
            parser.add_argument("-n", "--max-features", default=0, type=pos_int,
                                help=f"plot n features with {kwargs['max_feats_with']}", note="0 means no limit")
        elif name == "mdname":
            a = ("-n", "--name", ) if opt else ("name", )
            kw = {'type': model_exists(kwargs.get('force', False)), 'help': kwargs.get('help', "name of the model")}
            parser.add_argument(*a, **kw)
        elif name == "multiclass":
            parser.add_argument("-m", "--multiclass", action="store_true", help="process features using true labels",
                                note="if False, means binary classification (1:True/0:False/-1:Unlabelled)")
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
            parser.add_argument("-a", "--reduction-algorithm", default="PCA", choices=("ICA", "PCA"),
                                help="dimensionality reduction algorithm")
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
    if name not in Features.names():
        raise ValueError("Not a valid feature")
    return name


def figure_options(parser):
    group = parser.add_argument_group("figure options", before="extra arguments")
    for option, params in config._defaults['visualization'].items():
        kw = {}
        if params[1] != "BOOL":
            kw['metavar'] = params[1]
            if len(params) > 3:
                kw['type'] = _fix_args(params[3])
        else:
            kw['action'] = "store_true"
        group.add_argument(f"--{option.replace('_', '-')}", default=config[option], help=params[2], **kw)
    return group


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
                setattr(glob[name], "source", v)
            else:
                name = name[:-1]  # strip 's' at the end ; e.g. Algorithms => Algorithm
                if name in glob:
                    setattr(glob[name], "source", v)
                else:
                    glob['logger'].warning(f"Could not find a class matching '{k}'")


def yaml_file(path):
    from .items import load_yaml_config
    try:
        load_yaml_config(path)
    except Exception as e:
        from tinyscript import logging
        logging.getLogger("main").error(e)
        raise ValueError("bad YAML")
    return path
yaml_file.__name__ = "YAML file"

