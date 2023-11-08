# -*- coding: UTF-8 -*-
from tinyscript.helpers.data.types import folder_exists, json_config, pos_int


__all__ = ["add_argument", "expand_parameters", "legend_location", "percentage", "set_yaml"]


def add_argument(parser, name, **kwargs):
    """ Set a standard argument for the given parser. """
    from inspect import currentframe
    glob = currentframe().f_back.f_globals
    check = glob.get('check', lambda *a, **kw: True)
    if name == "dsname":
        def dataset_exists(string):
            if string == "all" or "*" in string or check(string) or kwargs.get('force', False):
                return string
            raise ValueError("Invalid dataset")
        note = kwargs.get('note', None)
        parser.add_argument(kwargs.get('argname', "name"), type=dataset_exists,
                            help=kwargs.get('help', "name of the dataset"),
                            **({} if note is None else {'note': note}))
    elif name == "folder":
        parser.add_argument("filename", help="binary to be represented ; format is regex")
        parser.add_argument("folder", type=folder_exists, help="target folder")
        if kwargs.get('alias', False):
            parser.add_argument("-a", "--alias", type=json_config, help="input label alias in JSON format")
        if kwargs.get('fmt', False):
            parser.add_argument("-f", "--format", default="png", choices=IMG_FORMATS, help="image format")
        if kwargs.get('extended', True):
            parser.add_argument("-l", "--label", nargs="*", action="extend", help="select specific label (keeps order)")
            parser.add_argument("-m", "--max-not-matching", type=pos_int, help="maximum number of labels not matching")
    elif name == "format":
        parser.add_argument("-f", "--format", default="png", choices=IMG_FORMATS, help="image file format for plotting")
    elif name == "mdname":
        def model_exists(string):
            if string == "all" or "*" in string or check(string) or kwargs.get('force', False):
                return string
            raise ValueError("Invalid model")
        a = ("-n", "--name", ) if kwargs.get('optional', False) else ("name", )
        kw = {'type': model_exists, 'help': kwargs.get('help', "name of the model")}
        parser.add_argument(*a, **kw)
    elif name == "query":
        parser.add_argument("-q", "--query", default=kwargs.get('default', "all"),
                            help="query for filtering records to be selected",
                            note="see <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html>")
    elif name == "xpname":
        def experiment_exists(string):
            if string == "all" or "*" in string or check(string) or kwargs.get('force', False):
                return string
            raise ValueError("Invalid experiment")
        note = kwargs.get('note', None)
        parser.add_argument("name", type=experiment_exists, help=kwargs.get('help', "name of the experiment"),
                            **({} if note is None else {'note': note}))
    else:
        raise ValueError("Argument '%s' not defined" % name)
    return parser


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


def legend_location(string):
    ver, hor = string.split("-", 1)
    if ver not in ["lower", "center", "upper"] or hor not in ["left", "center", "right"]:
        raise ValueError(string)
    return string.replace("-", " ")
legend_location.__name__ = "legend location"


def percentage(p):
    p = float(p)
    if 0. <= p <= 100.:
        return p / 100.
    raise ValueError


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
                    glob['logger'].warning("Could not find a class matching '%s'" % k)

