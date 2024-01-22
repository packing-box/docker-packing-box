# -*- coding: UTF-8 -*-
import argparse
import configparser
import re
import sys
import yaml
from os import listdir
from os.path import exists, expanduser, isdir, isfile, join, splitext


CONFIG   = None
CONFIGS  = ["algorithms.yml", "alterations.yml", "analyzers.yml", "detectors.yml", "features.yml", "packers.yml",
            "scenarios.yml", "unpackers.yml"]
DEFAULTS = {
    'workspace':   expanduser("~/.packing-box"),
    'experiments': "/mnt/share/experiments"
}
DSFILES  = ["data.csv", "metadata.json"]
EXPFILES = ["commands.rc", "README.md", "conf", "data", "datasets", "figures", "models", "scripts"]
MDFILES  = ["dump.joblib", "features.json", "metadata.json", "performance.csv"]

_fmt_name = lambda x: (x or "").lower().replace("_", "-")


def __output(l, ret=False):
    l.sort()
    if ret:
        return l
    if len(l) > 0:
        print(" ".join(l))
    return 0


def __parse_config():
    """ Helper function for parsing ~/.packing-box.conf """
    global CONFIG
    if CONFIG is None:
        cfg, CONFIG = configparser.ConfigParser(), {'experiment': None}
        try:
            cfg.read_file(expanduser("~/.packing-box.conf"))
            CONFIG['workspace'] = cfg['main'].get('workspace', DEFAULTS['workspace'])
            CONFIG['experiments'] = cfg['main'].get('experiments', DEFAULTS['experiments'])
        except (configparser.MissingSectionHeaderError, KeyError):
            CONFIG['workspace'] = DEFAULTS['workspace']
            CONFIG['experiments'] = DEFAULTS['experiments']
        exp_env = expanduser("~/.packing-box/experiment.env")
        if exists(exp_env):
            with open(exp_env) as f:
                CONFIG['experiment'] = f.read().strip()
    return CONFIG


def _configfile(cfgfile):
    def _wrapper(f):
        def _subwrapper(return_list=False):
            """ Decorator for listing something from the current workspace """
            cfg, parts = __parse_config(), ("conf", f"{cfgfile}.yml")
            path = join(cfg['experiment'] or cfg['workspace'], *parts)
            if not exists(path):
                path = join(cfg['workspace'], *parts)
            cfg, paths = {}, [join(path, f) for f in listdir(path) if splitext(f)[1] == ".yml"] if isdir(path) else \
                             [path]
            for path in paths:
                with open(path) as fp:
                    yaml_str = "\n".join(l for l in fp.readlines() if len(l.split(":")) > 1 and \
                                                                  not re.match(r"\!{1,2}", l.split(":", 1)[1].lstrip()))
                cfg.update(yaml.safe_load(yaml_str.replace("!!python", "")) or {})
            return __output(f(cfg), return_list)
        return _subwrapper
    return _wrapper


def _workspace(folder):
    def _wrapper(f):
        def _subwrapper(return_list=False):
            """ Decorator for listing something from the current workspace """
            cfg = __parse_config()
            root, l = join(cfg['experiment'] or cfg['workspace'], folder), []
            if not exists(root):
                return [] if return_list else 0
            for fp in listdir(root):
                if f(join(root, fp)):
                    l.append(fp)
            return __output(l, return_list)
        return _subwrapper
    return _wrapper


for item in ["alterations", "features", "scenarios"]:
    f1 = _configfile(item)(lambda cfg: sorted(list(x for x in cfg.keys() if not x.startswith("abstract_"))))
    f1.__doc__ = " List all %s available in the current workspace. " % item
    globals()['list_all_%s' % item] = f1
    f2 = _configfile(item)(lambda cfg: sorted([x for x, data in cfg.items() if not x.startswith("abstract_") and \
                                                            data.get('keep' if item == "features" else 'apply', True)]))
    f2.__doc__ = " List enabled %s available in the current workspace. " % item
    globals()['list_enabled_%s' % item] = f2


for item in ["analyzers", "detectors", "packers", "unpackers"]:
    f1 = _configfile(item)(lambda cfg: sorted(list(_fmt_name(x) for x in cfg.keys() if x != "defaults")))
    f1.__doc__ = " List all %s available in the current workspace. " % item
    globals()['list_all_%s' % item] = f1
    f2 = _configfile(item)(lambda cfg: sorted([_fmt_name(x) for x, data in cfg.items() if data.get('status') == "ok" \
                                                                                          and x != "defaults"]))
    f2.__doc__ = " List working %s available in the current workspace. " % item
    globals()['list_working_%s' % item] = f2


@_configfile("algorithms")
def list_all_algorithms(cfg):
    """ Main function for listing all analyzers available in the current workspace """
    l = []
    for section in ["Semi-Supervised", "Supervised", "Unsupervised"]:
        l.extend(list(cfg.get(section, {}).keys()))
    return sorted(list(set(_fmt_name(x) for x in l if x != "defaults")))


def list_config_keys():
    import pbox
    l = ["--" + opt.replace("_", "-") for opt, _ in config.items()]
    if len(l) > 0:
        print(" ".join(l))
    return 0


@_workspace("datasets")
def list_datasets(ds):
    """ Condition for listing datasets from the current workspace """
    return isdir(ds) and all(isfile(join(ds, fn)) for fn in DSFILES) and \
           (isdir(join(ds, "files")) or isfile(join(ds, "features.json"))) and \
           not any(fn not in DSFILES + ["alterations.json", "files", "features.json"] for fn in listdir(ds))


@_workspace("datasets")
def list_datasets_with_files(ds):
    """ Condition for listing datasets from the current workspace """
    return isdir(ds) and all(isfile(join(ds, fn)) for fn in DSFILES) and isdir(join(ds, "files")) and \
           not any(fn not in DSFILES + ["alterations.json", "files"] for fn in listdir(ds))


@_workspace("models")
def list_models(md):
    """ Condition for listing models from the current workspace """
    return isdir(md) and all(isfile(join(md, fn)) for fn in MDFILES)


def list_experiment_configs(return_list=False):
    """ Main function for listing available config file in an experiment """
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    xp = join(__parse_config()['experiments'], args.experiment)
    root = join(xp, "conf")
    l = [splitext(f)[0] for f in listdir(root)]
    if exists(join(xp, "commands.rc")):
        l.append("commands")
    return __output(l, return_list)


def list_experiments(return_list=False):
    """ Main function for listing experiments from the current workspace """
    root, l = __parse_config()['experiments'], []
    for f in listdir(root):
        xp = join(root, f)
        if isdir(xp) and all(isdir(join(xp, fn)) for fn in ["conf", "datasets", "models"]) and \
           not any(fn not in EXPFILES for fn in listdir(xp)) and \
           not any(fn not in CONFIGS for fn in listdir(join(xp, "conf"))):
            l.append(f)
    return __output(l, return_list)


def list_tools(return_list=False):
    """ Main function for listing tools """
    return __output([f for f in listdir(expanduser("~/.opt/tools")) if f not in ["?", "startup"]], return_list)

