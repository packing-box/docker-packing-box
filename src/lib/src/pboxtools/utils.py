# -*- coding: UTF-8 -*-
import argparse
import configparser
import sys
from os import listdir
from os.path import exists, expanduser, isdir, isfile, join, splitext


CONFIGS  = ["algorithms.yml", "analyzers.yml", "detectors.yml", "features.yml", "modifiers.yml", "packers.yml",
            "unpackers.yml"]
DEFAULTS = {
    'workspace':   expanduser("~/.packing-box"),
    'experiments': "/mnt/share/experiments"
}
DSFILES  = ["data.csv", "metadata.json"]
EXPFILES = ["commands.rc", "README.md", "conf", "datasets", "models", "scripts"]
MDFILES  = ["dump.joblib", "features.json", "metadata.json", "performance.csv"]


def _parse_config():
    """ Helper function for parsing ~/.packing-box.conf """
    cfg, d = configparser.ConfigParser(), {'experiment': None}
    try:
        cfg.read_file(expanduser("~/.packing-box.conf"))
        d['workspace'] = cfg['main'].get('workspace', DEFAULTS['workspace'])
        d['experiments'] = cfg['main'].get('experiments', DEFAULTS['experiments'])
    except (configparser.MissingSectionHeaderError, KeyError):
        d['workspace'] = DEFAULTS['workspace']
        d['experiments'] = DEFAULTS['experiments']
    exp_env = expanduser("~/.packing-box/experiment.env")
    if exists(exp_env):
        with open(exp_env) as f:
            d['experiment'] = f.read().strip()
    return d


def dslst():
    """ Main function for listing datasets from the current workspace """
    cfg = _parse_config()
    root, l = join(cfg['experiment'] or cfg['workspace'], "datasets"), []
    if not exists(root):
        return 0
    for f in listdir(root):
        ds = join(root, f)
        if isdir(ds) and all(isfile(join(ds, fn)) for fn in DSFILES) and \
           (isdir(join(ds, "files")) or isfile(join(ds, "features.json"))) and \
           not any(fn not in DSFILES + ["files", "features.json"] for fn in listdir(ds)):
            l.append(f)
    if len(l) > 0:
        print(" ".join(l))
    return 0


def dsflst():
    """ Main function for listing datasets with files from the current workspace """
    cfg = _parse_config()
    root, l = join(cfg['experiment'] or cfg['workspace'], "datasets"), []
    if not exists(root):
        return 0
    for f in listdir(root):
        ds = join(root, f)
        if isdir(ds) and all(isfile(join(ds, fn)) for fn in DSFILES) and isdir(join(ds, "files")) and \
           not any(fn not in DSFILES + ["files"] for fn in listdir(ds)):
            l.append(f)
    if len(l) > 0:
        print(" ".join(l))
    return 0


def mdlst():
    """ Main function for listing models from the current workspace """
    cfg = _parse_config()
    root, l = join(cfg['experiment'] or cfg['workspace'], "models"), []
    if not exists(root):
        return 0
    for f in listdir(root):
        md = join(root, f)
        if isdir(md) and all(isfile(join(md, fn)) for fn in MDFILES):
            l.append(f)
    if len(l) > 0:
        print(" ".join(l))
    return 0


def xpcfglst():
    """ Main function for listing available config file in an experiment """
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    root = join(_parse_config()['experiments'], args.experiment, "conf")
    l = [splitext(f)[0] for f in listdir(root)]
    if len(l) > 0:
        print(" ".join(l))
    return 0


def xplst():
    """ Main function for listing experiments from the current workspace """
    root, l = _parse_config()['experiments'], []
    for f in listdir(root):
        xp = join(root, f)
        if isdir(xp) and all(isdir(join(xp, fn)) for fn in ["conf", "datasets", "models"]) and \
           not any(fn not in EXPFILES for fn in listdir(xp)) and \
           not any(fn not in CONFIGS for fn in listdir(join(xp, "conf"))):
            l.append(f)
    if len(l) > 0:
        print(" ".join(l))
    return 0

