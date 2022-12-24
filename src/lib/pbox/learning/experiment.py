# -*- coding: UTF-8 -*-
from tinyscript import logging, ts

from .algorithm import *
from .dataset import *
from .features import *
from .model import *
from ..items import *
from ..common.config import *
from ..common.modifiers import *
from ..common.utils import *


__all__ = ["Experiment"]


class Experiment:
    """ Folder structure:
    
    [name]
      +-- conf          custom YAML configuration files
      +-- datasets      datasets specific to the experiment
      +-- models        models specific to the experiment
      +-- (scripts)     additional scripts
      +-- README.md     notes for explaining the experiment
    """
    @logging.bindLogger
    def __init__(self, name="experiment", **kw):
        name = check_name(ts.Path(name).basename)
        self.path = ts.Path(config['experiments'].joinpath(name), create=True).absolute()
        for folder in ["conf", "datasets", "models"]:
            folder = self.path.joinpath(folder)
            if not folder.exists():
                folder.mkdir()
        self.path.joinpath("README.md").touch()
        config['experiment'] = config['workspace'] = self.path
        for conf in self.path.joinpath("conf").listdir():
            config[conf.stem] = conf
    
    def __getitem__(self, name):
        """ Get something from the experiment folder, either a config file, a dataset or a model. """
        # case 1: 'name' matches a reserved word for a YAML configuration file
        for conf in config.DEFAULTS['definitions'].keys():
            if name == conf:
                conf = self.path.joinpath("conf").joinpath(conf + ".conf")
                if not conf.exists():
                    conf = config[conf]
                n = conf.capitalize()
                # if the name exists in the global scope, this will not trigger ; e.g. Features
                if not n in globals():
                    n = n[:-1]  # strip the last character ; e.g. Algorithms => Algorithm (class in global scope)
                cls = globals()[n]
                cls.source = str(conf)
                return cls.registry
        # case 2: 'name' matches a dataset from the experiment
        for ds in self.path.joinpath("datasets").listdir():
            if name == ds.stem:
                return open_dataset(ds)
        # case 3: 'name' matches a model from the experiment
        for ds in self.path.joinpath("models").listdir():
            if name == ds.stem:
                return open_model(ds)
        raise KeyError(name)
    
    def close(self, **kw):
        """ Close the currently open experiment. """
        del config['experiment']
    
    def edit(self, **kw):
        """ Edit the README or a YAML configuration file. """
        cfg_file = kw.get('config')
        if cfg_file is None:
            self.logger.debug("editing experiment's README.md...")
            edit_file(self.path.joinpath("README.md").absolute(), text=True, logger=self.logger)
        else:
            p = self.path.joinpath("conf", cfg_file + ".conf")
            if not p.exists():
                cfg = config[cfg_file]
                self.logger.debug("copying configuration file from '%s'..." % cfg)
                cfg.copy(p)
            self.logger.debug("editing experiment's %s configuration..." % cfg_file)
            edit_file(p, text=True, logger=self.logger)
    
    def show(self, **kw):
        """ Show an overview of the experiment. """
        Experiment.summarize(self.path)
    
    @property
    def basename(self):
        """ Dummy shortcut for dataset's path.basename. """
        return self.path.basename
    
    @staticmethod
    def check(folder):
        try:
            Experiment.validate(folder)
            return True
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def summarize(path=None):
        Dataset(load=False).list()
        Model(load=False).list()
    
    @staticmethod
    def validate(folder, warn=False, logger=None):
        f = config['experiments'].joinpath(folder)
        if not f.exists():
            raise ValueError("Folder does not exist")
        if not f.is_dir():
            raise ValueError("Input is not a folder")
        for fn in ["conf", "datasets", "models"]:
            if not f.joinpath(fn).exists():
                raise ValueError("Folder does not have %s" % fn)
        for cfg in f.joinpath("conf").listdir():
            if cfg not in config.DEFAULTS['definitions'].keys():
                raise ValueError("Unknown configuration file '%s'" % cfg)
        for fn in f.listdir(Path.is_dir):
            if fn not in ["conf", "datasets", "models", "scripts"] and warn and logger is not None:
                logger.warning("Unknown subfolder '%s'" % fn)
        for fn in f.listdir(Path.is_file):
            if fn not in ["README.md"] and warn and logger is not None:
                logger.warning("Unknown file '%s'" % fn)
        return f

