# -*- coding: UTF-8 -*-
from tinyscript import functools, hashlib, inspect, itertools, logging, os, re
from tinyscript.helpers import classproperty, confirm, set_exception, Path

set_exception("StructuralError", "ValueError")


__all__ = ["Entity"]

_CACHE = {}


class Entity:
    """ This class implements some base functionalities for abstractions based on folders
         (i.e Dataset, Experiment, Model). """
    def __new__(cls, name=None, load=True, name_check=True, **kwargs):
        self = super(Entity, cls).__new__(cls)
        self._loaded = False
        if load and name == "ALL":
            self.logger.warning(f"{self.entity.capitalize()} cannot be named '{name}' (reserved word)")
            return
        name = name or kwargs.get('folder')
        self.path = None if name is None else cls.path(name)
        if self.path and name_check:
            config.check(self.name)
        _CACHE.setdefault(cls.entity, {})
        h = int.from_bytes(hashlib.md5(str(self.path).encode()).digest(), "little")
        if h in _CACHE[cls.entity]:
            self = _CACHE[cls.entity][h]
        else:
            _CACHE[cls.entity][h] = self
        cls.logger.debug(f"creating {cls.__name__}({self.name})...")
        for attr, default in getattr(self, "DEFAULTS", {}).items():
            setattr(self, attr, default)
        if not load or self._loaded:
            return self
        if load and hasattr(self, "_load"):
            if self.path:
                self.path.mkdir(exist_ok=True)
            self._load()
            self._loaded = True
        return self
    
    def __len__(self):
        """ Custom entity's length. """
        raise NotImplementedError
    
    def __repr__(self):
        """ Custom entity's string representation. """
        t = " ".join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', self.__class__.__name__)).lower()
        return "<%s %s at 0x%x>" % (self.name, t, id(self))
    
    def __str__(self):
        """ Custom entity's string. """
        return self.name
    
    def _copy(self, destination, overwrite=None, name_check=True):
        """ Copy the current entity to a given destination. """
        cls = self.__class__
        dst = cls.path(destination)
        if self.path.absolute() != dst.absolute() and \
           not dst.exists() or cls.confirm_overwrite(dst, overwrite, name_check):
            self.logger.debug(f"copying {cls.entity} '{self.basename}' to {dst}...")
            self.path.copy(dst)
    
    def _purge(self, **kw):
        """ Purge the current entity. """
        if self.path:
            self.logger.debug(f"purging {self.__class__.entity} '{self.basename}'...")
            self.path.remove(error=False)
    
    def _remove(self):
        """ Remove the current entity. """
        self.logger.debug(f"removing {self.__class__.entity} '{self.basename}'...")
        self.path.remove(error=False)
    
    def exists(self):
        """ Dummy exists method. """
        return self.path.exists()
    
    def is_empty(self):
        """ Check if this instance has no data. """
        return len(self) == 0
    
    def is_valid(self):
        """ Check if this instance has a valid structure. """
        return self.__class__.check(self.path)
    
    def rename(self, name2=None, overwrite=None, **kw):
        """ Rename the current entity. """
        l, cls = self.logger, self.__class__
        path2 = config[f'{cls.entity}s'].joinpath(name2)
        if not path2.exists() or cls.confirm_overwrite(path2, overwrite):
            l.debug(f"renaming {cls.entity} '{self.basename}' to '{name2}'...")
            self.path.rename(path2)
            self.path = path2
            return True
        else:
            l.warning(f"{cls.entity.capitalize()} '{self.basename}' already exists")
            return False
    
    @property
    def basename(self):
        """ Dummy shortcut for entity's path.basename. """
        try:
            return self.path.basename
        except AttributeError:
            pass
    
    @property
    def name(self):
        """ Dummy alias to entity's basename. """
        return self.basename
    
    @classmethod
    def check(cls, folder, **kw):
        try:
            cls.validate(folder, **kw)
            return True
        except (TypeError, ValueError):
            return False
    
    @classmethod
    def check_entity(cls, folder, **kw):
        return any(c.check(folder) or c.has_empty_structure(folder) for c in cls.entity_classes)
    
    @classmethod
    def confirm_overwrite(cls, folder, overwrite=None, name_check=True):
        classes = cls.entity_classes
        instance = folder if isinstance(folder, tuple(classes)) else cls(folder, load=False, name_check=name_check)
        if any(c.has_empty_structure(folder) for c in classes):
            # do not just overwrite, remove it (e.g. with an experiment, there may be optional folders left behind)
            instance.path.remove()
            overwrite = True
        if instance.path.exists():
            if overwrite is None:
                overwrite = confirm(f"Are you sure you want to overwrite '{instance.path}' ?")
            if overwrite:
                instance._purge() if any(c.check(folder) for c in classes) else instance.path.remove()
            else:
                return False
        instance._load()
        return True
    
    @classmethod
    def count(cls):
        """ Count existing entity's occurrences from the current workspace. """
        return sum(1 for _ in Path(config[f'{cls.entity}s']).listdir(cls.check_entity))
    
    @classmethod
    def has_empty_structure(cls, folder):
        """ Check if the folder structure is compliant but has no file. """
        return not cls.check(folder, strict=True) and cls.check(folder, strict=True, empty=True)
    
    @classmethod
    def iteritems(cls, instantiate=True, load=False):
        """ Iterate over entity's occurrences, instantiating them if required. """
        for path in Path(config[f'{cls.entity}s']).listdir(cls.check_entity):
            yield cls(path, load=load) if instantiate else path
    
    @classmethod
    def load(cls, folder, **kw):
        """ Validate the target folder and instantiate the right entity class. """
        from _pickle import UnpicklingError
        # entities can be instantiated while not having created and validated yet ; load=True ensures that, depending on
        #  the entity, additional checks and settings were made
        kw['load'] = True
        for c in cls.entity_classes:
            try:
                return c(c.validate(folder, **kw), **kw)
            except (ValueError, UnpicklingError):
                # UnpicklingError occurs when loading an entity that contains items based on a YAML configuration that
                #  has fields loading pickled objects
                pass
        raise ValueError(f"{folder} is not a valid {cls.entity}")
    
    @classmethod
    def path(cls, folder):
        """ Compute a working path from a given folder according to the following convention:
            - If 'folder' has no OS separator, it is to be considered as an entity from the workspace
            - If 'folder' has OS separator(s), it will be handled as a common entity (either from the current workspace
               or from outside this)
        As a consequence, when targetting a local entity, e.g. 'my-test-dataset', it shall be prefixed with './'.
        """
        return Path(folder, expand=True).absolute() if os.sep in str(folder) else \
               config[f"{cls.entity}s"].joinpath(folder)
    
    @classmethod
    def purge(cls, name=None, **kw):
        """ Purge all items designated by 'name' for the given class 'cls'. """
        purged = False
        def _iter(n):
            # purge everything
            if n == "all":
                for obj in cls.iteritems():
                    yield obj
            # purge a bunch of datasets based on a wildcard
            elif "*" in (n or ""):
                n = r"^%s$" % n.replace("*", "(.*)")
                for obj in cls.iteritems():
                    yield obj
            # purge a single target dataset
            else:
                yield cls(n, load=False)
        for obj in _iter(name):
            getattr(obj, "close", lambda: 0)()
            obj._purge()
            purged = True
        if not purged:
            cls.logger.warning(f"No {cls.entity} to purge in workspace ({config[f'{cls.entity}s']})" if name == "all" \
                               else f"No {cls.entity} matched the given name")
    
    @classmethod
    def validate(cls, item, strict=False, empty=False, **kw):
        """ Validate entity's structure according to its STRUCTURE class attribute. This validates either a folder if
             class' STRUCTURE is a list (e.g. Dataset) or a single file if it is a string (e.g. DumpedModel). """
        p = cls.path(item)
        if not p.exists():
            raise StructuralError(f"'{p}' does not exist")
        if p.extension == "" and not p.is_dir():
            raise StructuralError(f"'{p}' is not a folder")
        if p.extension != "" and not p.is_file():
            raise StructuralError(f"'{p}' is not a file")
        structure = kw.get('structure', getattr(cls, "STRUCTURE", None))
        if structure:
            if isinstance(structure, list):
                structural, regex = [], re.compile(r"^(-?)(.*?)(\*?)$")
                for item in structure:
                    pres, item, opt = regex.search(item).groups()
                    present, optional, itype = pres == "", opt == "*", ["folder", "file"]["." in item]
                    if empty and itype == "file":
                        continue  # skip validation of files if only considering the empty structure
                    structural.append(item)
                    if present and not optional or strict:
                        if not p.joinpath(item).exists():
                            raise StructuralError(f"'{p}' has {itype} '{item}' missing")
                        if not getattr(p.joinpath(item), ["is_dir", "is_file"][itype == "file"])():
                            raise StructuralError(f"'{p}' has '{item}' that is not a {itype}")
                    elif not present and p.joinpath(item).exists():
                        raise StructuralError(f"'{p}' has '{item}' {itype} while it shall not")
                # determine if folder structure is empty
                has_no_file = True
                if empty:
                    has_no_file = len(list(p.listdir(Path.is_file))) == 0
                    if has_no_file:
                        for sp in p.listdir(Path.is_dir):
                            if len(list(sp.listdir())) > 0:
                                has_no_file = False
                                break
                # geenrate warnings if not considering empty structure
                if not has_no_file:
                    for check, itype in zip([Path.is_dir, Path.is_file], ["folder", "file"]):
                        for f in p.listdir(check):
                            if f not in structural:
                                msg = f"'{p}' has unknown {itype} '{f}'"
                                if strict:
                                    raise StructuralError(msg)
                                cls.logger.warning(msg)
            elif isinstance(structure, str):
                fn = Path(structure)
                if fn.stem == "*" and p.extension != fn.extension:
                    raise StructuralError(f"'{p}' is not a {fn.extension}")
            else:
                raise ValueError("Class' STRUCTURE attribute should be a list to validate a folder structure (e.g. "
                                 "Dataset) or a string to validate a file (e.g. DumpedModel)")
        return p
    
    @classproperty
    def entity(cls):
        return cls.entity_classes[0].__name__.lower()
    
    @classproperty
    def entity_classes(cls):
        classes, current_cls = [], cls
        # use cases:
        #  (1) BaseModel > [Model, DumpedModel]
        #  (2) Dataset > FilessDataset
        #  (3) Experiment
        # first, find the highest entity class in inheritance path (i.e. Dataset, BaseModel or Experiment)
        while hasattr(current_cls, "__base__"):
            if current_cls.__base__.__name__ == "Entity":
                break
            current_cls = current_cls.__base__
        # then, parse the current class and its children (in this order of precedence)
        for c in itertools.chain([current_cls], current_cls.__subclasses__()):
            if not c.__name__.startswith("Base"):
                classes.append(c)
        return classes
    
    @classproperty
    def instances(cls):
        return [cls(n) for n in cls.names]
    
    @classproperty
    def logger(cls):
        if not hasattr(cls, "_logger"):
            name = cls.__name__.lower()
            cls._logger = logging.getLogger(name)
            logging.setLogger(name)
        return cls._logger
    
    @classproperty
    def names(cls):
        return [p.basename for p in Path(config['%ss' % cls.__name__.lower()]).listdir(cls.check)]

