# -*- coding: UTF-8 -*-
from tinyscript import functools, inspect, itertools, logging, re
from tinyscript.helpers import classproperty, Path


__all__ = ["Entity"]


class Entity:
    """ This class implements some base functionalities for abstractions based on folders
         (i.e Dataset, Experiment, Model). """
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
    
    def exists(self):
        """ Dummy exists method. """
        return self.path.exists()
    
    def is_empty(self):
        """ Check if this instance has no data. """
        return len(self) == 0
    
    def is_valid(self):
        """ Check if this instance has a valid structure. """
        return self.__class__.check(self.path)
    
    @property
    def basename(self):
        """ Dummy shortcut for entity's path.basename. """
        return self.path.basename
    
    @property
    def name(self):
        """ Dummy alias to entity's basename. """
        return self.basename
    
    @classmethod
    def _purge(self, **kw):
        """ Entity-specific purge method, to be called by Entity.purge. """
        raise NotImplementedError
    
    @classmethod
    def check(cls, folder, **kw):
        try:
            cls.validate(folder)
            return True
        except (TypeError, ValueError):
            return False
    
    @classmethod
    def count(cls):
        """ Count existing entity's occurrences. """
        return sum(1 for _ in Path(config['%ss' % cls.__name__.lower()]).listdir(cls.check))
    
    @classmethod
    def iteritems(cls, instantiate=False):
        """ Iterate over entity's occurrences, instantiating them if required. """
        for path in Path(config['%ss' % cls.__name__.lower()]).listdir(lambda f: cls.check(f)):
            yield cls.load(path) if instantiate else path
    
    @classmethod
    def load(cls, folder, **kw):
        """ Validate the target folder and instantiate the right entity class. """
        from _pickle import UnpicklingError
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
        # entities can be instantiated while not having created and validated yet ; load=True ensures that, depending on
        #  the entity, additional checks and settings were made
        kw['load'] = True
        for c in classes:
            try:
                return c(c.validate(folder, **kw), **kw)
            except (ValueError, UnpicklingError):
                pass
        raise ValueError("%s is not a valid %s" % (folder, classes[0].__name__.lower()))
    
    @classmethod
    def purge(cls, name=None, **kw):
        """ Purge all items designated by 'name' for the given class 'cls'. """
        purged = False
        if name == "all":
            for obj in cls.iteritems(True):
                obj._purge()
                purged = True
        elif "*" in (name or ""):
            name = r"^%s$" % name.replace("*", "(.*)")
            for path in cls.iteritems(False):
                if re.search(name, path.stem):
                    cls.load(path)._purge()
                    purged = True
        else:
            obj = cls.load(name)
            obj._purge()
            getattr(obj, "close", lambda: 0)()
            purged = True
        return purged
    
    @classmethod
    def validate(cls, folder, **kw):
        """ Validation method, custom for each entity. """
        raise NotImplementedError
    
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

