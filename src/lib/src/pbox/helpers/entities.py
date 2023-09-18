# -*- coding: UTF-8 -*-
from abc import abstractmethod, ABC
from tinyscript import inspect, itertools, re
from tinyscript.helpers import Path


__all__ = ["AbstractEntity"]


class AbstractEntity(ABC):
    """ This class implements some base functionalities for abstractions based on folders
         (i.e Dataset, Experiment, Model). """
    @abstractmethod
    def __len__(self):
        """ Custom entity's length. """
        pass
    
    def __repr__(self):
        """ Custom entity's string representation. """
        t = " ".join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', self.__class__.__name__)).lower()
        return "<%s %s at 0x%x>" % (self.name, t, id(self))
    
    def __str__(self):
        """ Custom entity's string. """
        return self.name
    
    @property
    def basename(self):
        """ Dummy shortcut for entity's path.basename. """
        return self.path.basename
    
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
    
    def exists(self):
        """ Dummy exists method. """
        return self.path.exists()
    
    def is_empty(self):
        """ Check if this instance has no data. """
        return len(self) == 0
    
    def is_valid(self):
        """ Check if this instance has a valid structure. """
        return self.__class__.check(self.path)
    
    @classmethod
    def iteritems(cls, instantiate=False):
        """ Iterate over entity's occurrences, instantiating them if required. """
        for path in Path(config['%ss' % cls.__name__.lower()]).listdir(lambda f: cls.check(f)):
            yield cls.load(path) if instantiate else path
    
    @classmethod
    def load(cls, folder, **kw):
        """ Load the target folder with the right class. """
        classes, current_cls = [], cls
        # use cases:
        #  (1) BaseModel > [Model, DumpedModel]
        #  (2) Dataset > FilessDataset
        # first, find the highest entity class in inheritance path (i.e. Dataset or BaseModel)
        while hasattr(current_cls, "__base__"):
            if current_cls.__base__.__name__ == "AbstractEntity":
                break
            current_cls = current_cls.__base__
        # then, parse the current class and its children (in this order of precedence)
        for c in itertools.chain([current_cls], current_cls.__base__.__subclasses__()):
            if not c.__name__.startswith("Base"):
                classes.append(c)
        for c in classes:
            try:
                return c(c.validate(folder, **kw), **kw)
            except ValueError:
                pass
        raise ValueError("%s is not a valid %s" % classes[0].__name__.lower())
    
    @property
    def name(self):
        """ Dummy alias to entity's basename. """
        return self.basename
    
    @classmethod
    @abstractmethod
    def validate(cls, folder, **kw):
        """ Validation method, custom for each entity. """
        pass

