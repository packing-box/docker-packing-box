# -*- coding: UTF-8 -*-
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted
from tinyscript.helpers import Path, TempPath
from weka.classifiers import Classifier

from ..common.config import config
from ..common.item import Item, MetaItem


__all__ = ["Algorithm", "WekaClassifier"]


_sanitize_feature_name = lambda n: n.replace("<", "[lt]").replace(">", "[gt]")


class MetaAlgorithm(MetaItem):
    def __getattribute__(self, name):
        # this masks some attributes for child classes (e.g. Algorithm.registry can be accessed, but when the registry
        #  of child classes is computed, the child classes, e.g. RF, won't be able to access RF.registry)
        if name in ["get", "iteritems", "mro", "registry", "source"] and self._instantiable:
            raise AttributeError(name)
        return super(MetaItem, self).__getattribute__(name)

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, path):
        if '%ss' % self.__name__.lower() not in config:
            return
        p = Path(str(path or config['algorithms']))
        if hasattr(self, "_source") and self._source == p:
            return
        cls, self._source = Algorithm, p
        # open the .conf file associated to algorithms
        cls.registry, glob = [], globals()
        with p.open() as f:
            algos = yaml.load(f, Loader=yaml.Loader)
        # start parsing items of cls
        for category, items in algos.items():
            if category not in ["Semi-Supervised", "Supervised", "Unsupervised"]:
                raise ValueError("bad learning algorithm category")
            dflts = items.pop('defaults', {})
            dflts.setdefault('boolean', False)
            dflts.setdefault('multiclass', True)
            dflts.setdefault('parameters', {})
            dflts['labelling'] = {'Supervised': "full", 'Semi-Supervised': "partial", 'Unsupervised': "none"}[category]
            for algo, data in items.items():
                for k, v in dflts.items():
                    if k == "base":
                        raise ValueError("parameter 'base' cannot have a default value")
                    data.setdefault(k, v)
                # put the related algorithm in module's globals()
                if algo not in glob:
                    d = dict(cls.__dict__)
                    for a in ["get", "iteritems", "mro", "registry"]:
                        d.pop(a, None)
                    glob[algo] = type(algo, (cls, ), d)
                i = glob[algo]
                i._instantiable = True
                # now set attributes from YAML parameters
                for k, v in data.items():
                    setattr(i, "_" + k if k == "source" else k, v)
                glob['__all__'].append(algo)
                cls.registry.append(i())


class Algorithm(Item, metaclass=MetaAlgorithm):
    """ Algorithm abstraction. """
    def is_weka(self):
        """ Simple method for checking if the algorithm is based on a Weka class. """
        return self.base.__base__ is WekaClassifier


class WekaClassifier(Classifier):
    """ This class implements a binding for using the Decorate algorithm from the Weka framework the same way as SkLearn
         algorithms. """
    train_file = "/tmp/weka/train.arff"
    test_file  = "/tmp/weka/test.arff"
    
    def __init__(self, **kwargs):
        kwargs = {("-" + k if not k.startswith("-") else k): v for k, v in kwargs.items()}
        super(WekaClassifier, self).__init__(name=self._weka_base, ckargs=kwargs)
    
    def _predict(self, arff, **kwargs):
        check_is_fitted(self, "labels")
        if isinstance(arff, pd.DataFrame):
            arff = self._to_arff(arff, name="test")
        pred, proba = [], []
        for result in super(WekaClassifier, self).predict(str(arff)):
            idx = self.labels.index(result.predicted)
            pred.append(idx)
            pb = [0] * len(self.labels)
            pb[idx] = result.probability
            pb[(idx + 1) % 2] = 1 - result.probability
            proba.append(pb)
        return pd.np.array(pred), pd.np.array(proba)
    
    def fit(self, X, y, **kwargs):
        self.labels = sorted(set(map(str, y.tolist())))
        arff = self._to_arff(X, y, name="train")
        super(WekaClassifier, self).train(str(arff))
    
    def predict(self, X, **kwargs):
        return self._predict(X)[0]
    
    def predict_proba(self, X, **kwargs):
        return self._predict(X, **kwargs)[1]
    
    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    def _to_arff(self, data, target=None, destination=None, name="undefined"):
        """ This generates an ARFF file of the dataset/model, suitable for use with the Weka framework. """
        dest = destination or TempPath(prefix="weka-arff-", length=8).tempfile("data.arff")
        # prepare the destination file, create the base directory
        f = Path(dest)
        Path(f.dirname, create=True)
        # compute the list of features and their data types
        a, mlen, row1 = [], 0, data.iloc[0]
        for column in data:
            mlen = max(mlen, len(_sanitize_feature_name(column)))
        for column in data:
            value = row1[column]
            attr_type = ["string", "numeric"][isinstance(value, int) or isinstance(value, float)]
            a.append(("@ATTRIBUTE {: <%s} {}" % mlen).format(_sanitize_feature_name(column), attr_type))
        # compute the block of data
        if target is not None:
            d = [row[1].tolist() + [tgt] for row, tgt in zip(data.iterrows(), target.tolist())] \
                if isinstance(data, pd.DataFrame) else [list(row) + [tgt] for row, tgt in zip(data, target)]
        else:
            d = [row[1].tolist() + ["?"] for row in data.iterrows()] if isinstance(data, pd.DataFrame) else \
                [list(row) for row in data]
        d = "\n".join(",".join(map(str, row)) for row in d)
        with f.open('w') as arff:
            arff.write(("@RELATION \"{rel}\"\n\n{attr}\n@ATTRIBUTE {c: <%d} {cls}\n\n@DATA\n{data}" % mlen)
                       .format(rel=name, attr="\n".join(a), data=d, c="class", cls="{%s}" % ",".join(self.labels)))
        return f


class BayesNet(WekaClassifier):
    """ This implements the Bayes Network algorithm from Weka. """
    _weka_base = "weka.classifiers.bayes.BayesNet"


class Decorate(WekaClassifier):
    """ This implements the DECORATE algorithm from Weka. """
    _weka_base = "weka.classifiers.meta.Decorate"


class J48(WekaClassifier):
    """ This implements the J48 decision tree algorithm from Weka. """
    _weka_base = "weka.classifiers.trees.J48"


class Logistic(WekaClassifier):
    """ This implements the Multinomial Ridge Logistic Regression algorithm from Weka. """
    _weka_base = "weka.classifiers.functions.Logistic"


# initialize the registry of algorithms from the default source (~/.opt/algorithms.yml)
Algorithm.source = None

