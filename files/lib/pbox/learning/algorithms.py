# -*- coding: UTF-8 -*-
import pandas as pd
from tinyscript.helpers import Path
from weka.classifiers import Classifier

from ..common.item import Item
from ..common.utils import make_registry

__all__ = ["Algorithm", "WekaClassifier"]


class Algorithm(Item):
    """ Algorithm abstraction. """
    pass


class WekaClassifier(Classifier):
    """ This class implements a binding for using the Decorate algorithm from the Weka framework the same way as SkLearn
         algorithms. """
    train_file = "/tmp/weka/train.arff"
    test_file  = "/tmp/weka/test.arff"
    
    def __init__(self, **kwargs):
        m = kwargs.pop('model', None)
        if m:
            self.train_file = m.path.joinpath("train.arff")
            self.test_file = m.path.joinpath("test.arff")
        kwargs = {("-" + k if not k.startswith("-") else k): v for k, v in kwargs.items()}
        super(WekaClassifier, self).__init__(name=self.name, ckargs=kwargs)
    
    def fit(self, train_data, train_target):
        super(WekaClassifier, self).train(self.train_file)
    
    def predict(self, *args, **kwargs):
        self.predictions = []
        self.probabilities = []
        for result in super(WekaClassifier, self).predict(self.test_file):
            idx = self.labels.index(result.predicted)
            self.predictions.append(idx)
            proba = [0] * len(self.labels)
            proba[idx] = result.probability
            proba[(idx + 1) % 2] = 1 - result.probability
            self.probabilities.append(proba)
        return self.predictions
    
    def predict_proba(self, *args, **kwargs):
        if not hasattr(self, "probabilities"):
            self.predict()
        return pd.np.array(self.probabilities)
    
    def score(self, test_data, test_target):
        if not hasattr(self, "predictions"):
            self.predict()
        total = len(test_target)
        correct = 0
        for target, prediction in zip(test_target, self.predictions):
            if target == prediction:
                correct += 1
        return float(correct) / total
    
    @classmethod
    def to_arff(cls, model):
        """ This class method generates an ARFF file of the dataset/model, suitable for use with the Weka framework. """
        m = model
        l = len(m._features)
        ARFF_TEMPLATE = "@RELATION \"{rel}\"\n\n{attr}\n@ATTRIBUTE {c: <%s} {cls}\n\n@DATA\n{data}" % l
        for f, dss in zip([cls.train_file, cls.test_file], [m._train, m._test]):
            f = Path(f)
            Path(f.dirname, create=True)
            a = []
            for name, value in zip(m._features.keys(), m._data.iloc[0]):
                attr_type = ["string", "numeric"][isinstance(value, int) or isinstance(value, float)]
                a.append(("@ATTRIBUTE {: <%s} {}" % l).format(name, attr_type))
            d = "\n".join(",".join(map(str, list(row) + [tgt])) for row, tgt in zip(dss.data, dss.target))
            classes = list(map(lambda x: "" if str(x) == "nan" else str(x), set(dss.target['label'])))
            with f.open('w') as arff:
                arff.write(ARFF_TEMPLATE.format(rel=m._metadata['dataset']['name'], attr="\n".join(a), c="class",
                                                cls="{%s}" % ",".join(classes), data=d))


class BayesNet(WekaClassifier):
    """ This class implements the Bayes Network algorithm from Weka. """
    name = "weka.classifiers.bayes.BayesNet"


class Decorate(WekaClassifier):
    """ This class implements the DECORATE algorithm from Weka. """
    name = "weka.classifiers.meta.Decorate"


class J48(WekaClassifier):
    """ This class implements the J48 decision tree algorithm from Weka. """
    name = "weka.classifiers.trees.J48"


class Logistic(WekaClassifier):
    """ This class implements the Multinomial Ridge Logistic Regression algorithm from Weka. """
    name = "weka.classifiers.functions.Logistic"


make_registry(Algorithm)

