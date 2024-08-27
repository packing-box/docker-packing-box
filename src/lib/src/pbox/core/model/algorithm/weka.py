# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from tinyscript import functools
from tinyscript.helpers import Path, TempPath
from weka.classifiers import Classifier, WEKA_CLASSIFIERS


__all__ = ["WekaClassifier"]


def to_arff(name="undefined"):
    def _wrapper(f):
        @functools.wraps(f)
        def _subwrapper(self, *args, **kwargs):
            """ This wrapper generates an ARFF file of the dataset/model, suitable for use with the Weka framework. """
            destination = kwargs.get('destination')
            dest = destination or TempPath(prefix="weka-arff-", length=8).tempfile("data.arff")
            # compute the list of features
            a, mlen = [], 0
            mlen = max(map(len, self._feature_names))
            for fname in self._feature_names:
                # all features are numeric when called from .fit(...) or .predict(...) as they were preprocessed with
                #  the ._prepare(...) method of the Model class
                a.append(("@ATTRIBUTE {: <%s} numeric" % mlen).format(fname))
            # compute the block of data
            try:
                X, y = args[:2]
            except ValueError:
                X, y = args[0], None
            if y is not None:
                self.labels = sorted(set(map(str, y)))
                d = [row[0] + [tgt] for row, tgt in zip(X.iterrows(), y)] \
                    if isinstance(X, pd.DataFrame) else [list(row) + [tgt] for row, tgt in zip(X, y)]
            else:
                d = [row[0] + ["?"] for row in X.iterrows()] if isinstance(X, pd.DataFrame) else \
                    [list(row) + ["?"] for row in X]
            d = "\n".join(",".join(map(str, row)) for row in d)
            # create the destination ARFF file
            p = Path(dest)
            kwargs['arff'] = str(p)
            Path(p.dirname, create=True)
            with p.open('w') as arff:
                arff.write(("@RELATION \"{rel}\"\n\n{attr}\n@ATTRIBUTE {c: <%d} {cls}\n\n@DATA\n{data}" % mlen)
                           .format(rel=name, attr="\n".join(a), data=d, c="class", cls=f"{{{','.join(self.labels)}}}"))
            r = f(self, *args, **kwargs)
            if destination is None:
                p.remove()
                p.dirname.remove()
            return r
        return _subwrapper
    return _wrapper


class WekaClassifier(Classifier):
    """ This class implements a binding for using the Decorate algorithm from the Weka framework the same way as
         SkLearn algorithms. """
    def __init__(self, **kwargs):
        self._feature_names = list(map(lambda n: n.replace("<", "[lt]").replace(">", "[gt]"),
                                       kwargs.pop('feature_names')))
        kwargs = {("-" + k if not k.startswith("-") else k): v for k, v in kwargs.items()}
        super(WekaClassifier, self).__init__(name=self._weka_base, ckargs=kwargs)
    
    @functools.lru_cache
    def _predict(self, arff):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "labels")
        pred, proba = [], []
        for result in super(WekaClassifier, self).predict(arff):
            idx = self.labels.index(result.predicted)
            pred.append(idx)
            pb = [0] * len(self.labels)
            pb[idx] = result.probability
            pb[(idx + 1) % 2] = 1 - result.probability
            proba.append(pb)
        return np.array(pred), np.array(proba)
    
    @to_arff("train")
    def fit(self, X, y, **kwargs):
        super(WekaClassifier, self).train(kwargs['arff'])
    
    @to_arff("test")
    def predict(self, X, **kwargs):
        return self._predict(kwargs['arff'])[0]
    
    @to_arff("test")
    def predict_proba(self, X, **kwargs):
        return self._predict(kwargs['arff'])[1]
    
    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


for classifier in WEKA_CLASSIFIERS:
    name = classifier.split(".")[-1]
    if name not in globals():
        exec(f"class {name}(WekaClassifier): _weka_base = \"{classifier}\"")
    else:
        break  # already initialized
    

# not part of WEKA_CLASSIFIERS
class Decorate(WekaClassifier):
    """ This implements the DECORATE algorithm from Weka. """
    _weka_base = "weka.classifiers.meta.Decorate"

