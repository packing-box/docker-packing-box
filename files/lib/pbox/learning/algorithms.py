# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier as ABC, RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB as BNB, GaussianNB as GNB, MultinomialNB as MNB
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.svm import LinearSVC as LSVC, SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from weka.classifiers import Classifier

from ..utils import benchmark


__all__ = ["compute", "predict", "process", "run", "ALGORITHMS"]


# list of classifiers to be used
SELECTED = [
    ("AB", "Adaptive Boosting"),
    ("BN", 'Bayesian Network'),
    ("BNB", "Bernoulli Naive Bayes"),
    ("D", "Decorate"),
    ("DT", "Decision Tree"),
    ("GNB", "Gaussian Naive Bayes"),
    ("J48", "Decision Tree"),
    ("kNN", "k-Nearest Neighbors"),
    ("LR", "Logistic Regression"),
    ("LSVM", "Linear Support Vector Machine"),
    ("MLP", "Multi-Layer Perceptron"),
    ("MNB", "Multinomial Naive Bayes"),
    ("RF", "Random Forest"),
    ("SVM", "Support Vector Machine"),
]

ALGORITHMS = [abbr for abbr, _ in SELECTED]

# list of implemneted classifiers
CLASSIFIERS = {
    # Weka algorithms

    'D': lambda **kw: DEC(**kw),
    # Ref: http://weka.sourceforge.net/doc.stable/weka/classifiers/meta/Decorate.html
    
    'J48': lambda **kw: J48(**kw),
    # Ref: http://weka.sourceforge.net/doc.stable/weka/classifiers/meta/Decorate.html
    
    'BN': lambda **kw: BN(**kw),
    # Ref: http://weka.sourceforge.net/doc.dev/weka/classifiers/bayes/BayesNet.html
    
    'LR': lambda **kw: LR(**kw),
    # Ref: http://weka.sourceforge.net/doc.dev/weka/classifiers/functions/Logistic.html
    
    # SkLearn algorithms
    
    'AB': lambda **kw: ABC(**kw),
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    
    'RF': lambda **kw: RFC(**kw),
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    
    'BNB': lambda **kw: BNB(**kw),
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
    
    'GNB': lambda **kw: GNB(**kw),
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    
    'MNB': lambda **kw: MNB(**kw),
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    
    'kNN': lambda **kw: KNC(**kw),
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    
    'MLP': lambda **kw: MLPC(**kw),
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    
    'SVM': lambda **kw: SVC(**kw),
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    'LSVM': lambda **kw: LSVC(**kw),
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

    'DT': lambda **kw: DTC(**kw),
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
}

# static test parameters
RSTATE = 42
STATIC_PARAM = {
    'D': {'-S': RSTATE},
    #  defaults:
    #   -E    15                          Desired size of ensemble.
    #   -R    1.0                         Factor that determines number of artificial examples to generate.
    #   -S    1                           Random number seed.
    #   -I    50                          Number of iterations.
    #   -W    weka.classifiers.trees.J48  Full name of base classifier.

    'J48': {'-Q': RSTATE},
    #  defaults:
    #   -C    0.25                  Set confidence threshold for pruning.
    #   -M    2       Set minimum number of instances per leaf.
    #   -N    3       Set number of folds for reduced error pruning. One fold is used as pruning set.

    'BN': {},
    #  defaults:
    #   -E    weka.classifiers.bayes.net.estimate.SimpleEstimator   Estimator algorithm.
    #   -Q    weka.classifiers.bayes.net.search.SearchAlgorithm     Search algorithm.

    'LR': {},
    #  defaults:
    #   -M    -1      Set maximum number of iterations (-1: until convergence)

    'AB': {'random_state': RSTATE},
    #  defaults:
    #   base_estimator = DecisionTreeClassifier
    #   n_estimators   = 50
    #   learning_rate  = 1
    #   algorithm      = ’SAMME.R’

    'RF': {'criterion': "entropy", 'random_state': RSTATE},
    #  defaults:
    #   n_estimators = 10
    #   max_features = 'auto'
    #   max_depth    = None
    #   others: see Ref
    
    'BNB': {},
    #  defaults:
    #   alpha       = 1.0
    #   binarize    = 0.0
    #   fit_prior   = True
    #   class_prior = None

    'GNB': {},
    #  defaults:
    #   priors = None

    'MNB': {},
    #  defaults:
    #   alpha       = 1.0
    #   fit_prior   = True
    #   class_prior = None

    'kNN': {'weights': "uniform"},
    #  defaults:
    #   n_neighbors = 5
    #   leaf_size   = 30
    #   p           = 2 (euclidian distance)
    #   metric      = 'minkowski'
    #   others: see Ref

    'MLP': {'hidden_layer_sizes': (10, 2), 'random_state': RSTATE},
    #  defaults:
    #   hidden_layer_sizes = (100,)
    #   activation         = 'relu'
    #   solver             = 'adam'
    #   alpha              = 1e-4
    #   others: see Ref

    'SVM': {'probability': True, 'random_state': RSTATE},
    #  defaults:
    #   C      = 1.0
    #   kernel = 'rbf'
    #   degree = 3
    #   gamma  = 'auto'
    #   others: see Ref

    'LSVM': {'random_state': RSTATE},
    #  defaults:
    #   penalty = 'l2'
    #   loss    = 'squared_hinge'
    #   dual    = True
    #   tol     = 1e-4
    #   others: see Ref

    'DT': {'criterion': "entropy", 'random_state': RSTATE},
    #  defaults:
    #   splitter  = 'best'
    #   max_depth = None
    #   other: see Ref
}

# test parameter ranges for grid search cross-validation
CV_PARAM = {
    'RF':   {'max_depth': range(6, 21)},
    'kNN':  {'n_neighbors': range(1, 6, 2)},
    'SVM':  {'C': pd.np.logspace(4, 6, 3), 'gamma': pd.np.logspace(-3, -1, 3)},
    'LSVM': {'C': pd.np.logspace(1, 6, 6)},
    'DT':   {'max_depth': range(3, 11)},
}


@benchmark
def compute(dataset, name, cv=3, n_jobs=4, logger=None):
    """ Training method handling the cross-validation. """
    try:
        CLASSIFIERS[name]
    except KeyError:
        logger.debug("{} not available".format(name))
        dataset.classifier = None
        return
    # get classifer and grid search parameters
    try:
        params = STATIC_PARAM[name]
    except:
        params = {}
    try:
        param_grid = CV_PARAM[name]
    except:
        param_grid = None
    classifier = CLASSIFIERS[name](**params)
    classifier.labels = dataset.labels
    # if a param_grid is input, perform cross-validation and select the best classifier
    logger.debug("Processing {}...".format(name))
    if param_grid is not None:
        logger.debug("> Applying Grid Search (CV={})...".format(cv))
        grid = GridSearchCV(classifier, param_grid=param_grid, cv=cv,
                            scoring='accuracy', n_jobs=n_jobs)
        grid.fit(dataset.data, dataset.target)
        logger.debug("> Best parameters found:\n  {}".format(grid.best_params_))
        results = '\n'.join("  %0.3f (+/-%0.03f) for %r" % (m, s * 2, p) \
            for m, s, p in zip(grid.cv_results_['mean_test_score'],
                               grid.cv_results_['std_test_score'],
                               grid.cv_results_['params']))
        logger.debug(" > Grid scores:\n{}".format(results))
        params.update(grid.best_params_)
        classifier = CLASSIFIERS[name](**params)
    # now fit the (best) classifier and predict labels
    logger.debug("> Fitting the classifier...")
    classifier.fit(dataset.train.data, dataset.train.target)
    logger.debug("> Making predictions...")
    dataset.test.prediction = classifier.predict(dataset.test.data)
    dataset.test.proba = classifier.predict_proba(dataset.test.data)[:, 1]
    dataset.classifier = classifier


@benchmark
def predict(classifier, data, **kwargs):
    """ Prediction function. """
    return classifier.predict(data)


def process(algorithms, dataset, cv, n_jobs, logger=None):
    """ Generator for processing each algorithm from a given list. """
    # execute ML algorithms and collect metrics for display
    for name, description in algorithms:
        if isinstance(description, dict):
            description = description.values()[0]
        compute(dataset, name, cv, n_jobs, logger=logger, info=name)
        if dataset.classifier is None:
            continue
        yield [name, description] + metrics(dataset, logger=logger)


@benchmark
def run(**kwargs):
    """ Execute tests on the given dataset and classifiers. """
    branch = kwargs.pop("branch")
    logger = kwargs.get("logger")
    # get rid of useless kwargs
    kwargs.pop("_debug_level")
    kwargs.pop("verbose")
    # extract kwargs for the compute function
    cv = kwargs.pop("cv", 3)
    n_jobs = kwargs.pop("jobs", 4)
    extra = kwargs.pop("extra", False)
    algo = kwargs.pop("algorithm", None)
    # process the dataset
    only_dataset = kwargs.pop("dataset", False)
    force = kwargs.pop("force", False) or only_dataset
    dataset = Dataset(force=force, **kwargs)
    if only_dataset:
        print(dataset)
        return
    dataset.prepare()
    if branch == "test":
        c = kwargs.get("classes")
        c = ["class {}", "classes {}"][len(c) > 1].format(','.join(c))
        # process selected algorithms (from [CRESCI])
        table_data = [["", "Model", "Accuracy", "Precision", "Recall", "F-Measure", "MCC", "AUC"]]
        for row in process(SELECTED, dataset, cv, n_jobs, logger=logger):
            table_data.append(row)
        table = Table(highlight_best(table_data), "Results ([CRESCI] - {})".format(c))
        print(table.table)
        # process extra algorithms (not from [CRESCI])
        if extra:
            table_data = [["", "Model", "Accuracy", "Precision", "Recall", "F-Measure", "MCC", "AUC"]]
            for row in process(EXTRA, dataset, cv, n_jobs, logger=logger):
                table_data.append(row)
            table = Table(highlight_best(table_data), "Results (others - {})".format(c))
            print(table.table)
    elif branch == "predict":
        # train a selected classifier
        compute(dataset, algo, cv, n_jobs, logger=logger, info=algo)
        # load new data and predict new labels
        ds = Dataset(**kwargs)
        p = predict(dataset.classifier, ds.data, logger=logger, info=len(ds.data))
        for i, new_label in enumerate(p):
            logger.info("#{}: {}".format(i, ds.labels[new_label]))


# ---------------------------------------------- WEKA ALGORITHM BINDINGS -----------------------------------------------
class WekaClassifier(Classifier):
    """ This class implements a binding for using the Decorate algorithm from the Weka framework the same way as SkLearn
         algorithms. """
    train_file = "/tmp/weka/train.arff"
    test_file  = "/tmp/weka/test.arff"
    
    def __init__(self, **kwargs):
        super(WekaClassifier, self).__init__(name=self.name, ckargs=kwargs)
    
    def fit(self, train_data, train_target):
        super(WekaClassifier, self).train(self.train_file)
    
    def predict(self, *args, **kwargs):
        if not hasattr(self, "labels"):
            self.labels = [0, 1]
        self.predictions = []
        self.probabilities = []
        for result in super(WekaClassifier, self).predict(self.test_file):
            idx = self.labels.index(result.predicted)
            self.predictions.append(idx)
            proba = [0, 0]
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
        if not hasattr(self, "labels"):
            self.labels = [0, 1]
        total = len(test_target)
        correct = 0
        for target, prediction in zip(test_target, self.predictions):
            if target == prediction:
                correct += 1
        return float(correct) / total


class BN(WekaClassifier):
    """ This class implements the Bayes Network algorithm from Weka. """
    name = "weka.classifiers.bayes.BayesNet"


class DEC(WekaClassifier):
    """ This class implements the DECORATE algorithm from Weka. """
    name = "weka.classifiers.meta.Decorate"


class J48(WekaClassifier):
    """ This class implements the J48 decision tree algorithm from Weka. """
    name = "weka.classifiers.trees.J48"


class LR(WekaClassifier):
    """ This class implements the Multinomial Ridge Logistic Regression algorithm from Weka. """
    name = "weka.classifiers.functions.Logistic"

