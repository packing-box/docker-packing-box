# -*- coding: UTF-8 -*-
from sklearn.model_selection import GridSearchCV

from .algorithms import *
from .metrics import *
from ..utils import benchmark


__all__ = ["compute", "predict", "prepare", "process", "run", "ALGORITHMS"]


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


def prepare(dataset):
    """ Compute and attach sets of data from self.data and self.target to the instance.
    
    :pre: the dataset is assumed to be balanced
    """
    ds = dataset
    ds.logger.debug("> Preparing train and test subsets...")
    data = ds._data.loc[:, ds._data.columns != "label"]
    target = ds._data.loc[:, ds._data.columns == "label"]
    # scale the data
    data = MinMaxScaler().fit_transform(data)
    # prepare for sklearn
    class Dummy: pass
    ds.train, ds.test = Dummy(), Dummy()
    ds.train.data, ds.test.data, ds.train.target, ds.test.target = train_test_split(data, target)
    # prepare for Weka
    WekaClassifier.to_arff(ds)
    return self


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

