# -*- coding: UTF-8 -*-
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score


__all__ = ["metrics"]


def metrics(dataset, logger=None):
    """ Metrics computation method. """
    logger.debug("> Computing metrics...")
    # compute indicators
    tn, fp, fn, tp = confusion_matrix(dataset.test.target, dataset.test.prediction).ravel()
    # compute evaluation metrics:
    accuracy = float(tp + tn) / (tp + tn + fp + fn)
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)  # or also sensitivity
    f_measure = float(2 * precision * recall) / (precision + recall)
    mcc = matthews_corrcoef(dataset.test.target, dataset.test.prediction)
    auc = roc_auc_score(list(map(int, dataset.test.target)), dataset.test.proba)
    # return metrics for further display
    return list(map(lambda x: "%.3f" % x, [accuracy, precision, recall, f_measure, mcc, auc]))

