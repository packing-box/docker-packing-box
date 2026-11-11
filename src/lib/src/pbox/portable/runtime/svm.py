# -*- coding: UTF-8 -*-
""" Support Vector Classifier (SVC), as libsvm predicts with the model scikit-learn passes to it (private attributes
     support_vectors_, _dual_coef_, _intercept_, _n_support, _probA, _probB, _gamma). """
from math import exp, fabs, tanh


__all__ = ["svc"]


def _dot(x, y):
    s = 0.
    for a, b in zip(x, y):
        s += a * b
    return s


def _powi(base, times):
    """ libsvm's powi. """
    tmp, ret, t = base, 1., times
    while t > 0:
        if t % 2 == 1:
            ret *= tmp
        tmp = tmp * tmp
        t //= 2
    return ret


def _kernel(step, x, sv):
    k, g = step['kernel'], step['gamma']
    if k == "linear":
        return _dot(x, sv)
    if k == "poly":
        return _powi(g * _dot(x, sv) + step['coef0'], step['degree'])
    if k == "rbf":
        d = [a - b for a, b in zip(x, sv)]
        return exp(-g * _dot(d, d))
    if k == "sigmoid":
        return tanh(g * _dot(x, sv) + step['coef0'])
    raise ValueError(f"unsupported kernel '{k}'")


def _sigmoid_predict(dec, a, b):
    f = dec * a + b
    return exp(-f) / (1. + exp(-f)) if f >= 0 else 1. / (1 + exp(f))


def _multiclass_probability(k, r):
    """ libsvm's multiclass_probability (pairwise coupling). """
    p, q = [1. / k] * k, [[0.] * k for _ in range(k)]
    for t in range(k):
        for j in range(t):
            q[t][t] += r[j][t] * r[j][t]
            q[t][j] = q[j][t]
        for j in range(t + 1, k):
            q[t][t] += r[j][t] * r[j][t]
            q[t][j] = -r[j][t] * r[t][j]
    eps, qp = .005 / k, [0.] * k
    for _ in range(max(100, k)):
        pqp = 0.
        for t in range(k):
            qp[t] = 0.
            for j in range(k):
                qp[t] += q[t][j] * p[j]
            pqp += p[t] * qp[t]
        if max(fabs(qp[t] - pqp) for t in range(k)) < eps:
            break
        for t in range(k):
            diff = (-qp[t] + pqp) / q[t][t]
            p[t] += diff
            pqp = (pqp + diff * (diff * q[t][t] + 2 * qp[t])) / (1 + diff) / (1 + diff)
            for j in range(k):
                qp[j] = (qp[j] + diff * q[t][j]) / (1 + diff)
                p[j] /= (1 + diff)
    return p


def svc(step, x):
    sv, coef, rho, n_sv = step['support_vectors'], step['dual_coef'], step['rho'], step['n_support']
    k = len(n_sv)
    kv = [_kernel(step, x, s) for s in sv]
    start = [0] * k
    for i in range(1, k):
        start[i] = start[i - 1] + n_sv[i - 1]
    vote, dec, p = [0] * k, [], 0
    for i in range(k):
        for j in range(i + 1, k):
            s, si, sj = 0., start[i], start[j]
            for n in range(n_sv[i]):
                s += coef[j - 1][si + n] * kv[si + n]
            for n in range(n_sv[j]):
                s += coef[i][sj + n] * kv[sj + n]
            s -= rho[p]
            dec.append(s)
            vote[i if s > 0 else j] += 1
            p += 1
    best = 0
    for i in range(1, k):
        if vote[i] > vote[best]:
            best = i
    if not step.get('probability'):
        return best, None
    # Platt scaling (svm_predict_probability) ; NB: the predicted class remains the one of the votes, as with SVC
    min_prob, r, p = 1e-7, [[0.] * k for _ in range(k)], 0
    for i in range(k):
        for j in range(i + 1, k):
            r[i][j] = min(max(_sigmoid_predict(dec[p], step['prob_a'][p], step['prob_b'][p]), min_prob), 1 - min_prob)
            r[j][i] = 1 - r[i][j]
            p += 1
    # NB: scikit-learn's libsvm couples the pairwise probabilities even with 2 classes (no special case)
    proba = _multiclass_probability(k, r)
    return best, proba
