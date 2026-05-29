# -*- coding: UTF-8 -*-
import numpy as np


__all__ = ["proba_from_top_logprobs", "logsumexp"]

_PACKED_TOKENS = ("packed", " packed", "yes", " yes", "true", " true", "1", " 1")
_NOT_PACKED_TOKENS = ("not", " not", "no", " no", "false", " false", "0", " 0", "unpacked", " unpacked")


def logsumexp(values):
    arr = np.asarray(values, dtype=float)
    m = np.max(arr)
    return float(m + np.log(np.sum(np.exp(arr - m))))


def proba_from_top_logprobs(top_logprobs):
    """Convert first-token top_logprobs to binary probabilities.

    Returns None when token evidence is insufficient.
    Output format is [P(not-packed), P(packed)].
    """
    if not isinstance(top_logprobs, dict) or len(top_logprobs) == 0:
        return None
    packed_scores = [v for k, v in top_logprobs.items() if k in _PACKED_TOKENS]
    not_packed_scores = [v for k, v in top_logprobs.items() if k in _NOT_PACKED_TOKENS]
    if not packed_scores or not not_packed_scores:
        return None
    l_packed = logsumexp(packed_scores)
    l_not_packed = logsumexp(not_packed_scores)
    den = logsumexp([l_not_packed, l_packed])
    p_packed = float(np.exp(l_packed - den))
    return np.array([1.0 - p_packed, p_packed], dtype=float)
