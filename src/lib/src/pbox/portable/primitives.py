# -*- coding: UTF-8 -*-
""" Primitives used by the feature expressions, with no dependency outside the standard library.

These functions reproduce, operation for operation, the implementations pbox relies on (bintropy.entropy and
 pbox.helpers.utils.entropy), including their summation order, so that the values they compute are bit-identical:
 floating-point sums depend on the order of their terms.
"""
from collections import Counter
from math import ceil, log2


__all__ = ["block_entropy", "entropy", "zeropad"]

_TERMS, _TERMS_MAX_LENGTH = {}, 1 << 16


def _terms(lb):
    """ Entropy terms p * log2(p), with p = count / lb, for every count from 0 to lb (blocks of lb bytes) ; computed
         once per block length, with the same operations as bintropy, hence the same values. """
    t = _TERMS.get(lb)
    if t is None:
        t = _TERMS[lb] = [0.] + [(p := float(c) / lb) * log2(p) for c in range(1, lb + 1)]
    return t


def block_entropy(data, blocksize=0, ignore_half_block_zeros=False, ignore_half_block_same_byte=False):
    """ Shannon entropy, possibly per block of the given size ; same as bintropy.entropy (1.5.7).

    Returns the entropy of the whole input when blocksize is 0 and the input holds a single block, otherwise a tuple
     (entropies per block, average entropy), with None for the ignored blocks.

    NB: as in bintropy, 'ignore_half_block_zeros' has no effect on bytes (it compares the most common items, which are
         integers for bytes, with "\\x00" and b"\\x00") ; only 'ignore_half_block_same_byte' ignores blocks.

    NB: for speed, everything per block runs in C-implemented built-ins (set, map over bytes.count, max, sum), with the
         terms p * log2(p) taken from a table per block length (see _terms) ; the counts are the same integers and the
         terms are summed in the order of bintropy (that of set(block)), with the same sum(), hence a bit-identical
         result. """
    e, l = [], len(data)
    if l == 0:
        return ([], None) if blocksize > 0 else 0.
    bs = blocksize or l
    n_blocks, n_ignored = ceil(float(l) / bs), 0
    for i in range(0, l, bs):
        block, ignore = data[i:i+bs], False
        lb = len(block)
        values = set(block)
        chr_cts = list(map(block.count, values))
        if ignore_half_block_zeros or ignore_half_block_same_byte:
            if (max_occ := max(chr_cts)) > lb // 2:
                ignore = ignore_half_block_same_byte or ignore_half_block_zeros and \
                         any(c in [v for v, n in zip(values, chr_cts) if n == max_occ] for c in ["\x00", b"\x00"])
        if ignore:
            e.append(None)
            n_ignored += 1
            continue
        if lb <= _TERMS_MAX_LENGTH:
            e.append(-sum(map(_terms(lb).__getitem__, chr_cts)) or .0)
        else:
            e.append(-sum([p * log2(p) for p in [float(ctr) / lb for ctr in chr_cts]]) or .0)
    return (e, sum(n or 0 for n in e) / ((n_blocks - n_ignored) or 1)) if n_blocks > 1 or blocksize > 0 else e[0]


def entropy(data):
    """ Shannon entropy of bytes ; same as pbox.helpers.utils.entropy(data) (1-grams, all byte values).

    The terms are summed by decreasing count, ties in increasing byte order, as the n-grams distribution of exeplot
     sorts them (a stable sort of the byte counts). """
    l = len(data)
    if l == 0:
        return 0.
    cnt = Counter(data)
    counts = [cnt.get(b, 0) for b in range(max(data) + 1)]  # as numpy.bincount: up to the highest byte value present
    e = 0.
    for _, n in sorted(enumerate(counts), key=lambda p: p[1], reverse=True):
        e -= (p := n / l) * log2(p or 1)
    return e


def zeropad(n):
    """ Pad a list of integers with zeros up to n items ; same as tinyscript's zeropad for lists of integers.

    NB: as in tinyscript, a list longer than n is returned as is (not truncated), and an empty list gives n "\\x00"
         strings (not integers). """
    def _pad(lst):
        lst = list(lst)
        if len(lst) == 0:
            return n * ["\x00"]
        return lst if len(lst) >= n else lst + (n - len(lst)) * [0 if isinstance(lst[0], int) else "\x00"]
    return _pad
