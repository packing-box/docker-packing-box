# -*- coding: UTF-8 -*-
from elftools.elf.elffile import ELFFile
from magic import from_file
from math import ceil, log2
from pefile import PE
from re import match


__all__ = ["bintropy", "entropy"]


def bintropy(executable, full=False, blocksize=256, ignore_half_block_zeros=True):
    """ Simple implementation of Bintropy as of https://ieeexplore.ieee.org/document/4140989. """
    try:
        ft = from_file(str(executable))
    except OSError:
        return
    pe = match("PE32\+? executable ", ft) is not None
    if not pe and not match("(set[gu]id )?ELF (32|64)-bit ", ft):
        return  # only works for PE and ELF
    # compute the entropy of the whole executable
    if full:
        with open(str(executable), 'rb') as f:
            exe = f.read()
        return entropy(exe, blocksize)
    # compute a weighted entropy of all the sections of the executable
    else:
        e = {}
        if pe:
            for s in PE(str(executable)).sections:
                e[s.Name.strip(b"\x00").decode() or "main"] = entropy(s.get_data(), blocksize, ignore_half_block_zeros)
        else:
            with open(str(executable), 'rb') as f:
                elf = ELFFile(f)
                for s in elf.iter_sections():
                    n = s.name.strip()
                    e[n or "main"] = entropy(elf.get_section_by_name(n).data(), blocksize, ignore_half_block_zeros)
        return e


def entropy(something, blocksize=0, ignore_half_block_zeros=False):
    """ Shannon entropy, with the possibility to compute the entropy per block with a given size.
    
    :param something:               string or bytes
    :param blocksize:               block size to be considered for the total entropy
    :param ignore_half_block_zeros: ignore blocks in which at least half of the chars/bytes are zeros
    """
    e, l = [], len(something)
    if l == 0:
        return
    bs = blocksize or l
    n_blocks, n_ignored = ceil(float(l) / bs), 0
    for i in range(0, l, bs):
        block, n_zeros, ignore = something[i:i+bs], 0, False
        lb = len(block)
        # consider ignoring blocks in which more than half of the chars/bytes are zeros
        if ignore_half_block_zeros:
            lz = lb // 2
            for c in block:
                if isinstance(c, int) and c == 0 or isinstance(c, str) and ord(c) == 0:
                    n_zeros += 1
                if n_zeros > lz:
                    ignore = True
                    break
        if ignore:
            e.append(None)
            n_ignored += 1
            continue
        # if not ignored, process it
        d = {}
        for c in block:
            d.setdefault(c, 0)
            d[c] += 1
        e.append(-sum([p * log2(p) for p in [float(ctr) / lb for ctr in d.values()]]) or .0)
    # return the entropies per block and the average entropy of all blocks if n_blocks > 1
    return (e, sum([n or 0 for n in e]) / ((n_blocks - n_ignored) or 1)) if n_blocks > 1 else e[0]

