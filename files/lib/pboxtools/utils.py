# -*- coding: UTF-8 -*-
from elftools.elf.elffile import ELFFile
from magic import from_file
from math import log2
from pefile import PE
from re import match


__all__ = ["bintropy", "entropy"]


def bintropy(executable, full=False):
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
        return entropy(exe)
    # compute a weighted entropy of all the sections of the executable
    else:
        e = {}
        if pe:
            for s in PE(str(executable)).sections:
                e[s.Name.strip(b"\x00").decode() or "main"] = entropy(s.get_data())
        else:
            with open(str(executable), 'rb') as f:
                elf = ELFFile(f)
                for s in elf.iter_sections():
                    n = s.name.strip()
                    e[n or "main"] = entropy(elf.get_section_by_name(n).data())
        return e


def entropy(something):
    """ Shannon entropy. """
    d, l = {}, len(something)
    for c in something:
        d.setdefault(c, 0)
        d[c] += 1
    return -sum([p * log2(p) for p in [float(ctr) / l for ctr in d.values()]]) or 0

