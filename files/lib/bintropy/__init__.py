# -*- coding: UTF-8 -*-
from elftools.elf.elffile import ELFFile
from magic import from_file
from math import ceil, log2
from pefile import PE
from re import match


__all__ = ["bintropy", "entropy", "is_packed"]


THREASHOLD_AVERAGE_ENTROPY = 6.677
THREASHOLD_HIGHEST_ENTROPY = 7.199


def bintropy(executable, full=False, blocksize=256, ignore_half_block_zeros=True, decide=True, verbose=False):
    """ Simple implementation of Bintropy as of https://ieeexplore.ieee.org/document/4140989.
    
    :param executable:              path to the executable to be analyzed
    :param full:                    process the executable as a whole or per section only (cfr modes of operation)
    :param blocksize:               process per block of N bytes (0 means considering the executable as a single block)
    :param ignore_half_block_zeros: ignore blocks having more than half of zeros
    :param decide:                  decide if packed or not, otherwise simply return the entropy values
    :param verbose:                 display detailed information
    """
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
        if verbose:
            print("[DEBUG] Entropy (Shannon): {}".format(entropy(exe)))
        e = entropy(exe, blocksize, ignore_half_block_zeros)
        if verbose:
            print("[DEBUG] Entropy (average): {}".format(e[1]))
            iw = len(str(len(e[0])))
            for i, j in enumerate(e[0]):
                print(("  #{: <%s}: {}" % iw).format(i + 1, "-" if j is None else j))
        return is_packed(e[0], e[1], verbose) if decide else e
    # compute a weighted entropy of all the sections of the executable
    else:
        e, w = {}, {}
        if pe:
            for s in PE(str(executable)).sections:
                n = s.Name.strip(b"\x00").decode() or "main"
                e[n] = entropy(s.get_data(), blocksize, ignore_half_block_zeros)
                w[n] = len(s.get_data())
        else:
            with open(str(executable), 'rb') as f:
                elf = ELFFile(f)
                for s in elf.iter_sections():
                    n = s.name.strip() or "main"
                    e[n] = entropy(elf.get_section_by_name(n).data(), blocksize, ignore_half_block_zeros)
                    w[n] = len(elf.get_section_by_name(n).data())
        if verbose:
            print("[DEBUG] Entropy per section:")
            for name, entr in e.items():
                print("{}: {}{}".format(name, (entr or ("", "-"))[1], [" (average)", ""][entr is None]))
                for i, j in enumerate((entr or [[]])[0]):
                    print(("  #{: <%s}: {}" % len(str(len(entr[0])))).format(i + 1, "-" if j is None else j))
        if decide:
            # aggregate per-section entropy scores
            e2, e_avg2, t = 0, 0, 0
            for n, entr in e.items():
                if entr is None:
                    continue
                e2 += max(entr[0]) * w[n]
                e_avg2 += entr[1] * w[n]
                t += w[n]
            return is_packed(e2 / t, e_avg2 / t, verbose)
        return e


def entropy(something, blocksize=0, ignore_half_block_zeros=False):
    """ Shannon entropy, with the possibility to compute the entropy per block with a given size, possibly ignoring
         blocks in which at least half of the characters or bytes are zeros.
    
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


def is_packed(entropies, average, verbose=False):
    """ Decision criterium as of https://ieeexplore.ieee.org/document/4140989.
    
    :param entropies: the list of block entropy values or the highest block entropy value
    ;param average:   the average block entropy
    """
    if not isinstance(entropies, (list, tuple)):
        entropies = [entropies]
    entropies = [x for x in entropies if x is not None]
    max_e = max(entropies)
    c1 = average > THREASHOLD_AVERAGE_ENTROPY
    c2 = max_e > THREASHOLD_HIGHEST_ENTROPY
    if verbose:
        print("[DEBUG] Average entropy criteria (>{}): {} ({})".format(THREASHOLD_AVERAGE_ENTROPY, c1, average))
        print("[DEBUG] Highest entropy criteria (>{}): {} ({})".format(THREASHOLD_HIGHEST_ENTROPY, c2, max_e))
    return c1 and c2

