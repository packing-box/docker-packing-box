#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import lief
from pbox import *
from rich import inspect as inspect_pretty, pretty
from tinyscript import *
from tinyscript.helpers import *
pretty.install()


def get_sample(hash, folder="dataset-packed-pe", hashtype="sha256"):
    """ Get a sample by hash from the target dataset folder. """
    if hashtype not in hashlib.algorithms_available:
        raise ValueError("Bad hash algorithm")
    p = Path(folder or ".")
    for f in p.walk():
        if f.is_file() and getattr(hashlib, hashtype + "_file")(str(f)) == hash:
            return Executable(f)
    raise OSError(f"Could not find sample with hash '{hash}' in folder '{folder}'")


def pick_sample(folder="dataset-packed-pe", packed=False):
    """ Pick a sample randomly from the target dataset folder.
         NB: if the target has 'not-packed' and 'packed' folders, use them to discriminate between not packed and packed
              samples ; otherwise consider the entire folder
    """
    p = Path(folder or ".")
    p_np, p_p = p.joinpath("not-packed"), p.joinpath("packed")
    if p_np.is_dir() and p_p.is_dir():
        p = [p_np, p_p][packed]
    logger.debug(f"Target folder is: {p}")
    candidates = [f for f in p.walk()]
    while 1:
        try:
            f = random.choice(candidates)
            return Executable(f)
        except IndexError:
            raise IndexError(f"Could not find a matching sample in folder '{folder}'")
        except:
            candidates.remove(f)

