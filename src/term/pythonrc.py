#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import lief
from datetime import datetime
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


class Prompt:
    def __init__(self):
        self.start = datetime.now()
    
    def __str__(self):
        t = (datetime.now() - self.start).total_seconds()
        h, remainder = divmod(t, 3600)
        m, s = divmod(remainder, 60)
        s, r = str(s).split(".")
        h, m, s, r = int(h), int(m), int(s), float(f".{r}")
        return f"\x01\033[36m\x02{h:02d}:{m:02d}:{s:02d}.{r:.3f}\x01\033[0m\x02 >>> "

sys.ps1 = Prompt()
sys.ps2 = "               \x01\033[35m\x02...\x01\033[0m\x02 "

