#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import lief
from asciistuff import AsciiFile, Banner, Quote
from datetime import datetime
from pbox import *
from rich import inspect as inspect_pretty, pretty
from tinyscript import *
from tinyscript.helpers import *
pretty.install()


__QUOTES = [
    "Malware analysis turns chaos into intelligence.",
    "A packer hides intent; entropy unveils it.",
    "Every executable leaves a behavioral fingerprint — learn to read it.",
    "Machine learning can detect patterns humans overlook, but attackers adapt faster.",
    "Static analysis sees structure; dynamic analysis sees truth.",
    "Packed binaries are puzzles — entropy is your first clue.",
    "AI enhances detection, but human intuition closes the loop.",
    "Malware evolves, so must our classifiers.",
    "A sandbox is where malware shows its real face.",
    "Deception in code is art; uncovering it is science.",
    "Signatures die quickly; behavior endures.",
    "Reverse engineering is the archaeology of malicious intent.",
    "Entropy never lies — it whispers concealment.",
    "The best unpacker is understanding the packer’s logic.",
    "Automation scales, expertise refines.",
    "Machine learning fights malware, but explainability wins trust.",
    "Static bytes tell a story; dynamic traces tell the plot.",
    "A single opcode can betray a payload.",
    "Adversaries train models too — never forget that.",
    "Malware authors love obfuscation; analysts love patterns.",
    "Heuristics bridge the gap between rules and learning.",
    "Packing delays analysis, not detection.",
    "The debugger is your microscope into malicious life.",
    "Every ML model has blind spots — attackers find them first.",
    "Behavioral models age slower than signatures.",
    "Entropy spikes are often digital fingerprints of fear.",
    "Malware classification is a cat-and-mouse game of feature design.",
    "Dynamic unpacking is patience turned into insight.",
    "Machine learning needs context, not just data.",
    "Even obfuscation patterns can become features.",
    "Malware hides in code, but reveals itself in behavior.",
    "Packer layers are meant to confuse, not to conceal forever.",
    "Disassembly is the first step toward understanding digital deceit.",
    "The best analysts read binaries like novels — with curiosity and skepticism.",
    "Every malicious sample is a lesson in adversarial creativity.",
    "Machine learning without feature sanity is statistical guesswork.",
    "Polymorphism is evolution under pressure.",
    "Data drives models; intuition drives breakthroughs.",
    "Memory dumps tell truths binaries try to hide.",
    "Good analysis tools amplify insight, not replace it.",
    "The harder a binary resists unpacking, the more valuable the sample.",
    "Learning from malware requires both math and instinct.",
    "Adversarial AI is the next arms race in malware detection.",
    "Code similarity is the shadow of shared intent.",
    "Dynamic analysis reveals what obfuscation conceals.",
    "Malware families evolve faster than signature databases.",
    "A packed executable is a message wrapped in entropy.",
    "Machine learning models should explain, not just predict.",
    "Understanding malware means thinking like its creator.",
    "Every false positive is a reminder that precision matters as much as recall.",
]


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

__f = AsciiFile()
__f['title', {'fgcolor': "lolcat"}] = Banner("Packing-Box Console", font="cyberlarge")
__f['myquote', {'adjust': "right"}] = Quote(random.choice(__QUOTES), "ChatGPT5")
print(__f)

