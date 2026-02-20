#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import atexit
import lief
from asciistuff import AsciiFile, Banner, Quote
from datetime import datetime
from dsff import DSFF
from pbox import *
from rich import inspect as inspect_pretty, pretty
from tinyscript import *
from tinyscript.helpers import *
pretty.install()


def at_exit():
    from gc import collect
    collect()
atexit.register(at_exit)


def get_sample(hash, folder="dataset-packed-pe", hashtype="sha256"):
    """ Get a sample by hash from the target dataset folder. """
    if hashtype not in hashlib.algorithms_available:
        raise ValueError("Bad hash algorithm")
    p = Path(folder or ".")
    for f in p.walk():
        if f.is_file() and getattr(hashlib, hashtype + "_file")(str(f)) == hash:
            return Executable(f)
    raise OSError(f"Could not find sample with hash '{hash}' in folder '{folder}'")


def history(reverse=False, yield_error=True):
    import codeop
    import readline
    # collect blocks of code
    i, blocks, block = 1, [], ""
    while i <= readline.get_current_history_length():
        line = f"{readline.get_history_item(i)}\n"
        try:
            codeop.compile_command(block + line, symbol="exec")
            # valid block happens when there is a non-null block and the current line is back to no indent
            if block and codeop.compile_command(block, symbol="exec") and re.match(r"^[^\s]", line):
                blocks.append(block)
                block = ""
                continue  # rehandle the current line
        except Exception as e:
            # if non-null block while error, this is an incomplete block that can be collected if relevant
            if block:
                if yield_error:
                    blocks.append(block)
                block = ""
                continue  # rehandle the current line
            # if no block but yet there is an error, collect the line if relevant
            else:
                if yield_error:
                    blocks.append(line)
                block, line = "", ""  # pass the faulty line
        block += line
        i += 1
    # then yield them in the desired order
    for b in (blocks[::-1] if reverse else blocks):
        yield b


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


def reexec(*patterns):
    import inspect
    import re
    for pattern in patterns:
        for block in history(reverse=True, yield_error=False):
            if re.search(r"reexec\(.*?\)", block):  # avoid infinite loop
                continue
            if re.search(pattern, block):
                frame = inspect.currentframe().f_back
                exec(block, frame.f_globals, frame.f_locals)
                break


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

__QUOTES = [
    ("ChatGPT5", "Malware analysis turns chaos into intelligence."),
    ("ChatGPT5", "A packer hides intent; entropy unveils it."),
    ("ChatGPT5", "Every executable leaves a behavioral fingerprint — learn to read it."),
    ("ChatGPT5", "Machine learning can detect patterns humans overlook, but attackers adapt faster."),
    ("ChatGPT5", "Static analysis sees structure; dynamic analysis sees truth."),
    ("ChatGPT5", "Packed binaries are puzzles — entropy is your first clue."),
    ("ChatGPT5", "AI enhances detection, but human intuition closes the loop."),
    ("ChatGPT5", "Malware evolves, so must our classifiers."),
    ("ChatGPT5", "A sandbox is where malware shows its real face."),
    ("ChatGPT5", "Deception in code is art; uncovering it is science."),
    ("ChatGPT5", "Signatures die quickly; behavior endures."),
    ("ChatGPT5", "Reverse engineering is the archaeology of malicious intent."),
    ("ChatGPT5", "Entropy never lies — it whispers concealment."),
    ("ChatGPT5", "The best unpacker is understanding the packer’s logic."),
    ("ChatGPT5", "Automation scales, expertise refines."),
    ("ChatGPT5", "Machine learning fights malware, but explainability wins trust."),
    ("ChatGPT5", "Static bytes tell a story; dynamic traces tell the plot."),
    ("ChatGPT5", "A single opcode can betray a payload."),
    ("ChatGPT5", "Adversaries train models too — never forget that."),
    ("ChatGPT5", "Malware authors love obfuscation; analysts love patterns."),
    ("ChatGPT5", "Heuristics bridge the gap between rules and learning."),
    ("ChatGPT5", "Packing delays analysis, not detection."),
    ("ChatGPT5", "The debugger is your microscope into malicious life."),
    ("ChatGPT5", "Every ML model has blind spots — attackers find them first."),
    ("ChatGPT5", "Behavioral models age slower than signatures."),
    ("ChatGPT5", "Entropy spikes are often digital fingerprints of fear."),
    ("ChatGPT5", "Malware classification is a cat-and-mouse game of feature design."),
    ("ChatGPT5", "Dynamic unpacking is patience turned into insight."),
    ("ChatGPT5", "Machine learning needs context, not just data."),
    ("ChatGPT5", "Even obfuscation patterns can become features."),
    ("ChatGPT5", "Malware hides in code, but reveals itself in behavior."),
    ("ChatGPT5", "Packer layers are meant to confuse, not to conceal forever."),
    ("ChatGPT5", "Disassembly is the first step toward understanding digital deceit."),
    ("ChatGPT5", "The best analysts read binaries like novels — with curiosity and skepticism."),
    ("ChatGPT5", "Every malicious sample is a lesson in adversarial creativity."),
    ("ChatGPT5", "Machine learning without feature sanity is statistical guesswork."),
    ("ChatGPT5", "Polymorphism is evolution under pressure."),
    ("ChatGPT5", "Data drives models; intuition drives breakthroughs."),
    ("ChatGPT5", "Memory dumps tell truths binaries try to hide."),
    ("ChatGPT5", "Good analysis tools amplify insight, not replace it."),
    ("ChatGPT5", "The harder a binary resists unpacking, the more valuable the sample."),
    ("ChatGPT5", "Learning from malware requires both math and instinct."),
    ("ChatGPT5", "Adversarial AI is the next arms race in malware detection."),
    ("ChatGPT5", "Code similarity is the shadow of shared intent."),
    ("ChatGPT5", "Dynamic analysis reveals what obfuscation conceals."),
    ("ChatGPT5", "Malware families evolve faster than signature databases."),
    ("ChatGPT5", "A packed executable is a message wrapped in entropy."),
    ("ChatGPT5", "Machine learning models should explain, not just predict."),
    ("ChatGPT5", "Understanding malware means thinking like its creator."),
    ("ChatGPT5", "Every false positive is a reminder that precision matters as much as recall."),
    ("Google Gemini", "The 'Tail Jump' is the moment of transition where the unpacking stub cedes control to the original code."),
    ("Google Gemini", "The goal of a packer is to make the code unreadable on disk while remaining functional in memory."),
    ("Google Gemini", "If you see a file with high entropy and almost no imports, you are likely looking at a packed executable."),
    ("Google Gemini", "Packers were originally for compression; today, they are for survival."),
    ("Google Gemini", "The unpacking stub is a loader within a loader, a recursive trick to bypass static signatures."),
    ("Google Gemini", "Finding the Original Entry Point (OEP) is the primary objective of any manual unpacking exercise."),
    ("Google Gemini", "Modern packers don't just hide code; they virtualize it into a custom instruction set."),
    ("Google Gemini", "The Import Address Table (IAT) is the first thing a packer destroys and the last thing a reverse engineer must rebuild."),
    ("Google Gemini", "Packing is the art of making a file look like noise until the CPU starts singing."),
    ("Google Gemini", "UPX is the 'Hello World' of packing ; it teaches you the mechanics without the malice."),
    ("Google Gemini", "Hardware breakpoints are the scalpel used to find where the packer stops and the malware begins."),
    ("Google Gemini", "In a packed file, the data section is often the largest, hiding the dormant code in plain sight."),
    ("Google Gemini", "Anti-dumping techniques are the packer's way of saying: 'You can watch me run, but you can't take me home.'"),
    ("Google Gemini", "The entropy of a packed section often approaches the theoretical limit of 8.0 bits per byte."),
    ("Google Gemini", "Packing is a game of hide-and-seek played at the speed of instruction cycles."),
    ("Google Gemini", "The stub is the key that unlocks the binary, but the key usually destroys itself after use."),
    ("Google Gemini", "Self-modifying code is the heartbeat of a live unpacking process."),
    ("Google Gemini", "Generic unpacking relies on the fact that eventually, the CPU must see the original instructions."),
    ("Google Gemini", "A packer turns a surgical tool into a black box."),
    ("Google Gemini", "The transition from the unpacking stub to the OEP is the most vulnerable moment for any protected software."),
    ("Google Gemini", "Packing is the primary reason why static analysis is never enough."),
    ("Google Gemini", "A packer is a time-delay fuse for the logic inside an executable."),
    ("Google Gemini", "To defeat a packer, one must become comfortable with the chaos of the heap."),
    ("Google Gemini", "Multi-layer packing is like a Russian Matryoshka doll made of encrypted shellcode."),
    ("Google Gemini", "If VirtualAlloc or VirtualProtect are called repeatedly, the packer is likely building its stage."),
    ("Google Gemini", "The battle against packing is won in the debugger, not the disassembler."),
    ("Google Gemini", "Reverse engineering a packed file is the process of peeling back the layers of a developer's paranoia."),
    ("Perplexity", "Packing is the art of changing how an executable looks without changing what it does."),
    ("Perplexity", "Every packer adds a storyteller to your binary: the stub that must run before the real code can speak."),
    ("Perplexity", "If you do not understand the loader, you will never truly understand a packer."),
    ("Perplexity", "Unpacking is not about pressing a button; it is about reconstructing the original intent from layers of misdirection."),
    ("Perplexity", "Good packers compress bytes; great packers compress assumptions about how code should behave."),
    ("Perplexity", "A packed executable is a contract: the stub promises to restore the code, and the analyst promises to break that promise."),
    ("Perplexity", "The first breakpoint in a packed binary is not a line of code; it is the decision of where you think execution truly begins."),
    ("Perplexity", "Executable packing hides logic behind time, layers, and state instead of behind plain bytes."),
    ("Perplexity", "Static analysis ends where packing begins; dynamic analysis starts where the unpacking stub takes control."),
    ("Perplexity", "A packer is just a compiler whose only optimization goal is confusion."),
    ("Perplexity", "The strongest packing is not the one that is hardest to decompress but the one that makes you doubt what 'original' means."),
    ("Perplexity", "When you pack code, you trade CPU cycles and complexity for a thinner attack surface against casual inspection."),
    ("Perplexity", "Packed malware is not scary because it is hidden, but because defenders often stop at the first layer."),
    ("Perplexity", "Every unpacking routine is a roadmap back to the plaintext code if you are patient enough to follow it."),
    ("Perplexity", "Entropy tells you something is packed; control flow tells you how it unpacks."),
    ("Perplexity", "The real target of executable packing is not the CPU, it is the human reverse engineer."),
    ("Perplexity", "Virtualization-based packers do not hide instructions; they replace the instruction set with a new language you must learn."),
    ("Perplexity", "Anti-debug tricks in packers are not barriers, they are signposts saying 'you are getting close'."),
    ("Perplexity", "A universal unpacker is a myth; each packer family is its own dialect of obfuscation."),
    ("Perplexity", "Custom packers are often simpler than commercial ones, but the lack of documentation is their real protection."),
    ("Perplexity", "You can fingerprint many packers by their mistakes, not by their features."),
    ("Perplexity", "The original entry point of a packed executable is rarely where the story starts."),
    ("Perplexity", "Automated unpacking breaks when packers start to assume an adversarial sandbox."),
    ("Perplexity", "Packing does not make code secure; it just raises the minimum skill required to understand it."),
    ("Perplexity", "For defenders, packed executables are a reminder that signatures on disk are less useful than behavior in memory."),
    ("Perplexity", "The simplest unpacker is a debugger plus persistence."),
    ("Perplexity", "When analyzing a packed binary, your first goal is not to classify the packer but to recover a stable snapshot of the real code."),
    ("Perplexity", "Every layer of packing adds cost to both attacker and defender; the winner is whoever optimizes that cost better."),
    ("Perplexity", "Packing is a moving target because reverse engineering is a moving threat."),
    ("Perplexity", "If you learn to think like a packer, you will design much better unpacking strategies."),
    ("Microsoft Copilot", "The first rule of unpacking: trust the behavior, not the bytes."),
    ("Microsoft Copilot", "A packed binary is a puzzle that insists on being solved."),
    ("Microsoft Copilot", "When entropy spikes, curiosity should spike with it."),
    ("Microsoft Copilot", "Packer authors innovate; reverse engineers adapt."),
    ("Microsoft Copilot", "The unpacker’s mindset is simple: everything is reversible."),
    ("Microsoft Copilot", "Virtualization-based packers don’t hide code, they hide assumptions."),
    ("Microsoft Copilot", "A packer’s strength lies in misdirection, not magic."),
    ("Microsoft Copilot", "If the import table is empty, the fun is about to begin."),
    ("Microsoft Copilot", "Anti-debug tricks are just invitations to be clever."),
    ("Microsoft Copilot", "A packed executable is a story told backwards."),
    ("Microsoft Copilot", "The best unpackers are written in patience, not code."),
    ("Microsoft Copilot", "Compression hides structure; encryption hides intent."),
    ("Microsoft Copilot", "Every layer of packing is a layer of someone’s paranoia."),
    ("Microsoft Copilot", "The moment you see a stub, you know you’re not alone in the binary."),
    ("Microsoft Copilot", "A packer’s job is to confuse; yours is to stay curious."),
    ("Microsoft Copilot", "Dynamic analysis is the universal solvent for stubborn packers."),
    ("Microsoft Copilot", "A packer without anti-analysis is just a fancy zip file."),
    ("Microsoft Copilot", "The hallmark of a seasoned reverser is smiling at a TLS callback."),
    ("Microsoft Copilot", "Entropy doesn’t lie — it only whispers."),
    ("Microsoft Copilot", "Unpacking teaches humility; repacking teaches mastery."),
    ("Microsoft Copilot", "A custom packer is a signature in disguise."),
    ("Microsoft Copilot", "When the OEP feels impossible to find, you’re getting close."),
    ("Microsoft Copilot", "Packer evolution mirrors analyst evolution — endlessly iterative."),
    ("Microsoft Copilot", "The more opaque the binary, the brighter the analyst must think."),
    ("Microsoft Copilot", "In the end, every packer yields. The question is how much coffee it costs."),
    ("Claude", "Executable packing is the cloak of invisibility for software."),
    ("Claude", "To unpack a program is to reveal its hidden purpose."),
    ("Claude", "Packing can obfuscate the true intent of an executable."),
    ("Claude", "In reverse engineering, the unpacking process exposes the wizard behind the curtain."),
    ("Claude", "Every packed executable is a puzzle waiting to be solved."),
    ("Claude", "The art of unpacking is as crucial as the science behind packing."),
    ("Claude", "Intruders rarely distinguish between the original and the packed."),
    ("Claude", "Understanding executable packing is key to cybersecurity."),
    ("Claude", "A well-packed executable can mislead even the most seasoned analysts."),
    ("Claude", "Deconstructing layers is essential in revealing the end goal."),
    ("Claude", "Packing hides complexity; unpacking reveals it."),
    ("Claude", "Security through obscurity often hinges on effective packing."),
    ("Claude", "The first step in executable analysis is always unpacking."),
    ("Claude", "Sifting through bytes is where the true understanding begins."),
    ("Claude", "Effective packing can create a false sense of security."),
    ("Claude", "The code lies more under the surface than the pack."),
    ("Claude", "Every layer removed from a packed executable tells a story."),
    ("Claude", "Packing is a double-edged sword—it protects while it obscures."),
    ("Claude", "In the world of binaries, packing is both an art and a risk."),
    ("Claude", "Unpacking is not just a skill; it is an exploration."),
    ("Claude", "Executables packed with care are resilient against initial analyses."),
    ("Claude", "Behind every great executable is a packing strategy that shrouds it."),
    ("Claude", "Mastering executable unpacking is a rite of passage for reverse engineers."),
    ("Claude", "Packing techniques evolve, and so must our analysis skills."),
    ("Claude", "In the labyrinth of code, unpacking is the key to exit."),
    ("Claude", "The ability to unpack is synonymous with the ability to understand."),
    ("Claude", "An executable's packer can often speak louder than its code."),
    ("Claude", "The secrets of an executable lie hidden within its packed layers."),
    ("Claude", "Every packer has a weakness; uncovering it is the challenge."),
    ("Claude", "To pack is to protect, but to unpack is to empower."),
    ("Claude", "Reverse engineering is a dance between creation and decomposition."),
    ("Claude", "Executable packing is a veil, obscuring the intent behind the code."),
    ("Alexandre D'Hondt", "Executable packing involves transformations that modify a binary file that preserve its semantics while having its layout changed at rest."),
    ("Alexandre D'Hondt", "Contrary to what is often stated in the literature, executable packing is not only a matter of compression or encryption."),
    ("Michael Sikorski", "Malware analysis is like a cat-and-mouse game."),
    ("Michael Sikorski", "The most common way to hide a program’s purpose is to use a packer."),
    ("Michael Sikorski", "One problem is that malware writers can easily modify their code, thereby changing their program’s signature and evading virus scanners."),
    ("Michael Sikorski", "Packed and obfuscated malware contains very few strings.",),
    ("Michael Sikorski", "Basic static techniques are like looking at the outside of a body during an autopsy."),
    ("Michael Sikorski", "An early step when analyzing malware is to recognize that it is packed."),
    ("Andrew Honig", "A packer is essentially a wrapper around a program that compresses or encrypts it."),
    ("Ilfak Guilfanov", "A packer is only as strong as the analyst who refuses to give up."),
    ("Chris Eagle", "Packing hides intent, but never erases it."),
    ("Bruce Dang", "Every packer leaves a fingerprint; your job is to learn its handwriting."),
    ("Dennis Yurichev", "Unpacking is the art of turning chaos back into logic."),
    ("Alex Sotirov", "A good packer delays analysis; a great one teaches you patience."),
]

__f = AsciiFile()
__f['title', {'fgcolor': "lolcat"}] = Banner("Packing-Box Console", font="cyberlarge")
__f['myquote', {'adjust': "right"}] = Quote(*random.choice(__QUOTES)[::-1])
print(__f)

