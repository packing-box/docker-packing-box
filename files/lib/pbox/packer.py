#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from tinyscript import b, ensure_str, hashlib, logging, os, random, shlex, shutil, subprocess, ts
from tinyscript.helpers import execute_and_log as run, Path
from tinyscript.report import *


__all__ = ["exe_format", "Packer", "PACKERS"]  # this list is filled in with packer subclasses at the end of this module


CATEGORIES = {
    'All':    ["ELF", "Mach-O", "MSDOS", "PE"],
    'ELF':    ["ELF32", "ELF64"],
    'Mach-O': ["Mach-O32", "Mach-O64", "Mach-Ou"],
    'PE':     ["PE32", "PE64"],
}
OS_COMMANDS = subprocess.check_output("compgen -c", shell=True, executable="/bin/bash").splitlines()
SIGNATURES = {
    '^Mach-O 32-bit ':                         "Mach-O32",
    '^Mach-O 64-bit ':                         "Mach-O64",
    '^Mach-O universal binary ':               "Mach-Ou",
    '^MS-DOS executable ':                     "MSDOS",
    '^PE32\+? executable (.+?)\.Net assembly': ".NET",
    '^PE32 executable ':                       "PE32",
    '^PE32\+ executable ':                     "PE64",
    '^(set[gu]id )?ELF 32-bit ':               "ELF32",
    '^(set[gu]id )?ELF 64-bit ':               "ELF64",
}


def exe_format(*categories):
    categories = Packer.expand(*categories)
    def _wrapper(executable):
        best_fmt, l = None, 0
        for ftype, fmt in SIGNATURES.items():
            if fmt in categories and len(ftype) > l and ts.is_filetype(str(executable), ftype):
                best_fmt, l = fmt, len(ftype)
        return best_fmt
    return _wrapper


class Packer:
    """ Packer abstraction, suitable for subclassing and redefining its .cmd() method for shaping its command line. """
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        self.logger = logging.getLogger(self.name)
        self._bad = False
        self._error = False
    
    def check(self, *categories):
        """ Checks if the current packer has the given category """
        return any(c in Packer.expand(*self.categories) for c in Packer.expand(*(categories or ["All"])))
    
    def help(self):
        """ Returns a help message in Markdown format. """
        md = Report()
        if getattr(self, "description", None):
            md.append(Text(self.description))
        if getattr(self, "comment", None):
            md.append(Blockquote("**Note**: " + self.comment))
        md.append(Text("**Source**: " + self.source))
        md.append(Text("**Applies to**: " + ", ".join(sorted(Packer.expand(*self.categories, **{'once': True})))))
        if getattr(self, "references", None):
            md.append(Section("References"), List(*self.references, **{'ordered': True}))
        return md.md()
    
    def pack(self, executable, **kwargs):
        """ Runs the packer according to its command line format and checks if the executable has been changed by this
             execution. """
        logging.setLogger(self.name)
        exe = executable.split(os.sep)[-1]
        # check 1: is this packer operational ?
        if self.status <= 2:
            return False
        # check 2: is this packer able to process the input executable ?
        if exe_format(*self.categories)(executable) is None:
            return False
        # now pack the input executable, taking its SHA256 in order to check for changes
        s256 = hashlib.sha256_file(executable)
        self._error = False
        label = self.run(executable, **kwargs)
        if self._error:
            return
        elif s256 == hashlib.sha256_file(executable):
            self.logger.debug("%s's content was not changed by %s" % (exe, self.name))
            self._bad = True
            return
        # if packing succeeded, we can return packer's label
        self.logger.debug("%s packed with %s" % (exe, self.name))
        return label
    
    def run(self, executable, **kwargs):
        """ Customizable method for shaping the command line to run the packer on an input executable. """
        if run([self.name, executable], logger=self.logger)[-1] > 0:
            self._error = True
        return self.name
    
    def setup(self):
        """ Sets the packer up according to its install instructions. """
        if self.status == 0:
            return
        logging.setLogger(self.name)
        self.logger.info("Setting up %s..." % self.__class__.__name__)
        tmp, ubin = Path("/tmp"), Path("/usr/bin")
        result, rm, kw = None, True, {'logger': self.logger}
        cwd = os.getcwd()
        for cmd, arg in self.install.items():
            if isinstance(result, Path) and not result.exists():
                raise ValueError("Last command's result does not exist (%s) ; current: %s" % (result, cmd))
            # simple install through APT
            if cmd == "apt":
                run("apt -qy install %s" % arg, **kw)
            # change to the given dir (starting from the reference /tmp directory if no command was run before)
            elif cmd == "cd":
                result = (result or tmp).joinpath(arg)
                self.logger.debug("cd %s" % result)
                os.chdir(str(result))
            # copy a file from the previous location (or /tmp if not defined) to /usr/bin,
            #  making the destination file executable
            elif cmd == "copy":
                try:
                    arg1, arg2 = shlex.split(arg)
                except ValueError:
                    arg1, arg2 = arg, arg
                src, dst = (result or tmp).joinpath(arg1), ubin.joinpath(arg2)
                if run("cp %s %s" % (src, dst), **kw)[-1] == 0 and arg1 == arg2:
                    run("chmod +x %s" % dst, **kw)
            # execute the given command as is, with no pre-/post-condition, not altering the result state variable
            elif cmd == "exec":
                result = None
                run(arg, **kw)
            # git clone a project
            elif cmd == "git":
                result = (result or tmp).joinpath(Path(ts.urlparse(arg).path).stem)
                run("git clone %s %s" % (arg, result), **kw)
            # make a symbolink link in /usr/bin (if relative path) relatively to the previous considered location
            elif cmd == "ln":
                r = ubin.joinpath(self.name)
                run("ln -s %s %s" % (result.joinpath(arg), r), **kw)
                result = r
            elif cmd == "lsh":
                try:
                    arg1, arg2 = shlex.split(arg)
                except ValueError:
                    arg1, arg2 = "/opt/packers/%s" % self.name, arg
                result = ubin.joinpath(self.name)
                arg = "#!/bin/bash\nPWD=`pwd`\nif [[ \"$1\" = /* ]]; then TARGET=\"$1\"; else TARGET=\"$PWD/$1\"; fi" \
                      "\ncd %s\n./%s $TARGET $2\ncd $PWD" % (arg1, arg2)
                self.logger.debug("echo -en '%s' > %s" % (arg, result))
                try:
                    result.write_text(arg)
                    run("chmod +x %s" % result, **kw)
                except PermissionError:
                    self.logger.error("bash: %s: Permission denied" % result)
            # compile a C project
            elif cmd == "make":
                if not result.is_dir():
                    raise ValueError("Got a file ; should have a folder")
                files = [x.filename for x in result.listdir()]
                if "Makefile" in files:
                    if "configure.sh" in files:
                        if run("chmod +x configure.sh", **kw)[-1] == 0:
                            run("./configure.sh", **kw)
                    if run("make", **kw)[-1] == 0:
                        run("make install", **kw)
                elif "make.sh" in files:
                    if run("chmod +x make.sh", **kw)[-1] == 0:
                        run("sh -c './make.sh'", **kw)
                result = result.joinpath(arg)
            # move the previous location to /usr/bin (if relative path), make it executable if it is a file
            elif cmd == "move":
                if result is None:
                    result = tmp.joinpath("%s.*" % self.name)
                r = ubin.joinpath(arg)
                if run("mv %s %s" % (result, r), **kw)[-1] == 0 and r.is_file():
                    run("chmod +x %s" % r, **kw)
                result = r
            # remove a given directory (then bypassing the default removal at the end of all commands)
            elif cmd == "rm":
                run("rm -rf %s" % Path(arg), **kw)
                rm = False
            # manually set the result to be used in the next command
            elif cmd == "set":
                result = arg
            # manually set the result as a path object to be used in the next command
            elif cmd == "setp":
                result = tmp.joinpath(arg)
            # create a shell script to execute Bash code and make it executable
            elif cmd == "sh":
                result = ubin.joinpath(self.name)
                arg = "\n".join(arg.split("\\n"))
                arg = "#!/bin/bash\n%s" % arg
                self.logger.debug("echo -en '%s' > %s" % (arg, result))
                try:
                    result.write_text(arg)
                    run("chmod +x %s" % result, **kw)
                except PermissionError:
                    self.logger.error("bash: %s: Permission denied" % result)
            # decompress a RAR/ZIP archive to the given location (absolute or relative to /tmp)
            elif cmd in ["unrar", "untar", "unzip"]:
                ext = "." + cmd[-3:]
                if ext == ".tar":
                    ext = ".tar.gz"
                if result is None:
                    result = tmp.joinpath("%s%s" % (self.name, ext))
                if result and result.extension == ext:
                    r = tmp.joinpath(arg)
                    if ext == ".zip":
                        run("unzip -qqo %s -d %s" % (result, r), **kw)
                    elif ext == ".rar":
                        if not r.exists():
                            r.mkdir()
                        run("unrar x %s %s" % (result, r), **kw)
                    else:
                        run("mkdir -p %s" % r, **kw)
                        run("tar xzf %s -C %s" % (result, r), **kw)
                    result.remove()
                    result = r
                else:
                    raise ValueError("Not a %s file" % ext.lstrip(".").upper())
                if result and result.is_dir():
                    ld = list(result.listdir())
                    while len(ld) == 1 and ld[0].is_dir():
                        result = ld[0]
                        ld = list(result.listdir())
            # download a resource, possibly downloading 2-stage generated download links (in this case, the list is
            #  handled by downloading the URL from the first element then matching the second element in the URL's found
            #  in the downloaded Web page
            elif cmd == "wget":
                # (2-stage) dynamic download link
                rc = 0
                if isinstance(arg, list):
                    url = arg[0].replace("%%", "%")
                    for line in run("wget -qO - %s" % url, **kw)[0].splitlines():
                        line = line.decode()
                        m = re.search(r"href\s+=\s+(?P<q>[\"'])(.*)(?P=q)", line)
                        if m is not None:
                            url = m.group(1)
                            if Path(ts.urlparse(url).path).stem == (arg[1] if len(arg) > 1 else self.name):
                                break
                            url = arg[0]
                    if url != arg[0]:
                        result = tmp.joinpath(p + Path(ts.urlparse(url).path).extension)
                        run("wget -q -O %s %s" % (result, url), **kw)[-1]
                # normal link
                else:
                    result = tmp.joinpath(self.name + Path(ts.urlparse(arg).path).extension)
                    run("wget -q -O %s %s" % (result, arg.replace("%%", "%")), **kw)[-1]
        if os.getcwd() != cwd:
            self.logger.debug("cd %s" % cwd)
            os.chdir(cwd)
        if rm:
            run("rm -rf %s" % tmp.joinpath(self.name), **kw)
    
    @property
    def status(self):
        """ Get the status of packer's executable. """
        st = getattr(self, "_status", None)
        if st == "broken":  # manually set in packers.yml ; for packers that cannot be installed or run
            return 0
        elif st in ["gui", "todo"]:  # packer to be automated yet
            return 2
        elif b(self.name) not in OS_COMMANDS:  # when the setup failed
            return 1
        elif st == "ok":  # the packer runs, works correctly and was tested
            return 4
        # in this case, the binary/symlink exists but running the packer was not tested yet
        return 3
    
    @staticmethod
    def expand(*categories, **kw):
        """ 2-depth dictionary-based expansion function for resolving a list of executable categories. """
        selected = []
        for c in categories:                    # depth 1: e.g. All => ELF,PE OR ELF => ELF32,ELF64
            for sc in CATEGORIES.get(c, [c]):   # depth 2: e.g. ELF => ELF32,ELF64
                if kw.get('once', False):
                    selected.append(sc)
                else:
                    for ssc in CATEGORIES.get(sc, [sc]):
                        if ssc not in selected:
                            selected.append(ssc)
        return selected
    
    @staticmethod
    def get(packer):
        """ Simple method for returning the class of a packer based on its name (case-insensitive). """
        for p in PACKERS:
            if p.name == packer.lower():
                return p

# ------------------------------------------------ NON-STANDARD PACKERS ------------------------------------------------
class Amber(Packer):
    def run(self, executable, **kwargs):
        """ This packer packs an executable to a new predictible name. """
        out, err, retc = run([self.name, "-build", "-f", executable], logger=self.logger)
        self._error = retc != 0
        executable = Path(executable)
        executable2 = executable.dirname.joinpath("%s_packed%s" % (executable.stem, executable.extension))
        if not executable2.exists():
            return
        run(["mv", "-f", executable2, executable], logger=self.logger)
        return self.name


class Ezuri(Packer):
    key = None
    iv  = None
    
    def run(self, executable, **kwargs):
        """ This packer prompts for parameters. """
        P = subprocess.PIPE
        p = subprocess.Popen(["ezuri"], stdout=P, stderr=P, stdin=P)
        executable = Path(executable)
        self.logger.debug("ezuri ; inputs: src/dst=%s, procname=%s" % (executable, executable.stem))
        out, err = p.communicate(b("%(e)s\n%(e)s\n%(n)s\n%(k)s\n%(iv)s\n" % {
            'e': executable, 'n': executable.stem,
            'k': "" if Ezuri.key is None else Ezuri.key,
            'iv': "" if Ezuri.iv is None else Ezuri.iv,
        }))
        for l in out.splitlines():
            l = ensure_str(l)
            if not l.startswith("[?] "):
                self.logger.debug(l)
            if Ezuri.key is None and "Random encryption key (used in stub):" in l:
                Ezuri.key = l.split(":", 1)[1].strip()
            if Ezuri.iv is None and "Random encryption IV (used in stub):" in l:
                Ezuri.iv = l.split(":", 1)[1].strip()
        if err:
            self.logger.error(ensure_str(err.strip()))
            self._error = True
        return "%s[key:%s;iv:%s]" % (self.name, Ezuri.key, Ezuri.iv)


class MidgetPack(Packer):
    def run(self, executable, **kwargs):
        """ This packer prompts for a protection password (with confirmation). """
        password = random.randstr()
        P = subprocess.PIPE
        p = subprocess.Popen(["midgetpack", executable], stdout=P, stderr=P, stdin=P)
        self.logger.debug("midgetpack ; inputs: password=%s" % password)
        out, err = p.communicate(b("%(p)s\n%(p)s\n" % {'p': password}))
        packed = Path(executable).parent.joinpath("a.out")
        if packed.exists():
            run("mv -f %s %s" % (packed, executable))
            return "%s[pswd:%s]" % (self.name, password)
        self._error = True


class M0dern_P4cker(Packer):
    stubs = ["xor", "not", "xorp"]
    
    def run(self, executable, **kwargs):
        """ This packer allows to define 3 different stubs: XOR, NOT, XORP (see documentation). """
        stubs = self.stubs[:]
        self._error = True
        while len(stubs) > 0 and self._error:
            stub = random.choice(stubs)
            out, err, retc = run([self.name, executable, stub], logger=self.logger)
            for l in ensure_str(out).splitlines():
                if "Stub Injected" in l:
                    self._error = False
                    return "%s[%s]" % (self.name, stub)
            stubs.remove(stub)
# ----------------------------------------------------------------------------------------------------------------------

# dynamically makes Packer child classes from the PACKERS dictionary
PACKERS = []
for packer, data in ts.yaml_config(str(Path("/opt/packers/packers.yml"))).items():
    if packer not in globals():
        p = globals()[packer] = type(packer, (Packer, ), dict(Packer.__dict__))
    else:
        p = globals()[packer]
    for k, v in data.items():
        if k == "status":
            k = "_" + k
        setattr(p, k, v)
    __all__.append(packer)
    PACKERS.append(p())
