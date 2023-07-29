# -*- coding: UTF-8 -*-
from os.path import expanduser
from tinyscript import b, colored as _c, ensure_str, json, logging, os, random, re, shlex, subprocess, sys, ts
from tinyscript.helpers import execute_and_log as run, execute as run2, lazy_object, Path
from tinyscript.report import *

from ..core.config import *
from ..core.executable import Executable
from ..core.item import _init_item, update_logger
from ..helpers import *


__all__ = ["Base"]


# for a screenshot: "xwd -display $DISPLAY -root -silent | convert xwd:- png:screenshot.png"
GUI_SCRIPT = """#!/bin/bash
source ~/.bash_xvfb
{{preamble}}
SRC="$1"
NAME="$(basename "$1" | sed 's/\(.*\)\..*/\1/')"
DST="$HOME/.wine%(arch)s/drive_c/users/user/Temp/${1##*/}"
FILE="c:\\\\users\\\\user\\\\Temp\\\\${1##*/}"
cp -f "$SRC" "$DST"
WINEPREFIX=\"$HOME/.wine%(arch)s\" WINEARCH=win%(arch)s wine "$EXE" &
sleep .5
{{actions}}
ps -eaf | grep -v grep | grep -E -e "/bin/bash.+bin/$NAME" -e ".+/$NAME\.exe\$" \
                                 -e 'bin/wineserver$' -e 'winedbg --auto' \
                                 -e 'windows\\system32\\services.exe$' \
                                 -e 'windows\\system32\\conhost.exe --unix' \
                                 -e 'windows\\system32\\explorer.exe /desktop$' \
        | awk {'print $2'} | xargs kill -9
sleep .1
mv -f "$DST" "$SRC"{{postamble}}
"""
OS_COMMANDS = subprocess.check_output("compgen -c", shell=True, executable="/bin/bash").splitlines()
ERR_PATTERN = r"^\x07?\s*(?:\-\s*)?(?:\[(?:ERR(?:OR)?|\!)\]|ERR(?:OR)?\:)\s*"
PARAM_PATTERN = r"{{([^\{\}]*?)(?:\[([^\{\[\]\}]*?)\])?}}"
STATUS_DISABLED = ["broken", "commercial", "info", "useless"]
STATUS_ENABLED = lazy_object(lambda: [s for s in STATUS.keys() if s not in STATUS_DISABLED + ["not installed"]])
TEST_FILES = {
    'ELF32': [
        "/usr/bin/perl",
        "/usr/lib/wine/wine",
        "/usr/lib/wine/wineserver32",
        "/usr/libx32/crti.o",
        "/usr/libx32/libpcprofile.so",
    ],
    'ELF64': [
        "/bin/cat",
        "/bin/ls",
        "/bin/mandb",
        "/usr/lib/openssh/ssh-keysign",
        "/usr/lib/git-core/git",
        "/usr/lib/x86_64-linux-gnu/crti.o",
        "/usr/lib/x86_64-linux-gnu/libpcprofile.so",
        "/usr/lib/ld-linux.so.2",
    ],
    'MSDOS': [
        "~/.wine32/drive_c/windows/rundll.exe",
        "~/.wine32/drive_c/windows/system32/gdi.exe",
        "~/.wine32/drive_c/windows/system32/user.exe",
        "~/.wine32/drive_c/windows/system32/mouse.drv",
        "~/.wine32/drive_c/windows/system32/winaspi.dll",
    ],
    'PE32': [
        "~/.wine32/drive_c/windows/winhlp32.exe",
        "~/.wine32/drive_c/windows/system32/plugplay.exe",
        "~/.wine32/drive_c/windows/system32/winemine.exe",
        "~/.wine32/drive_c/windows/twain_32.dll",
        "~/.wine32/drive_c/windows/twain_32/sane.ds",
        "~/.wine32/drive_c/windows/system32/msscript.ocx",
        "~/.wine32/drive_c/windows/system32/msgsm32.acm",
    ],
    'PE64': [
        "~/.wine64/drive_c/windows/hh.exe",
        "~/.wine64/drive_c/windows/system32/spoolsv.exe",
        "~/.wine64/drive_c/windows/system32/dmscript.dll",
        "~/.wine64/drive_c/windows/twain_64/gphoto2.ds",
        "~/.wine64/drive_c/windows/system32/msscript.ocx",
        "~/.wine64/drive_c/windows/system32/msadp32.acm",
    ],
}


def _init_base():
    Item = _init_item()
    class Base(Item):
        """ Base item abstraction, for defining the common machinery for Detector, Packer and Unpacker.
        
        Instance methods:
          .check(*formats) [bool]
          .help() [str]
          .run(executable, **kwargs) [str|(str,time)]
          .setup(**kwargs)
          .test(files, **kwargs)
        
        Class methods:
          .get(item)
          .summary()
        """
        _enabled = STATUS_ENABLED
        
        def __init__(self):
            super(Base, self).__init__()
            self._formats_exp = expand_formats(*self.formats)
            self._bad = False
            self._error = None
            self._params = {}
            self.__cwd = None
            self.__init = False
        
        def __expand(self, line):
            return line.replace("$TMP", "/tmp/%ss" % self.type) \
                       .replace("$OPT", expanduser("~/.opt/%ss" % self.type)) \
                       .replace("$BIN", expanduser("~/.opt/bin")) \
                       .replace("$CWD", self.__cwd or "")
        
        @update_logger
        def _check(self, exe, silent=False):
            """ Check if the given executable can be processed by this item. """
            fmt = exe.format
            exe = Executable(exe)
            ext = exe.extension[1:]
            exc = getattr(self, "exclude", [])
            if isinstance(exc, dict):
                l = []
                for k, v in exc.items():
                    if fmt in expand_formats(k):
                        l.extend(v)
                exc = list(set(l))
            if not exe.exists():
                if not silent:
                    self.logger.warning("'%s' does not exist" % exe)
                return False
            if fmt not in self._formats_exp:
                if not silent:
                    self.logger.debug("does not handle %s executables" % fmt if fmt is not None else \
                                      "does not handle '%s'" % exe.filetype)
                return False
            for p in exc:
                if isinstance(p, dict):
                    disp, patt = list(p.items())[0]
                    if re.search(patt, exe.filetype):
                        if not silent:
                            self.logger.debug("does not handle %s files" % disp)
                        return False
            if ext in [_ for _ in exc if not isinstance(_, dict)]:
                if not silent:
                    self.logger.debug("does not handle .%s files" % ext)
                return False
            return True
        
        def _gui(self, target, arch, local=False):
            """ Prepare GUI script. """
            preamble = "PWD='`pwd`'\ncd '%s'\nEXE='%s'" % (target.dirname, target.filename) if local else \
                       "EXE='%s'" % target
            postamble = ["", "\ncd '$PWD'"][local]
            script = (GUI_SCRIPT % {'arch': arch}).replace("{{preamble}}", preamble).replace("{{postamble}}", postamble)
            cmd = re.compile(r"^(.*?)(?:\s+\((\d*\.\d*|\d+)\)(?:|\s+\[x(\d+)\])?)?$")
            slp = re.compile(r"sleep-per-(B|KB|MB)\s+(\d*\.\d*|\d+)")
            actions = []
            for action in self.gui:
                c, delay, repeat = cmd.match(action).groups()
                for i in range(int(repeat or 1)):
                    if re.match("(behave|click|get(_|mouselocation)|key(down|up)?|mouse(down|move(_relative)?|up)|search|"
                                "set_|type|(get|select)?window)", c):
                        m = re.match("click (\d{1,5}) (\d{1,5})$", c)
                        if m is not None:
                            c = "mousemove %s %s click" % m.groups()
                        c = "xdotool %s" % c
                        if c.endswith(" click") or c == "click":
                            c += " 1"  # set to left-click
                    elif slp.match(c):
                        s, t = slp.match(c).groups()
                        bs = [" --block-size=1%s" % s[0], ""][s == "B"]
                        c = "sleep $(bc <<< \"`ls -l%s $DST | cut -d' ' -f5`*%s\")" % (bs, t)
                    actions.append(c)
                    if delay:
                        actions.append("sleep %s" % delay)
            return script.replace("{{actions}}", "\n".join(actions))
        
        @update_logger
        def _test(self, silent=False):
            """ Preamble to the .test(...) method for validation and log purpose. """
            if self.status in STATUS_DISABLED + ["not installed"]:
                self.logger.warning("%s is %s" % (self.cname, self.status))
                return False
            logging.setLogger(self.name)
            if not silent:
                self.logger.info("Testing %s..." % self.cname)
            return True
        
        def check(self, *formats, **kwargs):
            """ Checks if the current item is applicable to the given formats. """
            silent = kwargs.get('silent', True)
            if self.status == "ok":
                if any(c in self._formats_exp for c in expand_formats(*(formats or ["All"]))):
                    return True
                if not silent:
                    self.logger.debug("does not apply to the selected formats")
                return False
            if not silent:
                self.logger.debug("disabled (status: %s)" % self.status)
            return False
        
        def help(self, extras=None):
            """ Returns a help message in Markdown format. """
            md, bn = Report(), self.__class__.__base__.__name__
            if getattr(self, "description", None):
                md.append(Text(self.description))
            if getattr(self, "comment", None):
                md.append(Blockquote("**Note**: " + self.comment))
            md.append(Text("**Source**    : " + self.source))
            md.append(Text("**Applies to**: " + ", ".join(sorted(expand_formats(*self.formats, **{'once': True})))))
            if bn == "Packer" and getattr(self, "alterations", None):
                md.append(Text("**Alterations** : " + ", ".join(self.alterations)))
            if bn == "Unpacker" and getattr(self, "packers", None):
                md.append(Text("**Can unpack**: " + ", ".join(self.packers)))
            for k, v in (extras or {}).items():
                md.append(Text("**%-10s**: " % k.capitalize() + v))
            if getattr(self, "references", None):
                md.append(Section("References"), List(*self.references, **{'ordered': True}))
            return md.md()
        
        @update_logger
        def run(self, executable, **kwargs):
            """ Customizable method for shaping the command line to run the item on an input executable. """
            retval = self.name
            use_output = False
            benchm, verb = kwargs.get('benchmark', False), kwargs.get('verbose', False) and getattr(self, "verbose", False)
            binary, weak = kwargs.get('binary', False), kwargs.get('weak', False)
            extra_opt = "" if kwargs.get('extra_opt') is None else kwargs['extra_opt'] + " "
            kw = {'logger': self.logger, 'silent': [], 'timeout': config['exec_timeout'], 'reraise': True}
            if not config['wine_errors']:
                kw['silent'].append(r"^[0-9a-f]{4}\:(?:err|fixme)\:")
            # local function for replacing tokens within the string lists, i.e. "silent" and "steps"
            _str = lambda o: "'" + str(o) + "'"
            def _repl(s, strf=lambda x: str(x)):
                return s.replace("{{executable}}", strf(executable)) \
                        .replace("{{executable.dirname}}", strf(executable.dirname)) \
                        .replace("{{executable.filename}}", strf(executable.filename)) \
                        .replace("{{executable.extension}}", strf(executable.extension)) \
                        .replace("{{executable.stem}}", strf(executable.dirname.joinpath(executable.stem)))
            kw['silent'].extend(list(map(_repl, getattr(self, "silent", []))))
            output, self.__cwd = "", os.getcwd()
            for step in getattr(self, "steps", ["%s %s%s" % (self.name.replace("_" ,"-"), extra_opt, _str(executable))]):
                step = self.__expand(step)
                if self.name in step:
                    i, opt = step.index(self.name), ""
                    if benchm:
                        opt += " --benchmark"
                    if binary:
                        opt += " --binary"
                    if weak:
                        opt += " --weak"
                    step = step[:i] + self.name + opt + step[i+len(self.name):]
                attempts = []
                # first, replace generic patterns
                step = _repl(step, _str)
                # now, replace a previous output and handle it as the return value
                if "{{output}}" in step:
                    step = step.replace("{{output}}", output)
                    use_output = True
                # then, search for parameter patterns
                m = re.search(PARAM_PATTERN, step)
                if m:
                    name, values = m.groups()
                    values = self._params.get(name, (values or "").split("|"))
                    for value in values:
                        disp = "%s=%s" % (name, value)
                        if len(values) == 2 and "" in values:
                            disp = "" if value == "" else name
                        attempts.append((re.sub(PARAM_PATTERN, value, step), disp))
                # now, run attempts for this step in random order until one succeeds
                random.shuffle(attempts)
                attempts = attempts or [step]
                for attempt in attempts[:]:
                    param = None
                    if isinstance(attempt, tuple) and len(attempt) == 2:
                        attempt, param = attempt
                    if attempt.startswith("cd "):
                        self.logger.debug(attempt)
                        os.chdir(attempt[3:].strip("'"))
                        continue
                    if verb:
                        attempt += " -v"
                    kw['shell'] = ">" in attempt
                    try:
                        output, error, retc = run(attempt, **kw)
                    except Exception as e:
                        self.logger.error(str(e))
                        output, error, retc = None, str(e), 1
                    output = ensure_str(output or NOT_LABELLED).strip()
                    # filter out error lines from stdout
                    out_err = "\n".join(re.sub(ERR_PATTERN, "", l) for l in output.splitlines() if re.match(ERR_PATTERN, l))
                    output  = "\n".join(l for l in output.splitlines() if not re.match(ERR_PATTERN, l) and l.strip() != "")
                    # update error string obtained from stderr
                    self._error = "\n".join(l for l in error.splitlines() \
                                            if all(re.search(p, l) is None for p in kw.get('silent', [])))
                    self._error = (self._error + "\n" + out_err).strip()
                    if self.name in attempt and benchm:
                        output, dt = shlex.split(output.splitlines()[-1])
                    if retc > 0:
                        if verb:
                            attempt = attempt.replace(" -v", "")
                        attempts.remove(attempt if param is None else (attempt, param))
                        if len(attempts) == 0:
                            return NOT_LABELLED
                    else:
                        if param:
                            retval += "[%s]" % param
                        break
            os.chdir(self.__cwd)
            r = output if use_output or getattr(self, "use_output", False) else retval
            if benchm:
                r = (r, dt)
            return r
        
        @update_logger
        def setup(self, **kw):
            """ Sets the item up according to its install instructions. """
            logging.setLogger(self.name)
            if self.status in STATUS_DISABLED:
                self.logger.info("Skipping %s..." % self.cname)
                self.logger.debug("Status: %s ; this means that it won't be installed" % self.status)
                return
            self.logger.info("Setting up %s..." % self.cname)
            opt, tmp = Path("~/.opt/%ss" % self.type, expand=True), Path("/tmp/%ss" % self.type)
            obin, ubin = Path("~/.opt/bin", create=True, expand=True), Path("~/.local/bin", create=True, expand=True)
            result, rm, wget = None, True, False
            self.__cwd = os.getcwd()
            for cmd in self.install:
                if isinstance(result, Path) and not result.exists():
                    self.logger.critical("Last command's result does not exist (%s) ; current: %s" % (result, cmd))
                    return
                kw = {'logger': self.logger, 'silent': getattr(self, "silent", [])}
                name = cmd.pop('name', "")
                if name != "":
                    self.logger.info("> " + name)
                # extract argument(s)
                cmd, arg = list(cmd.items())[0]
                try:
                    arg = self.__expand(arg)
                    arg1, arg2 = shlex.split(arg)
                except AttributeError:
                    arg = [self.__expand(a) for a in arg]  # e.g. 'exec' can handle 'arg' as a list of commands
                except ValueError:
                    arg1, arg2 = arg, arg
                # simple install through APT
                if cmd == "apt":
                    run("sudo apt-get -qqy install %s" % arg, **kw)
                # change to the given dir (starting from the reference /tmp/[ITEM]s directory if no command was run before)
                elif cmd == "cd":
                    result = (result or tmp).joinpath(arg)
                    if not result.exists():
                        self.logger.debug("mkdir '%s'" % result)
                        result.mkdir()
                    self.logger.debug("cd '%s'" % result)
                    os.chdir(str(result))
                # add the executable flag on the target
                elif cmd == "chmod":
                    run("chmod +x '%s'" % result.joinpath(arg), **kw)
                # copy a file from the previous location (or /tmp/[ITEM]s if not defined) to ~/.opt/bin, making the
                #  destination file executable
                elif cmd == "copy":
                    base = result or tmp
                    if arg1 == arg2:  # only the destination is provided
                        src, dst = base if base.is_file() else base.joinpath(self.name), ubin.joinpath(arg1)
                    else:             # both the source and destination are provided
                        src, dst = (base.dirname if base.is_file() else base).joinpath(arg1), ubin.joinpath(arg2)
                    dst.dirname.mkdir(exist_ok=True)
                    # if ~/.local/bin/... exists, then save it to ~/.opt/bin/... to superseed it
                    if dst.is_samepath(ubin.joinpath(arg2)) and dst.exists():
                        dst = obin.joinpath(arg2)
                    if run("cp %s'%s' '%s'" % (["", "-r "][src.is_dir()], src, dst), **kw)[-1] == 0 and dst.is_file():
                        run("chmod +x '%s'" % dst, **kw)
                    if arg1 == self.name:
                        rm = False
                    result = dst if dst.is_dir() else dst.dirname
                # execute the given shell command or the given list of shell commands
                elif cmd == "exec":
                    result = None
                    if not isinstance(arg, list):
                        arg = [arg]
                    for a in arg:
                        run(a, **kw)
                # git clone a project
                elif cmd in ["git", "gitr"]:
                    result = (result or tmp).joinpath(Path(ts.urlparse(arg).path).stem.lower() if arg1 == arg2 else arg2)
                    result.remove(False)
                    run("git clone --quiet %s%s '%s'" % (["", "--recursive "][cmd == "gitr"], arg1, result), **kw)
                # go build the target
                elif cmd == "go":
                    result = result if arg2 == arg else Path(arg2, expand=True)
                    cwd2 = os.getcwd()
                    os.chdir(str(result))
                    #result.joinpath("go.mod").remove(False)
                    run("go mod init '%s'" % arg1,
                        silent=["already exists", "creating new go.mod", "add module requirements", "go mod tidy"])
                    run("go build -o %s ." % self.name, silent=["downloading"])
                    os.chdir(cwd2)
                # create a shell script to execute the given target with its intepreter/launcher and make it executable
                elif cmd in ["java", "mono", "sh", "wine", "wine64"]:
                    r, txt, tgt = ubin.joinpath(self.name), "#!/bin/bash\n", (result or opt).joinpath(arg)
                    if cmd == "java":
                        txt += "java -jar \"%s\" \"$@\"" % tgt
                    elif cmd == "mono":
                        txt += "mono \"%s\" \"$@\"" % tgt
                    elif cmd == "sh":
                        txt += "\n".join(arg.split("\\n"))
                    elif cmd.startswith("wine"):
                        arch = ["32", "64"][cmd == "wine64"]
                        if hasattr(self, "gui"):
                            txt = self._gui(tgt, arch)
                        else:
                            txt += "WINEPREFIX=\"$HOME/.wine%s\" WINEARCH=win%s wine \"%s\" \"$@\"" % (arch, arch, tgt)
                    self.logger.debug("echo -en '%s' > '%s'" % (txt, r))
                    try:
                        r.write_text(txt)
                        run("chmod +x '%s'" % r, **kw)
                    except PermissionError:
                        self.logger.error("bash: %s: Permission denied" % r)
                    result = r
                #  create a symbolink link in ~/.local/bin (if relative path) relatively to the previous considered location
                elif cmd == "ln":
                    r = ubin.joinpath(self.name if arg1 == arg2 else arg2)
                    r.remove(False)
                    p = (result or tmp).joinpath(arg1)
                    run("chmod +x '%s'" % p, **kw)
                    run("ln -fs '%s' '%s'" % (p, r), **kw)
                    result = r
                # create a shell script to execute the given target from its source directory with its intepreter/launcher
                #  and make it executable
                elif cmd in ["lsh", "lwine", "lwine64"]:
                    arch = ["32", "64"][cmd == "lwine64"]
                    if cmd.startswith("lwine") and hasattr(self, "gui"):
                        arg = self._gui(result.joinpath(arg), arch, True)
                    else:
                        arg1 = opt.joinpath(self.name) if arg == arg1 == arg2 else Path(arg1, expand=True)
                        arg2 = "WINEPREFIX=\"$HOME/.wine%s\" WINEARCH=win%s wine \"%s\" \"$@\"" % (arch, arch, arg2) \
                               if cmd.startswith("lwine") else "./%s" % Path(arg2).basename
                        arg = "#!/bin/bash\nPWD=\"`pwd`\"\nif [[ \"$1\" = /* ]]; then TARGET=\"$1\"; else " \
                              "TARGET=\"$PWD/$1\"; fi\ncd \"%s\"\nset -- \"$TARGET\" \"${@:2}\"\n%s" \
                              "\ncd \"$PWD\"" % (arg1, arg2)
                    result = ubin.joinpath(self.name)
                    self.logger.debug("echo -en '%s' > '%s'" % (arg, result))
                    try:
                        result.write_text(arg)
                        run("chmod +x '%s'" % result, **kw)
                    except PermissionError:
                        self.logger.error("bash: %s: Permission denied" % result)
                # compile a project with Make
                elif cmd == "make":
                    if not result.is_dir():
                        self.logger.error("Got a file ; should have a folder")
                        return
                    os.chdir(str(result))
                    files = [x.filename for x in result.listdir()]
                    make = "make %s" % arg2 if arg2 != arg1 else "make"
                    if "CMakeLists.txt" in files:
                        kw['silent'] += ["Checking out ", "Cloning into ", "Updating "]
                        if run("cmake .", **kw)[-1] == 0:
                            run(make, **kw)
                    elif "Makefile" in files:
                        if "configure.sh" in files:
                            if run("chmod +x configure.sh", **kw)[-1] == 0:
                                run("./configure.sh", **kw)
                        if run(make, **kw)[-1] == 0:
                            ok = False
                            with result.joinpath("Makefile").open() as f:
                                for l in f:
                                    if l.startswith("install:"):
                                        ok = True
                                        break
                            if ok:
                                run("make install", **kw)
                    elif "make.sh" in files:
                        if run("chmod +x make.sh", **kw)[-1] == 0:
                            run("sh -c './make.sh'", **kw)
                    result = result.joinpath(arg1)
                # rename the current working directory and change to the new one
                elif cmd == "md":
                    result, cwd2 = (result or tmp).joinpath(arg), result
                    os.chdir(self.__cwd if self.__cwd != cwd2 else str(tmp))
                    run("mv -f '%s' '%s'" % (cwd2, result), **kw)
                    os.chdir(str(result))
                # simple install through PIP
                elif cmd == "pip":
                    run("pip3 -qq install --user --no-warn-script-location --ignore-installed --break-system-packages "
                        "%s" % arg, **kw)
                # remove a given directory (then bypassing the default removal at the end of all commands)
                elif cmd == "rm":
                    star = arg.endswith("*")
                    p = Path(arg[:-1] if star else arg).absolute()
                    if str(p) == os.getcwd():
                        self.__cwd = self.__cwd if self.__cwd is not None else p.parent
                        self.logger.debug("cd '%s'" % self.__cwd)
                        os.chdir(self.__cwd)
                    __sh = kw.pop('shell', None)
                    kw['shell'] = True
                    run("rm -rf '%s'%s" % (p, ["", "*"][star]), **kw)
                    if __sh is not None:
                        kw['shell'] = __sh
                    rm = False
                # manually set the result to be used in the next command
                elif cmd in ["set", "setp"]:
                    result = arg if cmd == "set" else tmp.joinpath(arg)
                # decompress a RAR/TAR/ZIP archive to the given location (absolute or relative to /tmp/[ITEM]s)
                elif cmd in ["un7z", "unrar", "untar", "unzip"]:
                    ext = "." + (cmd[-2:] if cmd == "un7z" else cmd[-3:])
                    # for TAR, fix the extension (may be .tar.bz2, .tar.gz, .tar.xz, ...)
                    if ext == ".tar" and isinstance(result, Path):
                        # it requires 'result' to be a Path ; this works i.e. after having downloaded the archive with Wget
                        ext = result.extension
                        # when the archive comes from /tmp
                    if result is None:
                        result = tmp.joinpath("%s%s" % (self.name, ext))
                    # when the archive is obtained from /tmp, 'result' was still None and was thus just set ; we still need
                    #  to fix the extension
                    if ext == ".tar":
                        for e in ["br", "bz2", "bz2", "gz", "xz", "Z"]:
                            if result.dirname.joinpath("%s.tar.%s" % (self.name, e)).exists():
                                ext = ".tar." + e
                                result = result.dirname.joinpath("%s%s" % (self.name, ext))
                                break
                    if result.extension == ext:
                        # decompress to the target folder but also to a temp folder if needed (for debugging purpose)
                        paths, first = [tmp.joinpath(arg1)], True
                        if kw.get('verbose', False):
                            paths.append(ts.TempPath(prefix="%s-setup-" % self.type, length=8))
                        # handle password with the second argument
                        pswd = ""
                        if arg2 != arg1:
                            pswd = " -p'%s'" % arg2 if ext == ".7z" else \
                                   " -P '%s'" % arg2 if ext == ".zip" else \
                                   " p'%s'" % arg2 if ext == ".rar" else \
                                   ""
                        for d in paths:
                            run_func = run if first else run2
                            if ext == ".tar.bz2":
                                run_func("bunzip2 -f '%s'" % result, **(kw if first else {}))
                                ext = ".tar"  # switch extension to trigger 'tar x(v)f'
                                result = result.dirname.joinpath(result.stem + ".tar")
                            cmd = "7z x '%s'%s -o'%s' -y" % (result, pswd, d) if ext == ".7z" else \
                                  "unzip%s -o '%s' -d '%s/'" % (pswd, result, d) if ext == ".zip" else \
                                  "unrar x%s -y -u '%s' '%s/'" % (pswd, result, d) if ext == ".rar" else \
                                  "tar xv%sf '%s' -C '%s'" % (["", "z"][ext == ".tar.gz"], result, d)
                            if ext not in [".7z", ".zip"]:
                                d.mkdir(parents=True, exist_ok=True)
                            # log execution (run) the first time, not afterwards (run2)
                            out = run_func(cmd, **(kw if first else {}))
                            first = False
                        # in case of wget, clean up the archive
                        if wget:
                            result.remove()
                        dest = paths[0]
                        # if the archive decompressed a new folder, parse the name and cd to it
                        out = ensure_str(out[0])  # select stdout from the output tuple (stdout, stderr[, return_code])
                        assets = []
                        for l in out.splitlines():
                            l, f = l.strip(), None
                            if l == "":
                                continue
                            if ext == ".zip":
                                if l.startswith("inflating: "):
                                    f = str(Path(l.split("inflating: ", 1)[1]).relative_to(dest)).split("/")[0]
                            elif ext == ".rar":
                                if l.startswith("Extracting "):
                                    f = re.split(r"\s+", l)[1].split("/")[0]
                            elif ext.startswith(".tar"):
                                f = l.split("/")[0]
                            elif ext == ".7z":
                                pass # no parsing for .7z ; a specific folder for the target item shall be declared anyway
                            if f is not None and f not in assets:
                                assets.append(f)
                        if len(assets) == 1 and dest.joinpath(assets[0]).is_dir():
                            dest = dest.joinpath(assets[0])
                        # if the destination is a dir, cd to subfolder as long as there is only one subfolder in the current
                        #  one, this makes 'dest' point to the most relevant folder within the unpacked archive
                        if dest and dest.is_dir():
                            ld = list(dest.listdir())
                            while len(ld) == 1 and ld[0].is_dir():
                                dest = dest.joinpath(ld[0].basename)
                                ld = list(dest.listdir())
                        # automatically move this folder to a standard one based on the item's name
                        #  e.g. '~/.opt/packers/hXOR-Packer v0.1' => ~/.opt/packers/hxor-packer
                        result = opt.joinpath(self.name)
                        if not result.is_samepath(dest):
                            if result.exists():
                                if dest.is_under(result):
                                    t = tmp.joinpath(self.name)
                                    run("mv -f '%s' '%s'" % (dest, t), **kw)
                                    dest = t
                                else:
                                    run("rm -rf '%s'" % result, **kw)
                            run("mv -f '%s' '%s'" % (dest, result), **kw)
                    else:
                        raise ValueError("%s is not a %s file" % (result, ext.lstrip(".").upper()))
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
                            result = tmp.joinpath(self.name + Path(ts.urlparse(url).path).extension)
                            run("wget -qO %s %s" % (result, url), **kw)[-1]
                    # single link
                    else:
                        single_arg = arg1 == arg2
                        url, tag = ts.urlparse(arg1), ""
                        parts = url.path.split(":")
                        if len(parts) == 2:
                            path, tag = parts
                            arg1, idx, regex = None, None, r"([a-zA-Z0-9]+(?:[-_\.][a-zA-Z0-9]+)*)(?:\[(\d)\]|\{(.*?)\})?$"
                            try:
                                tag, idx, pattern = re.match(regex, tag).groups()
                            except AttributeError:
                                pass
                            resp = json.loads(run("curl -Ls https://api.github.com/repos%s/releases/%s" % (path, tag))[0])
                            # case 1: https://github.com/username/repo:TAG{pattern} ; get file based on pattern
                            if pattern is not None:
                                try:
                                    for asset in resp['assets']:
                                        link = asset['browser_download_url']
                                        if re.search(pattern, ts.urlparse(link).path.split("/")[-1]):
                                            arg1 = link
                                            break
                                except KeyError:
                                    self.logger.warning("GitHub API may be blocking requests at the moment ; please try "
                                                        "again later")
                                    raise
                            # case 2: https://github.com/username/repo:TAG[X] ; get Xth file from the selected release
                            else:
                                # if https://github.com/username/repo:TAG, get the 1st file from the selected release
                                if idx is None:
                                    idx = "0"
                                if idx.isdigit():
                                    arg1 = resp['assets'][int(idx)]['browser_download_url']
                            if arg1 is None:
                                raise ValueError("Bad tag for the release URL: %s" % tag)
                        if url.netloc == "github.com" and tag != "":
                            url = ts.urlparse(arg1)
                        result = tmp.joinpath(self.name + Path(url.path).extension if single_arg else arg2)
                        run("wget -qO %s %s" % (result, arg1.replace("%%", "%")), **kw)[-1]
                    wget = True
            if self.__cwd != os.getcwd():
                self.logger.debug("cd %s" % self.__cwd)
                os.chdir(self.__cwd)
            target = tmp.joinpath(self.name)
            if rm and target.exists():
                run("rm -rf %s" % target, **kw)
        
        @update_logger
        def test(self, files=None, keep=False, **kw):
            """ Tests the item on some executable files. """
            if not self._test(kw.pop('silent', False)):
                return
            d = ts.TempPath(prefix="%s-tests-" % self.type, length=8)
            for fmt in self._formats_exp:
                hl = []
                if files:
                    l = [f for f in files if Executable(f).format in self._formats_exp]
                else:
                    l = TEST_FILES.get(fmt, [])
                if len(l) == 0:
                    continue
                self.logger.info(fmt)
                for exe in l:
                    exe = Executable(exe, expand=True)
                    if not self._check(exe):
                        continue
                    tmp = d.joinpath(exe.filename)
                    self.logger.debug(exe.filetype)
                    run("cp %s %s" % (exe, tmp))
                    run("chmod +x %s" % tmp)
                    # use the verb corresponding to the item type by shortening it by 2 chars ; 'packer' will give 'pack'
                    n = tmp.filename
                    label = getattr(self, self.type[:-2])(str(tmp))
                    h = Executable(str(tmp)).hash
                    if h not in hl:
                        hl.append(h)
                    getattr(self.logger, "failure" if label == NOT_PACKED else \
                                         "warning" if label == NOT_LABELLED else "success")(n)
                    self.logger.debug("Label: %s" % label)
                if len(l) > 1 and len(hl) == 1:
                    self.logger.warning("Packing gave the same hash for all the tested files: %s" % hl[0])
            if not keep:
                self.logger.debug("rm -f %s" % str(d))
                d.remove()
        
        @property
        def is_enabled(self):
            """ Simple check for determining if the class is enabled. """
            return self.status in self.__class__._enabled
        
        @property
        def is_variant(self):
            """ Simple check for determining if the class is a variant of another class. """
            return getattr(self.__class__, "parent", None) is not None
        
        @property
        def status(self):
            """ Get the status of item's binary. """
            st = getattr(self, "_status", None)
            return ["not ", ""][b(self.name) in OS_COMMANDS] + "installed" if st is None else st
        
        @classmethod
        def summary(cls, show=False, format="All", **kwargs):
            """ Make a summary table for the given class. """
            items, pheaders = [], ["Name", "Targets", "Status", "Source"]
            n_ok, n, descr = 0, 0, {}
            for item in cls.registry:
                s, ic = item.status, expand_formats(*getattr(item, "formats", ["All"]))
                # check if item is enabled, if it applies to the selected formats and if it is a variant
                if not show and (s in STATUS_DISABLED or item.is_variant) or \
                   all(c not in expand_formats(format) for c in ic):
                    continue
                # now, if keyword-arguments were given, exclude items that do not have the given values set
                _g = lambda attr: getattr(item, attr, "<empty>")
                if len(kwargs) > 0 and all(v not in [None, "All"] and \
                    (v not in _g(k) if isinstance(_g(k), list) else _g(k) != v) for k, v in kwargs.items()):
                    continue
                k = STATUS[s]
                descr.setdefault(k, [])
                if s not in descr[k]:
                    descr[k].append(s)
                n += 1
                if item.status == "ok":
                    n_ok += 1
                items.append([
                    item.cname,
                    ",".join(collapse_formats(*expand_formats(*item.formats))),
                    STATUS[item.status],
                    "<%s>" % item.source,
                ])
            descr = {k: "/".join(sorted(v)) for k, v in descr.items()}
            score = n if n == n_ok else "%d/%d" % (n_ok, n)
            return ([] if n == 0 else \
                    [Section("%ss (%s)" % (cls.__name__, score)), Table(items, column_headers=pheaders)]), descr
    return Base
Base = lazy_object(_init_base)

