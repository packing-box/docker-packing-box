# -*- coding: UTF-8 -*-
from os.path import expanduser
from tinyscript import b, colored as _c, ensure_str, json, logging, os, random, re, shlex, subprocess, ts
from tinyscript.helpers import execute_and_log as run, execute as run2, Path
from tinyscript.report import *

from ..common.config import *
from ..common.executable import Executable
from ..common.item import *
from ..common.utils import *


__all__ = ["Base"]


# for a screenshot: "xwd -display $DISPLAY -root -silent | convert xwd:- png:screenshot.png"
GUI_SCRIPT = """#!/bin/bash
source ~/.bash_xvfb
{{preamble}}
SRC="$1"
DST="~/.wine/drive_c/users~/Temp/${1##*/}"
FILE="c:\\\\users\\\\root\\\\Temp\\\\${1##*/}"
cp -f "$SRC" "$DST"
wine "$EXE" &
sleep .5
{{actions}}
ps -eaf | grep -v grep | grep $EXE | awk {'print $2'} > .pid-to-kill && kill `cat .pid-to-kill` && rm -f .pid-to-kill
sleep .1
mv -f "$DST" "$SRC"{{postamble}}
"""
OS_COMMANDS = subprocess.check_output("compgen -c", shell=True, executable="/bin/bash").splitlines()
ERR_PATTERN = r"^\x07?\s*(?:\-\s*)?(?:\[(?:ERR(?:OR)?|\!)\]|ERR(?:OR)?\:)\s*"
PARAM_PATTERN = r"{{([^\{\}]*?)(?:\[([^\{\[\]\}]*?)\])?}}"
STATUS = {
    'broken':        _c("☒", "magenta"),
    'commercial':    "💰",
    'gui':           _c("🗗", "cyan"),
    'info':          _c("ⓘ", "grey"),
    'installed':     _c("☑", "orange"),
    'not installed': _c("☒", "red"),
    'ok':            _c("☑", "green"),
    'todo':          _c("☐", "grey"),
    'useless':       _c("ⓘ", "grey"),
}
STATUS_DISABLED = ["broken", "commercial", "info", "useless"]
STATUS_ENABLED = [s for s in STATUS.keys() if s not in STATUS_DISABLED + ["not installed"]]
TEST_FILES = {
    'ELF32': [
        "/usr/bin/perl5.30-i386-linux-gnu",
        "/usr/lib/wine/wine",
        "/usr/lib/wine/wineserver32",
        "/usr/libx32/crti.o",
        "/usr/libx32/libpcprofile.so",
    ],
    'ELF64': [
        "/usr/bin/cat",
        "/usr/bin/ls",
        "/usr/lib/man-db/manconv",
        "/usr/lib/openssh/ssh-keysign",
        "/usr/lib/git-core/git",
        "/usr/lib/x86_64-linux-gnu/crti.o",
        "/usr/lib/x86_64-linux-gnu/libpcprofile.so",
        "/usr/lib/sudo/sudoers.so",
    ],
    'MSDOS': [
        "~/.wine/drive_c/windows/rundll.exe",
        "~/.wine/drive_c/windows/syswow64/gdi.exe",
        "~/.wine/drive_c/windows/syswow64/user.exe",
        "~/.wine/drive_c/windows/syswow64/mouse.drv",
        "~/.wine/drive_c/windows/syswow64/winaspi.dll",
    ],
    'PE32': [
        "~/.wine/drive_c/windows/winhlp32.exe",
        "~/.wine/drive_c/windows/syswow64/plugplay.exe",
        "~/.wine/drive_c/windows/syswow64/winemine.exe",
        "~/.wine/drive_c/windows/twain_32.dll",
        "~/.wine/drive_c/windows/twain_32/sane.ds",
        "~/.wine/drive_c/windows/syswow64/msscript.ocx",
        "~/.wine/drive_c/windows/syswow64/msgsm32.acm",
    ],
    'PE64': [
        "~/.wine/drive_c/windows/hh.exe",
        "~/.wine/drive_c/windows/system32/spoolsv.exe",
        "~/.wine/drive_c/windows/system32/dmscript.dll",
        "~/.wine/drive_c/windows/twain_64/gphoto2.ds",
        "~/.wine/drive_c/windows/system32/msscript.ocx",
        "~/.wine/drive_c/windows/system32/msadp32.acm",
    ],
}


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
        self.__init = False
    
    def __expand(self, line):
        return line.replace("$TMP", "/tmp/%ss" % self.type).replace("$OPT", expanduser("~/.opt/%ss" % self.type))
    
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
        if fmt not in self._formats_exp:
            if not silent:
                self.logger.debug("does not handle %s executables" % fmt)
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
    
    def _gui(self, target, local=False):
        """ Prepare GUI script. """
        preamble = "PWD='`pwd`'\ncd '%s'\nEXE='%s'" % (target.dirname, target.filename) if local else \
                   "EXE='%s'" % target
        postamble = ["", "\ncd '$PWD'"][local]
        script = GUI_SCRIPT.replace("{{preamble}}", preamble).replace("{{postamble}}", postamble)
        cmd = re.compile(r"^(.*?)(?:\s+\((\d*\.\d*|\d+)\)(?:|\s+\[x(\d+)\])?)?$")
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
            self.logger.warning("%s disabled" % self.name)
            self.logger.debug("Status: " + self.status)
        return False
    
    def help(self, extras=None):
        """ Returns a help message in Markdown format. """
        md = Report()
        if getattr(self, "description", None):
            md.append(Text(self.description))
        if getattr(self, "comment", None):
            md.append(Blockquote("**Note**: " + self.comment))
        md.append(Text("**Source**    : " + self.source))
        md.append(Text("**Applies to**: " + ", ".join(sorted(expand_formats(*self.formats, **{'once': True})))))
        if getattr(self, "packers", None):
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
        kw = {'logger': self.logger, 'silent': []}
        if not config['wine_errors']:
            kw['silent'].append(r"^[0-9a-f]{4}\:(?:err|fixme)\:")
        _str = lambda o: "'" + str(o) + "'"
        _repl = lambda s: s.replace("{{executable}}", _str(executable)) \
                           .replace("{{executable.dirname}}", _str(executable.dirname)) \
                           .replace("{{executable.filename}}", _str(executable.filename)) \
                           .replace("{{executable.extension}}", _str(executable.extension)) \
                           .replace("{{executable.stem}}", _str(executable.dirname.joinpath(executable.stem)))
        kw['silent'].extend(list(map(_repl, getattr(self, "silent", []))))
        output, cwd = "", os.getcwd()
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
            step = _repl(step)
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
                    self.logger.error("Bad command: %s" % attempt)
                    output, error, retc = "", str(e), 1
                output = ensure_str(output).strip()
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
                        return
                else:
                    if param:
                        retval += "[%s]" % param
                    break
        os.chdir(cwd)
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
        result, rm, wget, kw = None, True, False, {'logger': self.logger, 'silent': getattr(self, "silent", [])}
        cwd = os.getcwd()
        for cmd in self.install:
            if isinstance(result, Path) and not result.exists():
                self.logger.critical("Last command's result does not exist (%s) ; current: %s" % (result, cmd))
                return
            name = cmd.pop('name', "")
            if name != "":
                self.logger.info("> " + name)
            # extract argument(s)
            cmd, arg = list(cmd.items())[0]
            try:
                arg = self.__expand(arg)
                arg1, arg2 = shlex.split(arg)
            except AttributeError:
                arg = [self.__expand(a) for a in arg]  # e.g. exec can handle 'arg' as a list of commands
            except ValueError:
                arg1, arg2 = arg, arg
            # simple install through APT
            if cmd == "apt":
                kw['silent'] += ["apt does not have a stable CLI interface"]
                run("apt -qy install %s" % arg, **kw)
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
                src, dst = (result or tmp).joinpath(arg1), ubin.joinpath(arg2)
                # if ~/.local/bin/... exists, then save it to ~/.opt/bin/... to superseed it
                if dst.is_samepath(ubin.joinpath(arg2)) and dst.exists():
                    dst = obin.joinpath(arg2)
                if run("cp %s'%s' '%s'" % (["", "-r "][src.is_dir()], src, dst), **kw)[-1] == 0 and dst.is_file():
                    run("chmod +x '%s'" % dst, **kw)
                if arg1 == self.name:
                    rm = False
            # execute the given shell command or the given list of shell commands
            elif cmd == "exec":
                result = None
                if not isinstance(arg, list):
                    arg = [arg]
                for a in arg:
                    run(a, **kw)
            # git clone a project
            elif cmd in ["git", "gitr"]:
                result = (result or tmp).joinpath(Path(ts.urlparse(arg).path).stem.lower())
                result.remove(False)
                run("git clone --quiet %s%s '%s'" % (["", "--recursive "][cmd == "gitr"], arg, result), **kw)
            # go build the target
            elif cmd == "go":
                result = result if arg2 == arg else Path(arg2, expand=True)
                cwd2 = os.getcwd()
                os.chdir(str(result))
                result.joinpath("go.mod").remove(False)
                run("go mod init '%s'" % arg1, silent=["creating new go.mod", "add module requirements", "go mod tidy"])
                run("go build -o %s ." % self.name)
                os.chdir(cwd2)
            # create a shell script to execute the given target with its intepreter/launcher and make it executable
            elif cmd in ["java", "mono", "sh", "wine"]:
                r, txt, tgt = ubin.joinpath(self.name), "#!/bin/bash\n", (result or opt).joinpath(arg)
                if cmd == "java":
                    txt += "java -jar \"%s\" \"$@\"" % tgt
                elif cmd == "mono":
                    txt += "mono \"%s\" \"$@\"" % tgt
                elif cmd == "sh":
                    txt += "\n".join(arg.split("\\n"))
                elif cmd == "wine":
                    if hasattr(self, "gui"):
                        txt = self._gui(tgt)
                    else:
                        txt += "wine \"%s\" \"$@\"" % tgt
                self.logger.debug("echo -en '%s' > '%s'" % (txt, r))
                try:
                    r.write_text(txt)
                    run("chmod +x '%s'" % r, **kw)
                except PermissionError:
                    self.logger.error("bash: %s: Permission denied" % r)
                result = r
            #  create a symbolink link in ~/.local/bin (if relative path) relatively to the previous considered location
            elif cmd == "ln":
                r = ubin.joinpath(self.name)
                r.remove(False)
                run("ln -fs '%s' '%s'" % ((result or tmp).joinpath(arg), r), **kw)
                result = r
            # create a shell script to execute the given target from its source directory with its intepreter/launcher
            #  and make it executable
            elif cmd in ["lsh", "lwine"]:
                if cmd == "lwine" and hasattr(self, "gui"):
                    arg = self._gui(result.joinpath(arg), True)
                else:
                    arg1 = opt.joinpath(self.name) if arg == arg1 == arg2 else Path(arg1, expand=True)
                    arg2 = "wine \"%s\" \"$@\"" % arg2 if cmd == "lwine" else "./%s" % arg2
                    arg = "#!/bin/bash\nPWD=\"`pwd`\"\nif [[ \"$1\" = /* ]]; then TARGET=\"$1\"; else " \
                          "TARGET=\"$PWD/$1\"; fi\ncd \"%s\"\n%s \"$TARGET\" \"$2\"\ncd \"$PWD\"" % (arg1, arg2)
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
                os.chdir(cwd if cwd != cwd2 else str(tmp))
                run("mv -f '%s' '%s'" % (cwd2, result), **kw)
                os.chdir(str(result))
            # simple install through PIP
            elif cmd == "pip":
                run("pip3 -qq install --user --no-warn-script-location --ignore-installed %s" % arg, **kw)
            # remove a given directory (then bypassing the default removal at the end of all commands)
            elif cmd == "rm":
                run("rm -rf '%s'" % Path(arg).absolute(), **kw)
                rm = False
            # manually set the result to be used in the next command
            elif cmd in ["set", "setp"]:
                result = arg if cmd == "set" else tmp.joinpath(arg)
            # decompress a RAR/TAR/ZIP archive to the given location (absolute or relative to /tmp/[ITEM]s)
            elif cmd in ["un7z", "unrar", "untar", "unzip"]:
                ext = "." + (cmd[-2:] if cmd == "un7z" else cmd[-3:])
                # for tar, do not use the extension as a check (may be .tar.bz2, .tar.gz, .tar.xz, ...)
                if ext == ".tar" and isinstance(result, Path):
                    ext = result.extension
                if result is None:
                    result = tmp.joinpath("%s%s" % (self.name, ext))
                # rectify ext and result if .tar.xz
                result2 = result.dirname.joinpath("%s.tar.xz" % self.name)
                if cmd[-3:] == "tar" and (not result.exists() or result.extension == ".tar.xz") and result2.exists():
                    ext, result = ".tar.xz", result2
                if result.extension == ext:
                    # decompress to the target folder but also to a temp folder if needed (for debugging purpose)
                    paths, first = [tmp.joinpath(arg)], True
                    if kw.get('verbose', False):
                        paths.append(ts.TempPath(prefix="%s-setup-" % self.type, length=8))
                    for d in paths:
                        run_func = run if first else run2
                        if ext == ".tar.bz2":
                            run_func("bunzip2 -f '%s'" % result, **(kw if first else {}))
                            ext = ".tar"  # switch extension to trigger 'tar x(v)f'
                            result = result.dirname.joinpath(result.stem + ".tar")
                        cmd = "7z x '%s' -o'%s' -y" % (result, d) if ext == ".7z" else \
                              "unzip -o '%s' -d '%s/'" % (result, d) if ext == ".zip" else \
                              "unrar x -y -u '%s' '%s/'" % (result, d) if ext == ".rar" else \
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
                        result = tmp.joinpath(p + Path(ts.urlparse(url).path).extension)
                        run("wget -q -O %s %s" % (result, url), **kw)[-1]
                # single link
                else:
                    url, tag = ts.urlparse(arg), ""
                    parts = url.path.split(":")
                    if len(parts) == 2:
                        path, tag = parts
                        arg, idx, regex = None, None, r"([a-zA-Z0-9]+(?:[-_\.][a-zA-Z0-9]+)*)(?:\[(\d)\]|\{(.*?)\})?$"
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
                                        arg = link
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
                                arg = resp['assets'][int(idx)]['browser_download_url']
                        if arg is None:
                            raise ValueError("Bad tag for the release URL: %s" % tag)
                    if url.netloc == "github.com" and tag != "":
                        url = ts.urlparse(arg)
                    result = tmp.joinpath(self.name + Path(url.path).extension)
                    run("wget -q -O %s %s" % (result, arg.replace("%%", "%")), **kw)[-1]
                wget = True
        if cwd != os.getcwd():
            self.logger.debug("cd %s" % cwd)
            os.chdir(cwd)
        if rm:
            run("rm -rf %s" % tmp.joinpath(self.name), **kw)
    
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
                exe = Executable(exe)
                if not self._check(exe):
                    continue
                tmp = d.joinpath(exe.filename)
                self.logger.debug(exe.filetype)
                run("cp %s %s" % (exe, tmp))
                run("chmod +x %s" % tmp)
                # use the verb corresponding to the item type by shortening it by 2 chars ; 'packer' will give 'pack'
                n = tmp.filename
                h, label = getattr(self, self.type[:-2])(str(tmp), include_hash=True)
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

