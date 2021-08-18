# -*- coding: UTF-8 -*-
from tinyscript import b, colored as _c, ensure_str, json, logging, os, random, re, shlex, subprocess, ts
from tinyscript.helpers import execute_and_log as run, is_executable, is_file, is_folder, Path
from tinyscript.report import *

from ..common.config import config
from ..common.executable import Executable
from ..common.item import Item
from ..common.utils import *


__all__ = ["Base"]


# for a screenshot: "xwd -display $DISPLAY -root -silent | convert xwd:- png:screenshot.png"
GUI_SCRIPT = """#!/bin/bash
source /root/.bash_xvfb
{{preamble}}
SRC="$1"
DST="/root/.wine/drive_c/users/root/Temp/${1##*/}"
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
PARAM_PATTERN = r"{{(.*?)(?:\[(.*?)\])?}}"
STATUS = {
    'broken':        _c("☒", "magenta"),
    'commercial':    _c("💰", "yellow"),
    'gui':           _c("🗗", "grey"),
    'info':          _c("ⓘ", "grey"),
    'installed':     _c("☑", "orange"),
    'not installed': _c("☒", "red"),
    'ok':            _c("☑", "green"),
    'todo':          _c("☐", "grey"),
    'useless':       _c("ⓘ", "grey"),
}
STATUS_DISABLED = ["broken", "commercial", "info", "useless"]
TEST_FILES = {
    'ELF32': [
        "/usr/bin/perl5.30-i386-linux-gnu",
        "/usr/lib/wine/wine",
        "/usr/lib/wine/wineserver32",
        "/usr/libx32/crti.o",
        "/usr/libx32/libpcprofile.so",
    ],
    'ELF64': [
        "/usr/bin/ls",
        "/usr/lib/man-db/manconv",
        "/usr/lib/openssh/ssh-keysign",
        "/usr/lib/git-core/git",
        "/usr/lib/x86_64-linux-gnu/crti.o",
        "/usr/lib/x86_64-linux-gnu/libpcprofile.so",
        "/usr/lib/sudo/sudoers.so",
    ],
    'MSDOS': [
        "/root/.wine/drive_c/windows/rundll.exe",
        "/root/.wine/drive_c/windows/syswow64/gdi.exe",
        "/root/.wine/drive_c/windows/syswow64/user.exe",
        "/root/.wine/drive_c/windows/syswow64/mouse.drv",
        "/root/.wine/drive_c/windows/syswow64/winaspi.dll",
    ],
    'PE32': [
        "/root/.wine/drive_c/windows/winhlp32.exe",
        "/root/.wine/drive_c/windows/syswow64/plugplay.exe",
        "/root/.wine/drive_c/windows/syswow64/winemine.exe",
        "/root/.wine/drive_c/windows/twain_32.dll",
        "/root/.wine/drive_c/windows/twain_32/sane.ds",
    ],
    'PE64': [
        "/root/.wine/drive_c/windows/hh.exe",
        "/root/.wine/drive_c/windows/system32/spoolsv.exe",
        "/root/.wine/drive_c/windows/system32/dmscript.dll",
        "/root/.wine/drive_c/windows/twain_64/gphoto2.ds",
    ],
}


class Base(Item):
    """ Base item abstraction, for defining the common machinery for Detector, Packer and Unpacker.
    
    Instance methods:
      .check(*categories) [bool]
      .help() [str]
      .run(executable, **kwargs) [str|(str,time)]
      .setup(**kwargs)
      .test(files, **kwargs)
    
    Class methods:
      .get(item)
      .summary()
    """
    def __init__(self):
        super(Base, self).__init__()
        self._categories_exp = expand_categories(*self.categories)
        self._bad = False
        self._error = False
        self._params = {}
        self.__init = False
    
    def __getattribute__(self, name):
        if name == getattr(super(Base, self), "type", "")[:-2]:
            # this part sets the internal logger up while triggering the main method (e.g. pack) for the first time ;
            #  this is necessary as the registry of subclasses is made at loading, before Tinyscript initializes its
            #  main logger whose config must be propagated to child loggers
            if not self.__init:
                logging.setLogger(self.name)
                self.__init = True
            # check: is this item operational ?
            if self.status in STATUS_DISABLED + ["not installed", "todo"]:
                self.logger.debug("Status: %s" % self.status)
                return lambda *a, **kw: False
        return super(Base, self).__getattribute__(name)
    
    def _check(self, exe):
        """ Check if the given executable can be processed by this item. """
        c = exe.category
        exe = Executable(exe)
        ext = exe.extension[1:]
        exc = getattr(self, "exclude", [])
        if isinstance(exc, dict):
            l = []
            for k, v in exc.items():
                if c in expand_categories(k):
                    l.extend(v)
            exc = list(set(l))
        if c not in self._categories_exp:
            self.logger.debug("%s does not handle %s executables" % (self.cname, c))
            return False
        if ext in exc:
            self.logger.debug("%s does not handle .%s files" % (self.cname, ext))
            return False
        return True
    
    def _gui(self, target, local=False):
        """ Prepare GUI script. """
        preamble = "PWD=\"`pwd`\"\ncd \"%s\"\nEXE=\"%s\"" % (target.dirname, target.filename) if local else \
                   "EXE=\"%s\"" % target
        postamble = ["", "\ncd \"$PWD\""][local]
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
    
    def _test(self, silent=False):
        """ Preamble to the .test(...) method. """
        if self.status in STATUS_DISABLED + ["not installed"]:
            self.logger.warning("%s is %s" % (self.cname, self.status))
            return False
        logging.setLogger(self.name)
        if not silent:
            self.logger.info("Testing %s..." % self.cname)
        return True
    
    def check(self, *categories):
        """ Checks if the current item is applicable to the given categories. """
        return any(c in self._categories_exp for c in expand_categories(*(categories or ["All"])))
    
    def help(self):
        """ Returns a help message in Markdown format. """
        md = Report()
        if getattr(self, "description", None):
            md.append(Text(self.description))
        if getattr(self, "comment", None):
            md.append(Blockquote("**Note**: " + self.comment))
        md.append(Text("**Source**: " + self.source))
        md.append(Text("**Applies to**: " + ", ".join(sorted(expand_categories(*self.categories, **{'once': True})))))
        if getattr(self, "references", None):
            md.append(Section("References"), List(*self.references, **{'ordered': True}))
        return md.md()
    
    def run(self, executable, **kwargs):
        """ Customizable method for shaping the command line to run the item on an input executable. """
        retval = self.name
        use_output, benchmark, verbose = False, kwargs.get('benchmark', False), kwargs.get('verbose', False)
        kw = {'logger': self.logger}
        output = None
        cwd = os.getcwd()
        for step in getattr(self, "steps", ["%s %s" % (self.name, executable)]):
            if self.name in step and benchmark:
                i = step.index(self.name)
                step = step[:i] + self.name + " -b" + step[i+len(self.name):]
            attempts = []
            # first, replace generic patterns
            step = step.replace("{{executable}}", str(executable)) \
                       .replace("{{executable.dirname}}", str(executable.dirname)) \
                       .replace("{{executable.filename}}", executable.filename) \
                       .replace("{{executable.extension}}", executable.extension) \
                       .replace("{{executable.stem}}", str(executable.dirname.joinpath(executable.stem)))
            # now, replace a previous output and handle it as the return value
            if "{{output}}" in step:
                step = step.replace("{{output}}", ensure_str(output or ""))
                use_output = True
            # then, search for parameter patterns
            m = re.search(PARAM_PATTERN, step)
            if m:
                name, values = m.groups()
                values = (values or "").split("|")
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
                    os.chdir(attempt[3:])
                    continue
                if verbose:
                    attempt += " -v"
                kw['shell'] = ">" in attempt
                kw['silent'] = getattr(self, "silent", [])
                if not config['wine_errors']:
                    kw['silent'].append(r"^[0-9a-f]{4}:(err|fixme):")
                try:
                    output, _, retc = run(attempt, **kw)
                except:
                    self.logger.error("Command: %s" % attempt)
                    raise
                output = ensure_str(output)
                if self.name in attempt and benchmark:
                    output, dt = shlex.split(output)
                if retc > 0:
                    attempts.remove(attempt if param is None else (attempt, param))
                    if len(attempts) == 0:
                        self._error = True
                        return
                else:
                    if param:
                        retval += "[%s]" % param
                    break
        os.chdir(cwd)
        r = (output or None) if use_output or getattr(self, "use_output", False) else retval
        if benchmark:
            r = (r, dt)
        return r
    
    def setup(self, **kw):
        """ Sets the item up according to its install instructions. """
        logging.setLogger(self.name)
        if self.status in STATUS_DISABLED:
            self.logger.info("Skipping %s..." % self.cname)
            self.logger.debug("Status: %s ; this means that it won't be installed" % self.status)
            return
        self.logger.info("Setting up %s..." % self.cname)
        tmp, obin, ubin = Path("/tmp"), Path("/opt/bin"), Path("/usr/bin")
        result, rm, kw = None, True, {'logger': self.logger, 'silent': getattr(self, "silent", [])}
        cwd = os.getcwd()
        for cmd, arg in self.install.items():
            if isinstance(result, Path) and not result.exists():
                self.logger.critical("Last command's result does not exist (%s) ; current: %s" % (result, cmd))
                return
            # simple install through APT
            if cmd == "apt":
                kw['silent'] += ["apt does not have a stable CLI interface"]
                run("apt -qy install %s" % arg, **kw)
            # change to the given dir (starting from the reference /tmp directory if no command was run before)
            elif cmd == "cd":
                result = (result or tmp).joinpath(arg)
                if not result.exists():
                    self.logger.debug("mkdir \"%s\"" % result)
                    result.mkdir()
                self.logger.debug("cd \"%s\"" % result)
                os.chdir(str(result))
            # copy a file from the previous location (or /tmp if not defined) to /opt/bin, making the destination file
            #  executable
            elif cmd == "copy":
                try:
                    arg1, arg2 = shlex.split(arg)
                except ValueError:
                    arg1, arg2 = arg, arg
                src, dst = (result or tmp).joinpath(arg1), ubin.joinpath(arg2)
                # if /usr/bin/... exists, then save it to /opt/bin/... to superseed it
                if dst.is_samepath(ubin.joinpath(arg2)) and dst.exists():
                    dst = obin.joinpath(arg2)
                if run("cp %s\"%s\" \"%s\"" % (["", "-r"][src.is_dir()], src, dst), **kw)[-1] == 0 and dst.is_file():
                    run("chmod +x \"%s\"" % dst, **kw)
            # execute the given command as is, with no pre-/post-condition, not altering the result state variable
            #  (if a list of commands is given, execute them all
            elif cmd == "exec":
                result = None
                if not isinstance(arg, list):
                    arg = [arg]
                for a in arg:
                    run(a, **kw)
            # git clone a project
            elif cmd in ["git", "gitr"]:
                result = (result or tmp).joinpath(Path(ts.urlparse(arg).path).stem)
                result.remove(False)
                run("git clone -q %s%s \"%s\"" % (["", "--recursive "][cmd == "gitr"], arg, result), **kw)
            # create a shell script to execute Bash code and make it executable
            elif cmd in ["java", "sh", "wine"]:
                r, txt, tgt = ubin.joinpath(self.name), "#!/bin/bash\n", result.joinpath(arg)
                if cmd == "java":
                    txt += "java -jar \"%s\" \"$@\"" % tgt
                elif cmd == "sh":
                    txt += "\n".join(arg.split("\\n"))
                elif cmd == "wine":
                    if hasattr(self, "gui"):
                        txt = self._gui(tgt)
                    else:
                        txt += "wine \"%s\" \"$@\"" % tgt
                self.logger.debug("echo -en '%s' > \"%s\"" % (txt, r))
                try:
                    r.write_text(txt)
                    run("chmod +x \"%s\"" % r, **kw)
                except PermissionError:
                    self.logger.error("bash: %s: Permission denied" % r)
                result = r
            # make a symbolink link in /usr/bin (if relative path) relatively to the previous considered location
            elif cmd == "ln":
                r = ubin.joinpath(self.name)
                r.remove(False)
                run("ln -s \"%s\" \"%s\"" % ((result or tmp).joinpath(arg), r), **kw)
                result = r
            elif cmd in ["lsh", "lwine"]:
                if cmd == "lwine" and hasattr(self, "gui"):
                    arg = self._gui(result.joinpath(arg), True)
                else:
                    try:
                        arg1, arg2 = shlex.split(arg)
                    except ValueError:
                        arg1, arg2 = "/opt/%ss/%s" % (self.type, self.name), arg
                    arg2 = "wine \"%s\" \"$@\"" % arg2 if cmd == "lwine" else "./%s" % arg2
                    arg = "#!/bin/bash\nPWD=`pwd`\nif [[ \"$1\" = /* ]]; then TARGET=\"$1\"; else TARGET=\"$PWD/$1\";" \
                          " fi\ncd \"%s\"\n%s \"$TARGET\" \"$2\"\ncd \"$PWD\"" % (arg1, arg2)
                result = ubin.joinpath(self.name)
                self.logger.debug("echo -en '%s' > \"%s\"" % (arg, result))
                try:
                    result.write_text(arg)
                    run("chmod +x \"%s\"" % result, **kw)
                except PermissionError:
                    self.logger.error("bash: %s: Permission denied" % result)
            # compile a C project
            elif cmd == "make":
                if not result.is_dir():
                    self.logger.error("Got a file ; should have a folder")
                    return
                try:
                    arg, opt = shlex.split(arg)
                except ValueError:
                    opt = None
                os.chdir(str(result))
                files = [x.filename for x in result.listdir()]
                if "CMakeLists.txt" in files:
                    kw['silent'] += ["Checking out ", "Cloning into ", "Updating "]
                    if run("cmake .", **kw)[-1] == 0:
                        run("make %s" % opt if opt else "make", **kw)
                elif "Makefile" in files:
                    if "configure.sh" in files:
                        if run("chmod +x configure.sh", **kw)[-1] == 0:
                            run("./configure.sh", **kw)
                    if run("make %s" % opt if opt else "make", **kw)[-1] == 0:
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
                result = result.joinpath(arg)
            # move the previous location to /usr/bin (if relative path), make it executable if it is a file
            elif cmd == "move":
                result = (result or tmp).joinpath(arg)
                r = ubin.joinpath(self.name)
                # if /usr/bin/... exists, then save it to /opt/bin/... to superseed it
                if r.is_samepath(ubin.joinpath(self.name)) and r.exists():
                    r = obin.joinpath(self.name)
                if run("mv -f \"%s\" \"%s\"" % (result, r), **kw)[-1] == 0 and r.is_file():
                    run("chmod +x \"%s\"" % r, **kw)
                result = r
            # simple install through PIP
            elif cmd == "pip":
                run("pip3 -q install %s" % arg, **kw)
            # remove a given directory (then bypassing the default removal at the end of all commands)
            elif cmd == "rm":
                run("rm -rf \"%s\"" % Path(arg).absolute(), **kw)
                rm = False
            # manually set the result to be used in the next command
            elif cmd == "set":
                result = arg
            # manually set the result as a path object to be used in the next command
            elif cmd == "setp":
                result = tmp.joinpath(arg)
            # decompress a RAR/ZIP archive to the given location (absolute or relative to /tmp)
            elif cmd in ["unrar", "untar", "unzip"]:
                ext = "." + cmd[-3:]
                if ext == ".tar":
                    ext = ".tar.gz"
                if result is None:
                    result = tmp.joinpath("%s%s" % (self.name, ext))
                result2 = result.dirname.joinpath("%s.tar.xz" % self.name)
                if cmd[-3:] == "tar" and not result.exists() and result2.exists():
                    ext, result = ".tar.xz", result2
                if result.extension == ext:
                    r = tmp.joinpath(arg)
                    if ext == ".zip":
                        run("unzip -qqo \"%s\" -d \"%s\"" % (result, r), **kw)
                    elif ext == ".rar":
                        r.mkdir(parents=True, exist_ok=True)
                        run("unrar x \"%s\" \"%s\"" % (result, r), **kw)
                    else:
                        r.mkdir(parents=True, exist_ok=True)
                        run("tar x%sf \"%s\" -C \"%s\"" % (["", "z"][ext == ".tar.gz"], result, r), **kw)
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
        self.logger.debug("cd %s" % cwd)
        os.chdir(cwd)
        if rm:
            run("rm -rf %s" % tmp.joinpath(self.name), **kw)
    
    def test(self, files=None, **kw):
        """ Tests the item on some executable files. """
        if not self._test(kw.pop('silent', False)):
            return
        d = ts.TempPath(prefix="%s-tests-" % self.type, length=8)
        for category in self._categories_exp:
            if files:
                l = [f for f in files if Executable(f).category in self._categories_exp]
            else:
                l = TEST_FILES.get(category, [])
            if len(l) == 0:
                continue
            self.logger.info(category)
            for exe in l:
                exe = Executable(exe)
                if not self._check(exe):
                    continue
                tmp = d.joinpath(exe.filename)
                self.logger.debug(exe.filetype)
                run("cp %s %s" % (exe, tmp))
                run("chmod +x %s" % tmp)
                # use the verb corresponding to the item type by shortening it by 2 chars ; 'packer' will give 'pack'
                label, n = getattr(self, self.type[:-2])(str(tmp)), tmp.filename
                getattr(self.logger, "failure" if label is None else "warning" if label is False else "success")(n)
                self.logger.debug("Label: %s" % label)
        d.remove()
    
    @property
    def status(self):
        """ Get the status of item's binary. """
        st = getattr(self, "_status", None)
        if st is None:
            return ["not ", ""][b(self.name) in OS_COMMANDS] + "installed"
        return st
    
    @classmethod
    def get(cls, item):
        """ Simple class method for returning the class of an item based on its name (case-insensitive). """
        for i in cls.registry:
            if i.name == (item.name if isinstance(item, Base) else item).lower():
                return i
    
    @classmethod
    def summary(cls, show=False):
        items = []
        pheaders = ["Name", "Targets", "Status", "Source"]
        n, descr = 0, {}
        for item in cls.registry:
            s = item.status
            if not show and s in STATUS_DISABLED:
                continue
            k = STATUS[s]
            descr.setdefault(k, [])
            if s not in descr[k]:
                descr[k].append(s)
            n += 1
            items.append([
                item.cname,
                ",".join(collapse_categories(*expand_categories(*item.categories))),
                STATUS[item.status],
                "<%s>" % item.source,
            ])
        descr = {k: "/".join(sorted(v)) for k, v in descr.items()}
        return ([] if n == 0 else [Section("%ss (%d)" % (cls.__name__, n)), Table(items, column_headers=pheaders)]), \
               descr

