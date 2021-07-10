# -*- coding: UTF-8 -*-
from functools import wraps
from tinyscript import b, colored as _c, ensure_str, inspect, json, logging, os, random, re, shlex, subprocess, ts
from tinyscript.helpers import execute_and_log as run, is_executable, is_file, is_folder, yaml_config, Path
from tinyscript.report import *

from .executable import Executable
from ..utils import expand_categories


__all__ = ["class_or_instance_method", "file_or_folder_or_dataset", "make_registry", "Base"]


OS_COMMANDS = subprocess.check_output("compgen -c", shell=True, executable="/bin/bash").splitlines()
PARAM_PATTERN = r"{{(.*?)(?:\[(.*?)\])?}}"
STATUS = [_c("✗", "magenta"), _c("✗", "red"), _c("?", "grey"), _c("✓", "yellow"), _c("✓", "green")]
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


def file_or_folder_or_dataset(method):
    """ This decorator allows to handle, as the first positional argument of an instance method, either an executable,
         a folder with executables or the executable files from a Dataset. """
    @wraps(method)
    def _wrapper(self, *args, **kwargs):
        # collect executables and folders from args
        n, e, l = -1, [], {}
        # exe list extension function
        def _extend_e(i):
            if is_file(i) and is_executable(i) and i not in e:
                e.append(i)
            elif is_folder(i):
                i = Path(i)
                # check if it is a dataset
                if i.joinpath("files").is_dir() and \
                   all(i.joinpath(f).is_file() for f in ["data.csv", "features.json", "labels.json", "names.json"]):
                    with i.joinpath("labels.json").open() as f:
                        l.update(json.load(f))
                    # if so, move to the dataset's "files" folder
                    i = i.joinpath("files")
                for f in i.listdir(is_executable):
                    f = str(f)
                    if f not in e:
                        e.append(f)
            else:
                return False
            return True
        # use the extension function to parse:
        # - positional arguments up to the last valid file/folder
        # - then the 'file' keyword-argument
        for n, a in enumerate(args):
            # if not a valid file, folder or dataset, stop as it is another type of argument
            if not _extend_e(a):
                break
        args = tuple(args[n+1:])
        for a in kwargs.pop('file', []):
            _extend_e(a)
        # then handle the list
        r, kwargs['silent'] = [], False
        for exe in e:
            kwargs['label'] = l.get(Path(exe).stem, -1)
            r.append(method(self, exe, *args, **kwargs))
            kwargs['silent'] = True
        return r[0] if len(r) == 1 else r
    return _wrapper


def make_registry(cls):
    """ Make class' registry of child classes and fill the __all__ list in. """
    cls.registry = []
    itype = cls.__name__.lower()
    glob = inspect.getparentframe().f_back.f_globals
    for item, data in yaml_config(str(Path("/opt/%ss.yml" % itype))).items():
        if item not in glob:
            i = glob[item] = type(item, (cls, ), dict(cls.__dict__))
        else:
            i = glob[item]
        for k, v in data.items():
            if k == "exclude":
                v = {c: sv for sk, sv in v.items() for c in expand_categories(sk)}
            elif k == "status":
                k = "_" + k
            setattr(i, k, v)
        glob['__all__'].append(item)
        cls.registry.append(i())


# based on: https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod
class class_or_instance_method(classmethod):
    def __get__(self, ins, typ):
        return (super().__get__ if ins is None else self.__func__.__get__)(ins, typ)


class Base:
    """ Item abstraction, for defining the common machinery for Detector, Packer and Unpacker.
    
    Attributes:
      status [int]
        0: broken
        1: not installed
        2: gui|todo
        3: installed
        4: tested
      use_output [bool]
    
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
        self.name = self.__class__.__name__.lower()
        self.type = self.__class__.__base__.__name__.lower()
        self.logger = logging.getLogger(self.name)
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
            if self.status <= 2:
                self.logger.debug("Status: %s" % ["broken", "not installed", "todo"][self.status])
                return lambda *a, **kw: False
        return super(Base, self).__getattribute__(name)
    
    def _test(self, silent=False):
        """ Preamble to the .test(...) method. """
        if self.status < 2:
            self.logger.warning("%s %s" % (self.__class__.__name__, ["broken", "not installed"][self.status]))
            return
        logging.setLogger(self.name)
        if not silent:
            self.logger.info("Testing %s..." % self.__class__.__name__)
    
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
                output, _, retc = run(attempt, **kw)
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
        if self.status == 0:
            self.logger.debug("Status: broken")
            return
        logging.setLogger(self.name)
        self.logger.info("Setting up %s..." % self.__class__.__name__)
        tmp, obin, ubin = Path("/tmp"), Path("/opt/bin"), Path("/usr/bin")
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
                if not result.exists():
                    self.logger.debug("mkdir %s" % result)
                    result.mkdir()
                self.logger.debug("cd %s" % result)
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
                if run("cp %s%s %s" % (["", "-r"][src.is_dir()], src, dst), **kw)[-1] == 0 and dst.is_file():
                    run("chmod +x %s" % dst, **kw)
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
                run("git clone -q %s%s %s" % (["", "--recursive "][cmd == "gitr"], arg, result), **kw)
            # make a symbolink link in /usr/bin (if relative path) relatively to the previous considered location
            elif cmd == "ln":
                r = ubin.joinpath(self.name)
                run("ln -s %s %s" % ((result or tmp).joinpath(arg), r), **kw)
                result = r
            elif cmd == "lsh":
                try:
                    arg1, arg2 = shlex.split(arg)
                except ValueError:
                    arg1, arg2 = "/opt/%ss/%s" % (self.type, self.name), arg
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
                try:
                    arg, opt = shlex.split(arg)
                except ValueError:
                    opt = None
                os.chdir(str(result))
                files = [x.filename for x in result.listdir()]
                if "CMakeLists.txt" in files:
                    if run("cmake .", silent=["Checking out ", "Cloning into ", "Updating "], **kw)[-1] == 0:
                        run("make %s" % opt if opt else "make", **kw)
                elif "Makefile" in files:
                    if "configure.sh" in files:
                        if run("chmod +x configure.sh", **kw)[-1] == 0:
                            run("./configure.sh", **kw)
                    if run("make %s" % opt if opt else "make", **kw)[-1] == 0:
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
                if run("mv %s %s" % (result, r), **kw)[-1] == 0 and r.is_file():
                    run("chmod +x %s" % r, **kw)
                result = r
            # simple install through PIP
            elif cmd == "pip":
                run("pip3 -q install %s" % arg, **kw)
            # remove a given directory (then bypassing the default removal at the end of all commands)
            elif cmd == "rm":
                run("rm -rf %s" % Path(arg).absolute(), **kw)
                rm = False
            # manually set the result to be used in the next command
            elif cmd == "set":
                result = arg
            # manually set the result as a path object to be used in the next command
            elif cmd == "setp":
                result = tmp.joinpath(arg)
            # create a shell script to execute Bash code and make it executable
            elif cmd in ["sh", "wine"]:
                r = ubin.joinpath(self.name)
                arg = "\n".join(arg.split("\\n"))
                arg = "#!/bin/bash\n%s" % (arg if cmd == "sh" else "wine %s \"$@\"" % result.joinpath(arg))
                self.logger.debug("echo -en '%s' > %s" % (arg, r))
                try:
                    r.write_text(arg)
                    run("chmod +x %s" % r, **kw)
                except PermissionError:
                    self.logger.error("bash: %s: Permission denied" % r)
                result = r
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
                        run("unzip -qqo %s -d %s" % (result, r), **kw)
                    elif ext == ".rar":
                        if not r.exists():
                            r.mkdir()
                        run("unrar x %s %s" % (result, r), **kw)
                    else:
                        run("mkdir -p %s" % r, **kw)
                        run("tar x%sf %s -C %s" % (["", "z"][ext == ".tar.gz"], result, r), **kw)
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
        self._test(kw.get('silent', False))
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
                ext = exe.extension[1:]
                if exe.category not in self._categories_exp or \
                   ext in getattr(self, "exclude", {}).get(exe.category, []):
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
        if st in ["broken", "useless"]:  # manually set in [item].yml ; for items that cannot be installed/run or that
            return 0          #  are useless (e.g. because it relies on another already existing item)
        elif st in ["gui", "todo"]:  # item to be automated yet
            return 2
        elif b(self.name) not in OS_COMMANDS:  # when the setup failed or is incomplete
            return 1
        elif st == "ok":  # the item runs, works correctly and was tested
            return 4
        # in this case, the binary/symlink exists but running the item was not tested yet
        return 3
    
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
        pfooters = [" ", " ", " "]
        descr = ["broken", "not installed", "todo", "installed", "functional"]
        pfooters.append(" ; ".join("%s: %s" % (s, d) for s, d in \
                        zip(STATUS if show else STATUS[1:], descr if show else descr[1:])))
        n = 0
        for item in cls.registry:
            if not show and item.status == 0:
                continue
            n += 1
            items.append([
                item.__class__.__name__,
                ",".join(item.categories),
                STATUS[item.status],
                "<%s>" % item.source,
            ])
        return [] if n == 0 else [Section("%ss (%d)" % (cls.__name__, n)),
                                  Table(items, column_headers=pheaders, column_footers=pfooters)]

