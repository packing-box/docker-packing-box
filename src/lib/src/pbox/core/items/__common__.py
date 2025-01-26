# -*- coding: UTF-8 -*-
from os.path import expanduser
from tinyscript import ensure_str, json, logging, os, random, re, shlex, subprocess
from tinyscript.helpers import execute_and_log as run, execute as run2, urlparse, Path, TempPath
from tinyscript.report import *

from ..executable import Executable
from ...helpers import *


__all__ = ["Base"]


def _init_base():
    from ...helpers.items import _init_item
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
            for attr in re.findall(r"<<(.*?)>>", line):
                try:
                    line = line.replace(f"<<{attr}>>", getattr(self, attr))
                except:
                    pass
            return line.replace("$TMP", f"/tmp/{self.type}s") \
                       .replace("$OPT", expanduser(f"~/.opt/{self.type}s")) \
                       .replace("$BIN", expanduser("~/.opt/bin")) \
                       .replace("$LOC", expanduser("~/.local")) \
                       .replace("$HOME", expanduser("~")) \
                       .replace("$CWD", self.__cwd or "")
        
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
                    self.logger.warning(f"'{exe}' does not exist")
                return False
            if fmt not in self._formats_exp:
                if not silent:
                    self.logger.debug(f"does not handle {fmt} executables" if fmt is not None else \
                                      f"does not handle '{exe.filetype}'")
                return False
            for p in exc:
                if isinstance(p, dict):
                    disp, patt = list(p.items())[0]
                    if re.search(patt, exe.filetype):
                        if not silent:
                            self.logger.debug(f"does not handle {disp} files")
                        return False
            if ext in [_ for _ in exc if not isinstance(_, dict)]:
                if not silent:
                    self.logger.debug(f"does not handle .{ext} files")
                return False
            return True
        
        def _gui(self, target, arch, local=False):
            """ Prepare GUI script. """
            preamble = f"PWD='`pwd`'\ncd '{target.dirname}'\nEXE='{target.filename}'" if local else f"EXE='{target}'"
            postamble = ["", "\ncd '$PWD'"][local]
            script = (GUI_SCRIPT % {'arch': arch}).replace("{{preamble}}", preamble).replace("{{postamble}}", postamble)
            cmd = re.compile(r"^(.*?)(?:\s+\((\d*\.\d*|\d+)\)(?:|\s+\[x(\d+)\])?)?$")
            slp = re.compile(r"sleep-per-(B|KB|MB)\s+(\d*\.\d*|\d+)")
            actions = []
            for action in self.gui:
                c, delay, repeat = cmd.match(action).groups()
                for i in range(int(repeat or 1)):
                    if re.match("(behave|click|get(_|mouselocation)|key(down|up)?|mouse(down|move(_relative)?|up)|"
                                "search|set_|type|(get|select)?window)", c):
                        m = re.match("click (\d{1,5}) (\d{1,5})$", c)
                        if m is not None:
                            x, y = m.groups()
                            c = f"mousemove {x} {y} click"
                        c = f"xdotool {c}"
                        if c.endswith(" click") or c == "click":
                            c += " 1"  # set to left-click
                    elif slp.match(c):
                        s, t = slp.match(c).groups()
                        bs = [f" --block-size=1{s[0]}", ""][s == "B"]
                        c = f"sleep $(bc <<< \"`ls -l{bs} $DST | cut -d' ' -f5`*{t}\")"
                    actions.append(c)
                    if delay:
                        actions.append(f"sleep {delay}")
            return script.replace("{{actions}}", "\n".join(actions))
        
        def _test(self, silent=False):
            """ Preamble to the .test(...) method for validation and log purpose. """
            if self.status in STATUS_DISABLED + ["not installed"]:
                self.logger.warning(f"{self.cname} is {self.status}")
                return False
            logging.setLogger(self.name)
            if not silent:
                self.logger.info(f"Testing {self.cname}...")
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
                self.logger.debug(f"disabled (status: {self.status})")
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
        
        def run(self, executable, **kwargs):
            """ Customizable method for shaping the command line to run the item on an input executable. """
            retval = self.name
            use_output = False
            verb = kwargs.get('verbose', False) and getattr(self, "verbose", False)
            bmark, binary, weak = kwargs.get('benchmark', False), kwargs.get('binary', False), kwargs.get('weak', False)
            extra_opt = "" if kwargs.get('extra_opt') is None else kwargs['extra_opt'] + " "
            kw = {'logger': [None, self.logger][verb], 'timeout': getattr(self, "timeout", config['exec_timeout']),
                  'reraise': True, 'silent': []}
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
            steps = getattr(self, "steps", [f"{self.name.replace('_', '-')} {extra_opt}{_str(executable)}"])
            for step in map(self.__expand, steps):
                if self.name in step:
                    i, opt = step.index(self.name), ""
                    if bmark:
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
                m = PARAM_PATTERN.search(step)
                if m:
                    name, values = m.groups()
                    values = self._params.get(name, (values or "").split("|"))
                    for value in values:
                        disp = f"{name}={value}"
                        if len(values) == 2 and "" in values:
                            disp = "" if value == "" else name
                        attempts.append((PARAM_PATTERN.sub(value, step), disp))
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
                    outerr = "\n".join(ERR_PATTERN.sub("", l) for l in output.splitlines() if ERR_PATTERN.match(l))
                    output = "\n".join(l for l in output.splitlines() if not ERR_PATTERN.match(l) and l.strip() != "")
                    # update error string obtained from stderr
                    self._error = "\n".join(l for l in error.splitlines() \
                                            if all(re.search(p, l) is None for p in kw.get('silent', [])))
                    self._error = (self._error + "\n" + outerr).strip()
                    if self.name in attempt and bmark:
                        output, dt = shlex.split(output.splitlines()[-1])
                    if retc > 0:
                        if verb:
                            attempt = attempt.replace(" -v", "")
                        attempts.remove(attempt if param is None else (attempt, param))
                        if len(attempts) == 0:
                            return NOT_LABELLED
                    else:
                        if param:
                            retval += f"[{param}]"
                        break
            os.chdir(self.__cwd)
            r = output if use_output or getattr(self, "use_output", False) else retval
            if bmark:
                r = (r, dt)
            return r
        
        def setup(self, **kw):
            """ Sets the item up according to its install instructions. """
            logging.setLogger(self.name)
            if self.status in STATUS_DISABLED:
                self.logger.info(f"Skipping {self.cname}...")
                self.logger.debug(f"Status: {self.status} ; this means that it won't be installed")
                return
            self.logger.info(f"Setting up {self.cname}...")
            opt, tmp = Path(f"~/.opt/{self.type}s", expand=True), Path(f"/tmp/{self.type}s")
            obin, ubin = Path("~/.opt/bin", create=True, expand=True), Path("~/.local/bin", create=True, expand=True)
            result, rm, wget = None, True, False
            self.__cwd = os.getcwd()
            for cmd in self.install:
                if isinstance(result, Path) and not result.exists():
                    self.logger.critical(f"Last command's result does not exist ({result}) ; current: {cmd}")
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
                    run(f"sudo apt-get -qqy install {arg}", **kw)
                # change to the given dir (starting from reference /tmp/[ITEM]s directory if no command was run before)
                elif cmd == "cd":
                    result = (result or tmp).joinpath(arg)
                    if not result.exists():
                        self.logger.debug(f"mkdir '{result}'")
                        result.mkdir()
                    self.logger.debug(f"cd '{result}'")
                    os.chdir(str(result))
                # add the executable flag on the target
                elif cmd == "chmod":
                    run(f"chmod +x '{result.joinpath(arg)}'", **kw)
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
                    if run(f"cp {['', '-r '][src.is_dir()]}'{src}' '{dst}'", **kw)[-1] == 0 and dst.is_file():
                        run(f"chmod +x '{dst}'", **kw)
                    if arg1 == self.name:
                        rm = False
                    result = dst if dst.is_dir() else dst.dirname
                # install a Debian package using dpkg
                elif cmd == "dpkg":
                    result = None
                    run(f"sudo dpkg -i {arg}", **kw)
                # execute the given shell command or the given list of shell commands
                elif cmd == "exec":
                    result = None
                    if not isinstance(arg, list):
                        arg = [arg]
                    for a in arg:
                        run(a, **kw)
                # simple install through Ruby Gems
                elif cmd == "gem":
                    cwd2 = os.getcwd()
                    os.chdir(str(result))
                    run(f"gem build {arg}.gemspec --silent", **kw)
                    gem = list(result.listdir(filter_func=lambda p: p.extension == ".gem"))[0]
                    run(f"gem install {gem}", silent=["The environment variable HTTP_PROXY is discouraged."])
                    os.chdir(cwd2)
                # git clone a project
                elif cmd in ["git", "gitr"]:
                    result = (result or tmp).joinpath(Path(urlparse(arg).path).stem.lower() if arg1 == arg2 else arg2)
                    result.remove(False)
                    run(f"git clone --quiet {['', '--recursive '][cmd == 'gitr']}{arg1} '{result}'", **kw)
                # go build the target
                elif cmd == "go":
                    result = result if arg2 == arg else Path(arg2, expand=True)
                    cwd2 = os.getcwd()
                    os.chdir(str(result))
                    #result.joinpath("go.mod").remove(False)
                    run(f"go mod init '{arg1}'",
                        silent=["already exists", "creating new go.mod", "add module requirements", "go mod tidy"])
                    run(f"go build -o {self.name} .", silent=["downloading"])
                    os.chdir(cwd2)
                # create a shell script to execute the given target with its intepreter/launcher and make it executable
                elif cmd in ["java", "mono", "sh", "wine", "wine64"]:
                    r, txt, tgt = ubin.joinpath(self.name), "#!/bin/bash\n", (result or opt).joinpath(arg)
                    if cmd == "java":
                        txt += f"java -jar \"{tgt}\" \"$@\""
                    elif cmd == "mono":
                        txt += f"mono \"{tgt}\" \"$@\""
                    elif cmd == "sh":
                        txt += "\n".join(arg.split("\\n"))
                    elif cmd.startswith("wine"):
                        arch = ["32", "64"][cmd == "wine64"]
                        if hasattr(self, "gui"):
                            txt = self._gui(tgt, arch)
                        else:
                            txt += f"WINEPREFIX=\"$HOME/.wine{arch}\" WINEARCH=win{arch} {cmd} \"{tgt}\" \"$@\""
                    self.logger.debug(f"echo -en '{txt}' > '{r}'")
                    try:
                        r.write_text(txt)
                        run(f"chmod +x '{r}'", **kw)
                    except PermissionError:
                        self.logger.error(f"bash: {r}: Permission denied")
                    result = r
                #  create a symbolink link in ~/.local/bin
                elif cmd == "ln":
                    r = ubin.joinpath(self.name if arg1 == arg2 else arg2)
                    r.remove(False)
                    p = (result or tmp).joinpath(arg1)
                    run(f"chmod +x '{p}'" if p.is_under("~") else f"sudo chmod +x '{p}'", **kw)
                    run(f"ln -fs '{p}' '{r}'", **kw)
                    result = r
                # create a shell script to execute the given target from its source directory with its intepreter/
                #  launcher and make it executable
                elif cmd in ["lsh", "lwine", "lwine64"]:
                    arch = ["32", "64"][cmd == "lwine64"]
                    if cmd.startswith("lwine") and hasattr(self, "gui"):
                        arg = self._gui(result.joinpath(arg), arch, True)
                    else:
                        arg1 = opt.joinpath(self.name) if arg == arg1 == arg2 else Path(arg1, expand=True)
                        arg2 = f"WINEPREFIX=\"$HOME/.wine{arch}\" WINEARCH=win{arch} wine \"{arg2}\" \"$@\"" \
                               if cmd.startswith("lwine") else f"./{Path(arg2).basename}"
                        arg = f"#!/bin/bash\nPWD=\"`pwd`\"\nif [[ \"$1\" = /* ]]; then TARGET=\"$1\"; else TARGET=" \
                              f"\"$PWD/$1\"; fi\ncd \"{arg1}\"\nset -- \"$TARGET\" \"{'${@:2}'}\"\n{arg2}\ncd \"$PWD\""
                    result = ubin.joinpath(self.name)
                    self.logger.debug(f"echo -en '{arg}' > '{result}'")
                    try:
                        result.write_text(arg)
                        run(f"chmod +x '{result}'", **kw)
                    except PermissionError:
                        self.logger.error(f"bash: {result}: Permission denied")
                # compile a project with Make
                elif cmd == "make":
                    if not result.is_dir():
                        self.logger.error("Got a file ; should have a folder")
                        return
                    os.chdir(str(result))
                    files = [x.filename for x in result.listdir()]
                    make = f"make {arg2}" if arg2 != arg1 else "make"
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
                    run(f"mv -f '{cwd2}' '{result}'", **kw)
                    os.chdir(str(result))
                # simple install through PIP
                elif cmd in ["pip", "pipr"]:
                    opt = ""
                    if cmd == "pipr":
                        result = (result or tmp).joinpath(arg)
                        opt += "-r "
                    if arg1 != arg2:
                        opt += f"{arg2} "
                    run(f"pip3 -qq install --user --no-warn-script-location --ignore-installed --break-system-packages "
                        f"{opt}{result if cmd == 'pipr' else arg1}", **kw)
                # prepend a line (e.g. a missing shebang) to the targe file
                elif cmd == "prepend":
                    result = (result or tmp).joinpath(arg1)
                    run(f"sed -i '1i{arg2}' {result}", **kw)
                # remove a given directory (then bypassing the default removal at the end of all commands)
                elif cmd == "rm":
                    star = arg.endswith("*")
                    p = Path(arg[:-1] if star else arg).absolute()
                    if str(p) == os.getcwd():
                        self.__cwd = self.__cwd if self.__cwd is not None else p.parent
                        self.logger.debug(f"cd '{self.__cwd}'")
                        os.chdir(self.__cwd)
                    __sh = kw.pop('shell', None)
                    kw['shell'] = True
                    run(f"rm -rf '{p}'{['', '*'][star]}", **kw)
                    if __sh is not None:
                        kw['shell'] = __sh
                    rm = False
                # manually set the result to be used in the next command
                elif cmd in ["set", "setp"]:
                    result = arg if cmd == "set" else tmp.joinpath(arg)
                # decompress a RAR/TAR/ZIP archive to the given location (absolute or relative to /tmp/[ITEM]s)
                elif cmd in ["un7z", "unrar", "untar", "unzip"]:
                    ext = f".{cmd[-[3, 2][cmd == 'un7z']:]}"
                    # for TAR, fix the extension (may be .tar.bz2, .tar.gz, .tar.xz, ...)
                    if ext == ".tar" and isinstance(result, Path):
                        # it requires 'result' to be a Path instance ; this works i.e. after having downloaded with Wget
                        ext = result.extension
                        # when the archive comes from /tmp
                    if result is None:
                        result = tmp.joinpath(f"{self.name}{ext}")
                    # when the archive is obtained from /tmp, 'result' was still None and was thus just set ; we still
                    #  need to fix the extension
                    if ext == ".tar":
                        for e in ["br", "bz2", "bz2", "gz", "xz", "Z"]:
                            if result.dirname.joinpath(f"{self.name}.tar.{e}").exists():
                                ext = ".tar." + e
                                result = result.dirname.joinpath(f"{self.name}{ext}")
                                break
                    if result.extension == ext:
                        # decompress to the target folder but also to a temp folder if needed (for debugging purpose)
                        paths, first = [tmp.joinpath(arg1)], True
                        if kw.get('verbose', False):
                            paths.append(TempPath(prefix=f"{self.type}-setup-", length=8))
                        # handle password with the second argument
                        pswd = ""
                        if arg2 != arg1:
                            pswd = f" -p'{arg2}'" if ext == ".7z" else \
                                   f" -P '{arg2}'" if ext == ".zip" else \
                                   f" p'{arg2}'" if ext == ".rar" else ""
                        for d in paths:
                            run_func = run if first else run2
                            if ext == ".tar.bz2":
                                run_func(f"bunzip2 -f '{result}'", **(kw if first else {}))
                                ext = ".tar"  # switch extension to trigger 'tar x(v)f'
                                result = result.dirname.joinpath(result.stem + ".tar")
                            cmd = f"7z x '{result}'{pswd} -o'{d}' -y" if ext == ".7z" else \
                                  f"unzip{pswd} -o '{result}' -d '{d}/'" if ext == ".zip" else \
                                  f"unrar x{pswd} -y -u '{result}' '{d}/'" if ext == ".rar" else \
                                  f"tar xv{['', 'z'][ext == '.tar.gz']}f '{result}' -C '{d}'"
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
                                pass # no parsing for .7z ; specific folder for the target item shall be declared anyway
                            if f is not None and f not in assets:
                                assets.append(f)
                        if len(assets) == 1 and dest.joinpath(assets[0]).is_dir():
                            dest = dest.joinpath(assets[0])
                        # if the destination is a dir, cd to subfolder as long as there is only one subfolder in the
                        #  current one, this makes 'dest' point to the most relevant folder within the unpacked archive
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
                                    run(f"mv -f '{dest}' '{t}'", **kw)
                                    dest = t
                                else:
                                    run(f"rm -rf '{result}'", **kw)
                            if not result.is_under(dest):
                                run(f"mv -f '{dest}' '{result}'", **kw)
                    else:
                        raise ValueError(f"{result} is not a {ext.lstrip('.').upper()} file")
                # download a resource, possibly downloading 2-stage generated download links (in this case, the list is
                #  handled by downloading the URL from the first element then matching the second element in the URL's
                #  found in the downloaded Web page
                elif cmd == "wget":
                    # (2-stage) dynamic download link
                    rc = 0
                    if isinstance(arg, list):
                        url = arg[0].replace("%%", "%")
                        for line in run(f"wget -qO - {url}", **kw)[0].splitlines():
                            line = line.decode()
                            m = re.search(r"href\s+=\s+(?P<q>[\"'])(.*)(?P=q)", line)
                            if m is not None:
                                url = m.group(1)
                                if Path(urlparse(url).path).stem == (arg[1] if len(arg) > 1 else self.name):
                                    break
                                url = arg[0]
                        if url != arg[0]:
                            result = tmp.joinpath(self.name + Path(urlparse(url).path).extension)
                            run(f"wget -qO {result} {url}", **kw)[-1]
                    # single link
                    else:
                        single_arg = arg1 == arg2
                        url, tag = urlparse(arg1), ""
                        parts = url.path.split(":")
                        if len(parts) == 2:
                            path, tag = parts
                            arg1, idx = None, None
                            regex = r"([a-zA-Z0-9]+(?:[-_\.][a-zA-Z0-9]+)*)(?:\[(\d)\]|\{(.*?)\})?$"
                            try:
                                tag, idx, pattern = re.match(regex, tag).groups()
                            except AttributeError:
                                pass
                            link = f"https://api.github.com/repos{path}/releases/{tag}"
                            resp = json.loads(run(f"curl -Ls {link}")[0])
                            # case 1: https://github.com/username/repo:TAG{pattern} ; get file based on pattern
                            if pattern is not None:
                                try:
                                    for asset in resp['assets']:
                                        link = asset['browser_download_url']
                                        if re.search(pattern, urlparse(link).path.split("/")[-1]):
                                            arg1 = link
                                            break
                                except KeyError:
                                    if resp.get('status') == "404":
                                        self.logger.error(f"{link} not found")
                                    else:
                                        self.logger.warning("GitHub API may be blocking requests at the moment ; please"
                                                            " try again later")
                                        self.logger.debug(resp.content)
                                    raise
                            # case 2: https://github.com/username/repo:TAG[X] ; get Xth file from the selected release
                            else:
                                # if https://github.com/username/repo:TAG, get the 1st file from the selected release
                                if idx is None:
                                    idx = "0"
                                if idx.isdigit():
                                    arg1 = resp['assets'][int(idx)]['browser_download_url']
                            if arg1 is None:
                                raise ValueError(f"Bad tag for the release URL: {tag}")
                        if url.netloc == "github.com" and tag != "":
                            url = urlparse(arg1)
                        result = tmp.joinpath(self.name + Path(url.path).extension if single_arg else arg2)
                        run(f"wget -qO {result} {arg1.replace('%%', '%')}", **kw)[-1]
                    wget = True
            if self.__cwd != os.getcwd():
                self.logger.debug(f"cd {self.__cwd}")
                os.chdir(self.__cwd)
            target = tmp.joinpath(self.name)
            if rm and target.exists():
                run(f"rm -rf {target}", **kw)
        
        def test(self, files=None, keep=False, **kw):
            """ Tests the item on some executable files. """
            # execute a self-test to check that item 'self' is enabled
            if not self._test(kw.pop('silent', False)):
                return
            # then handle input files and test 'self' on them
            d = TempPath(prefix=f"{self.type}-tests-", length=8)
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
                    run(f"cp {exe} {tmp}")
                    run(f"chmod +x {tmp}")
                    # use the verb corresponding to the item type by shortening it by 2 chars ; 'packer' => 'pack'
                    n = tmp.filename
                    label = getattr(self, self.type[:-2])(str(tmp))
                    h = Executable(str(tmp)).hash
                    if h not in hl:
                        hl.append(h)
                    getattr(self.logger, "failure" if label == NOT_PACKED else \
                                         "warning" if label == NOT_LABELLED else "success")(n)
                    self.logger.debug(f"Label: {label}")
                if len(l) > 1 and len(hl) == 1:
                    self.logger.warning(f"Packing gave the same hash for all the tested files: {hl[0]}")
            if not keep:
                self.logger.debug(f"rm -f {d}")
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
            return ["not ", ""][self.name.encode() in OS_COMMANDS] + "installed" if st is None else st
        
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
                    ["", f"<{item.source}>"][item.source != ""],
                ])
            descr = {k: "/".join(sorted(v)) for k, v in descr.items()}
            score = n if n == n_ok else f"{n_ok}/{n}"
            return ([] if n == 0 else \
                    [Section(f"{cls.__name__}s ({score})"), Table(items, column_headers=pheaders)]), descr
    return Base
Base = lazy_object(_init_base)

