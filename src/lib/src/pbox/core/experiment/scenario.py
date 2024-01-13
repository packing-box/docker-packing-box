# -*- coding: UTF-8 -*-
from tinyscript import itertools, os, re, subprocess, sys
from tinyscript.helpers import get_parser, Path #execute_and_log as run, get_parser, Path
from tinyscript.report import *

from ...helpers.items import load_yaml_config
from ...helpers.rendering import render


__all__ = ["Scenario"]

_REPLACEMENTS = {'dir': "directory", 'max': "maximum", 'min': "minimum"}
_TOOLS = [f.basename for f in Path("~/.opt/tools", expand=True).listdir() if f.basename not in ["?", "startup"]]
__cls = None
__initialized = False


def __init_metascenario():
    global Scenario
    from ...helpers.items import MetaItem  # this imports the lazy object Proxy (with its specific id(...))
    MetaItem.__name__                       # this forces the initialization of the Proxy
    from ...helpers.items import MetaItem  # this reimports the loaded metaclass (hence, getting the right id(...))
    
    class MetaScenario(MetaItem):
        def __getattribute__(self, name):
            # this masks some attributes for child classes (e.g. Scenario.registry can be accessed, but when the
            #  registry of child classes is computed, the child classes, e.g. RF, won't be able to access RF.registry)
            if name in ["get", "iteritems", "mro", "registry"] and self._instantiable:
                raise AttributeError("'%s' object has no attribute '%s'" % (self.__name__, name))
            return super(MetaScenario, self).__getattribute__(name)
        
        @property
        def source(self):
            if not hasattr(self, "_source"):
                self.source = None
            return self._source
        
        @source.setter
        def source(self, path):
            p = Path(str(path or config['scenarios']), expand=True)
            if hasattr(self, "_source") and self._source == p:
                return
            Scenario.__name__  # force initialization
            cls, self._source, glob = Scenario, p, globals()
            # remove the child classes of the former registry from the global scope
            for child in getattr(self, "registry", []):
                glob.pop(child.cname, None)
            cls.registry = []
            # start parsing items of cls
            for name, data in load_yaml_config(p):
                # put the related scenario in module's globals()
                d = dict(cls.__dict__)
                for a in ["get", "iteritems", "mro", "registry"]:
                    d.pop(a, None)
                i = glob[name] = type(name, (cls, ), d)
                i._instantiable = True
                # now set attributes from YAML parameters
                for k, v in data.items():
                    setattr(i, "_" + k if k == "source" else k, v)
                glob['__all__'].append(name)
                cls.registry.append(i())
    return MetaScenario
lazy_load_object("MetaScenario", __init_metascenario)


def __init_scenario():
    global __cls, __initialized, MetaScenario
    from ...helpers.items import Item  # this imports the lazy object Proxy (with its specific id(...))
    Item.__name__                       # this forces the initialization of the Proxy
    from ...helpers.items import Item  # this reimports the loaded metaclass (hence, getting the right id(...))
    MetaScenario.__name__
    
    class Scenario(Item, metaclass=MetaScenario):
        """ Scenario abstraction. """
        def __getattribute__(self, name):
            # this masks some attributes for child instances in the same way as for child classes
            if name in ["get", "iteritems", "mro", "registry"] and self._instantiable:
                raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))
            return super(Scenario, self).__getattribute__(name)
        
        def _compose_readme(self):
            #TODO
            pass
        
        def run(self, fail_stop=True, commit=True, experiment=None, quiet=False, **kwargs):
            r = [Title(re.sub(r"[-_]", " ", self.name).title())]
            if hasattr(self, "description"):
                r += [Text(self.description)]
            obj = "The objective is to " + self.objective[0].lower() + self.objective[1:]
            r += [Blockquote(obj), Rule(), Text("\n")]
            render(*r)
            # parse steps, ask for inputs and run
            args = None
            for n, step in enumerate(self.steps):
                if not isinstance(step, dict) or "name" not in step or "command" not in step:
                    raise ValueError("bad format ; step should be a dictionary with keys 'name' and 'command'")
                name, cmd, params = step['name'], step['command'], step.get('parameters', {})
                tool = cmd.split()[0]
                if tool not in _TOOLS:
                    tool = config['workspace'].joinpath("scripts", tool)
                    if not tool.exists():
                        raise ValueError(f"bad command ; should be one of {'|'.join(_TOOLS)} or a script from the "
                                         f"'scripts' folder of the experiment")
                    if re.search(r"<.*?>", cmd):
                        raise ValueError(f"bad script command ; does not support inputs or state variables")
                else:
                    parser = get_parser(tool, logger=Scenario.logger, command=cmd)
                    args = parser.parse_args(namespace=args)
                    cmd = args._command
                    # STATE VARIABLES to be added to arguments namespace
                    if tool == "dataset" and hasattr(args, "command") and args.command == "ingest":
                        args.name = args.prefix + RENAME_FUNCTIONS[args.rename](Path(args.folder).stem)
                # handle parameters if any
                if isinstance(step, dict) and 'parameters' in step:
                    cmds = []
                    for values in itertools.product(*list(params.values())):
                        cmd2 = cmd
                        for option, value in zip(p.keys(), values):
                            if isinstance(value, bool):
                                if value:
                                    cmd2 += f" --{option.replace('_', '-')}"
                            elif isinstance(value, str):
                                cmd2 += f" --{option} '{value}'"
                            elif isinstance(value, (list, tuple)):
                                cmd2 += f" --{option} {','.join(value)}"
                            else:
                                cmd2 += f" --{option} {value}"
                        cmds.append(cmd2)
                else:
                    cmds = [cmd]
                # once the command is ready, run it
                render(Text(f"{n+1}. {name}"), Text("\n"))
                try:
                    for cmd in cmds:
                        if quiet:
                            #out, err, retc = run(cmd, logger=Scenario.logger, silent=[" \[INFO\] ", "^\-\s[a-z]+"])
                            from shlex import split
                            p = subprocess.Popen(split(cmd), stdout=subprocess.PIPE, universal_newlines=True)
                            for l in iter(p.stdout.readline, ""):
                                print(l)
                            p.stdout.close()
                            retc = p.wait()
                            if retc != 0 and fail_stop:
                                Scenario.logger.critical("Scenario stopped")
                                sys.exit(retc)
                            # STATE VARIABLES to be added to arguments namespace (based on tool's output)
                            if tool == "model" and hasattr(args, "command") and args.command == "train":
                                for l in out.split(b"\n"):
                                    if b"Name: " in l:
                                        args.name = l.split(b"Name: ")[1].decode()
                        else:
                            print(cmd)
                            os.system(cmd)
                        if experiment is not None:
                            experiment.commit(cmd, force=True)
                except KeyboardInterrupt:
                    Scenario.logger.warning("Scenario interrupted")
                    sys.exit(0)
            print("")
        
        @property
        def source(self):
            if not hasattr(self, "_source"):
                self.__class__._source = ""
            return self.__class__._source
    # ensure it initializes only once (otherwise, this loops forever)
    if not __initialized:
        __initialized = True
        # initialize the registry of scenarios from the default source (~/.packing-box/conf/scenarios.yml)
        Scenario.source = None  # needs to be initialized, i.e. for the 'experiment' tool as the registry is used for
    if __cls:                   #  choices, even though the relying YAML config can be tuned via --scenarios-set
        return __cls
    __cls = Scenario
    return Scenario
lazy_load_object("Scenario", __init_scenario)

