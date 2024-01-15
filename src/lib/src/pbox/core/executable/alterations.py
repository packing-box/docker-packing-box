# -*- coding: UTF-8 -*-
from tinyscript import logging

from ...helpers import dict2, expand_formats, get_data, get_format_group, load_yaml_config, MetaBase


__all__ = ["Alterations"]


class Alteration(dict2):
    _multi_expr = True
    
    def __call__(self, executable, namespace, **kwargs):
        self._exe, l = executable, self._logger
        l.debug(f"applying alterations to {executable.stem}%s..." % ["", f" ({self.loop} steps)"][self.loop > 1])
        # loop the specified number of times or accross sections or simply once if not specified
        # IMPORTANT NOTE: looping accross sections is done on a primary parsing of the executable, otherwise, a
        #                  generator of sections could change during iteration because of alterations such as adding a
        #                  new section ; this should be considered when designing an alteration
        iterator = range(self.loop) if isinstance(self.loop, int) else \
                   list(executable.parse(self.parser).__iter__()) if self.loop == "sections" else \
                   None  # invalid 'loop' value
        if iterator is None:
            l.warning("Bad 'loop' value (should be positive integer or 'sections')")
            return
        for i in iterator:
            parsed = executable.parse(self.parser)
            namespace.update({'binary': parsed, executable.group.lower(): parsed})
            if self.loop == "sections":
                namespace['section'] = i
            try:
                func = super().__call__(namespace, silent=self.fail=="stop")
                if func is None:
                    continue
                if not isinstance(func, (list, tuple)):
                    func = [func]
                for f in func:
                    if not callable(f):
                        continue
                    f(parsed, l)
            except:
                # unexpected error, thrown exception and leave
                if self.fail == "error":
                    raise
                # in this case, the error is not known, exception shall be thrown and next iterations shall be tried
                elif self.fail == "continue":
                    continue
                # in this situation, it is known that the loop will fail at some point, no exception trace is needed
                elif self.fail == "stop":
                    break
                # in this case, warn the user and stop
                elif self.fail == "warn":
                    l.warning()
                    break
                else:
                    raise ValueError("Bad 'fail' value (should be one of: continue|error|stop)")
            l.debug(f"rebuilding binary (build config: {parsed._build_config})")
            parsed.build()
    
    # default values on purpose not set via self.setdefault(...)
    @cached_property
    def apply(self):
        return self.get('apply', False)
    
    @cached_property
    def fail(self):
        return self.get('fail', "error")
    
    @cached_property
    def loop(self):
        return self.get('loop', 1)
    
    # 'parser' parameter in the YAML config has precedence on the globally configured parser
    @cached_property
    def parser(self):
        try:
            p = self._exe.shortgroup
            delattr(self, "_exe")
        except AttributeError:
            p = "default"
        return self.get('parser', config['%s_parser' % p])


class Alterations(list, metaclass=MetaBase):
    """ This class parses the YAML definitions of alterations to be applied to executables. It works as a list that
         contains the names of alterations applied to the executable given an input executable. """
    namespaces = None
    registry   = None
    
    def __init__(self, exe=None, select=None, warn=False):
        a, l = Alterations, self.__class__.logger
        # parse YAML alterations definition once
        if a.registry is None:
            src = a.source  # WARNING! this line must appear BEFORE a.registry={} because the first time that the
                            #           source attribute is called, it is initialized and the registry is reset to None
            l.debug(f"loading alterations from {src}...")
            a.namespaces, a.registry, dsbcnt = {}, {}, 0
            # collect properties that are applicable for all the alterations
            for name, params in load_yaml_config(src):
                r = params.pop('result', {})
                # consider most specific alterations first, then those for intermediate format classes and finally the
                #  collapsed class "All"
                for flist in [expand_formats("All"), [f for f in FORMATS.keys() if f != "All"], ["All"]]:
                    for fmt in flist:
                        expr = r.get(fmt) if isinstance(r, dict) else str(r)
                        if expr:
                            alt = Alteration(params, name=name, result=expr, logger=l)
                            for subfmt in expand_formats(fmt):
                                a.registry.setdefault(subfmt, {})
                                a.registry[subfmt][alt.name] = alt
                            if not alt.apply:
                                l.debug(f"{alt.name} is disabled")
                                dsbcnt += 1
            # consider re-enabling only alterations for which there is no more than one alteration per format ;
            #  if multiple alterations on the same format, leave these disabled
            reenable = Alterations.names()
            for fmt, alts in a.registry.items():
                if len(alts) > 1:
                    for alt in alts.keys():
                        try:
                            reenable.remove(alt)
                        except ValueError:
                            continue
            for single_alt in reenable:
                l.debug(f"re-enabling {single_alt} as it is a single applicable alteration")
                for fmt, alts in a.registry.items():
                    found = False
                    for alt, obj in alts.items():
                        if single_alt == alt:
                            obj.apply = found = True
                            break
                    if found:
                        break
                dsbcnt -= 1
            tot = len(Alterations.names())
            l.debug(f"{tot} alterations loaded ({tot-dsbcnt} enabled)")
        # check the list of selected alterations if relevant, and filter out bad names (if warn=True)
        for name in (select or [])[:]:
            if name not in a.registry[exe.format]:
                msg = f"Alteration '{name}' does not exist"
                if warn:
                    l.warning(msg)
                    select.remove(name)
                else:
                    raise ValueError(msg)
        if exe is not None:
            parser = None
            # prepare the namespace if not done yet for the target executable format
            if exe.format not in a.namespaces:
                from .modifiers import Modifiers
                # create the namespace for the given executable format (including the 'grid' helper defined hereafter)
                a.namespaces[exe.format] = {'grid': grid, 'logger': l}
                # add constants specific to the target executable format
                a.namespaces[exe.format].update(get_data(exe.format))
                # add format-related data and modifiers
                a.namespaces[exe.format].update(Modifiers()[exe.group])
                # add format-specific alterations
                a.namespaces[exe.format].update(a.registry[exe.format])
            for name, alteration in a.registry[exe.format].items():
                if select is None and not alteration.apply or select is not None and name not in select:
                    continue
                # run the alteration given the format-specific namespace
                try:
                    alteration(exe, a.namespaces[exe.format])
                    l.debug(f"applied alteration '{alteration.name}'")
                    self.append(name)
                except NotImplementedError as e:
                    l.warning(f"'{e.args[0]}' is not supported yet for parser '{exe.parsed.parser}'")
                except Exception as e:
                    l.exception(e)
    
    @staticmethod
    def names(format="All"):
        Alterations(None)  # force registry initialization
        l = []
        for c in expand_formats(format):
            l.extend(list(Alterations.registry.get(c, {}).keys()))
        return sorted(list(set(l)))


def grid(modifier, params_grid, **eval_data):
    """ Run another modifier multiple times with different parameters.
    
    :param modifier:    modifier function
    :param params_grid: list of dictionnaries containing the parameter values
    """
    def _wrapper(parser=None, executable=None, **kw):
        d, e = {}, executable
        d.update(Alterations.namespaces[e.format])
        d.update(eval_data)
        for params in params_grid:
            parse_executable(parser, e, d)
            d.update(params)
            parser = modifier(d, parser=parser, executable=e, **kw)
            for p in params.keys():
                del d[p]
        return parser
    return _wrapper

