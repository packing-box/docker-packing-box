# -*- coding: UTF-8 -*-
from tinyscript import logging
from tinyscript.report import *

from ...helpers import dict2, expand_formats, get_data, get_format_group, load_yaml_config, render, MetaBase

set_exception("ExecutableBuildError", "RuntimeError")


__all__ = ["Alterations"]


class Alteration(dict2):
    _multi_expr = True
    
    def __call__(self, executable, namespace, **kwargs):
        self._exe, parsed, l = executable, executable.parse(self.parser), self._logger
        l.debug(f"applying alterations to {executable.stem}" \
                f"{['', f' ({self.loop} steps)'][isinstance(self.loop, int) and self.loop > 1]}...")
        # loop the specified number of times or accross sections or simply once if not specified
        # IMPORTANT NOTE: looping accross sections is done on a primary parsing of the executable, otherwise, a
        #                  generator of sections could change during iteration because of alterations such as adding a
        #                  new section ; this should be considered when designing an alteration
        iterator = range(self.loop) if isinstance(self.loop, int) else \
                   list(parsed.__iter__()) if self.loop == "sections" else None  # invalid 'loop' value
        if iterator is None:
            l.warning("Bad 'loop' value (should be positive integer or 'sections')")
            return
        if self.build not in ["incremental", "once"]:
            l.warning("Bad 'build' value (should be 'incremental' or 'once') ; set to default ('once')")
            self.build = "once"
        with self._exe.open('rb') as f:
            before = [_ for _ in f.read()]
        # incremental build function
        def _build(p, b, n=1):
            l.debug(f"> rebuilding binary (build config: {parsed._build_cfg})...")
            a, _log = p.build(), getattr(l, {'error': "error", 'warn': "warning"}.get(self.fail, "debug"))
            if not a:
                _log(f"{parsed.path}: build failed")
            elif b == a:
                _log(f"{parsed.path}: unchanged after build")
            try:
                p = executable.parse(self.parser)
            except UnknownFormatError:
                l.warning(f"{self._exe}: corrupted (unknown format{['', ' after alteration'][n > 0]})")
                return
            return a
        # start iterating over 'iterator'
        built = False
        for n, i in enumerate(iterator):
            l.debug(f"> starting iteration #{n+1}...")
            if n == 0 or self.build == "incremental":
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
            except Exception as e:
                # unexpected error, thrown exception and leave
                if self.fail == "error":
                    raise
                # in this case, the error is not known, exception shall be thrown and next iterations shall be tried
                elif self.fail == "continue":
                    l.debug(f"{self._exe}")
                    l.exception(e)
                    continue
                # in this situation, it is known that the loop will fail at some point, no exception trace is needed
                elif self.fail == "stop":
                    l.debug(f"> finished in {n} iteration{['','s'][n>1]}")
                    break
                # in this case, warn the user and stop
                elif self.fail == "warn":
                    l.warning(f"{self._exe}: {e} ({e.__class__.__name__})")
                    break
                else:
                    raise ValueError("Bad 'fail' value (should be one of: continue|error|stop)")
            if self.build == "incremental":
                parsed, before = _build(parsed, before, n)
                built = True
        if not built:
            _build(parsed, before)
    
    # default values on purpose not set via self.setdefault(...)
    @cached_property
    def apply(self):
        return self.get('apply', False) or self.get('order', 0) > 0
    
    @cached_property
    def build(self):
        return self.get('build', "once")
    
    @cached_property
    def fail(self):
        return self.get('fail', "error")
    
    @cached_property
    def loop(self):
        return self.get('loop', 1)
    
    @cached_property
    def order(self):
        return self.get('order', 0)
    
    # 'parser' parameter in the YAML config has precedence on the globally configured parser
    @cached_property
    def parser(self):
        try:
            p = self._exe.shortgroup
        except AttributeError:
            p = "default"
        return self.get('parser', config[f'{p}_parser'])


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
        if exe is not None:
            d = a.registry[exe.format]
            # check the list of selected alterations if relevant, and filter out bad names (if warn=True)
            for name in (select or list(d.keys()))[:]:
                if name not in d:
                    msg = f"Alteration '{name}' does not exist"
                    if warn:
                        l.warning(msg)
                        select.remove(name)
                    else:
                        raise ValueError(msg)
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
                a.namespaces[exe.format].update(d)
            def_order = max(d.values(), key=lambda x: x.order).order + 1
            for name, alteration in sorted(d.items(), key=lambda x: x[1].order or def_order):
                if select is None and not alteration.apply or select is not None and name not in select:
                    continue
                # run the alteration given the format-specific namespace
                try:
                    alteration(exe, a.namespaces[exe.format])
                    info = [f" ({alteration.order})", ""][alteration.order == def_order]
                    l.debug(f"applied alteration '{alteration.name}'{info}")
                    self.append(name)
                except NotImplementedError as e:
                    l.warning(f"'{e.args[0]}' is not supported yet for parser '{exe.parsed.parser}'")
                except Exception as e:
                    l.exception(e)
    
    @classmethod
    def show(cls, **kw):
        """ Show an overview of the alterations. """
        from ...helpers.utils import pd
        cls.logger.debug(f"computing alterations overview...")
        formats = list(Alterations.registry.keys())
        # collect counts
        counts = {}
        for fmt in formats:
            counts[fmt] = len(Alterations.registry[fmt])
        render(Section(f"Counts"), Table([list(counts.values())], column_headers=formats))
    
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

