# -*- coding: UTF-8 -*-
import builtins
import statistics
from tinyscript import functools, inspect, itertools, logging, random, re, string
from tinyscript.helpers import get_terminal_size, is_file, is_folder, is_generator, is_iterable, reduce, \
                               set_exception, txt_terminal_render, zeropad, Path, TempPath
from tinyscript.helpers.expressions import WL_NODES

from .files import Locator
from .formats import expand_formats, format_shortname
from .utils import benchmark, entropy, pd

set_exception("NotInstantiable", "TypeError")


__all__ = ["dict2", "load_yaml_config", "tag_from_references", "Item", "MetaBase", "MetaItem", "References"]

_EMPTY_DICT = {}
_EVAL_NAMESPACE = {k: getattr(builtins, k) for k in ["abs", "all", "any", "bool", "divmod", "float", "hash", "hex",
                                                     "id", "int", "list", "map", "next", "oct", "ord", "pow", "range",
                                                     "range2", "round", "set", "str", "sum", "tuple", "type"]}
for k in dir(statistics):
    if not k.startswith("_") and isinstance(getattr(statistics, k), type(lambda: 0)):
        _EVAL_NAMESPACE[k] = getattr(statistics, k)
_EVAL_NAMESPACE['avg'] = _EVAL_NAMESPACE['fmean']
_EXPLODE_QUERY_FIELDS = ["categories", "formats", "tags"]
_WL_EXTRA_NODES = ("arg", "arguments", "keyword", "lambda")


def __init_yaml(yaml):
    # Custom Dumper to save special objects
    class CustomDumper(yaml.SafeDumper):
        def represent_list(self, data):
            args = [yaml.representer.SafeRepresenter.represent_data(self, item) for item in data]
            return yaml.SequenceNode(tag="tag:yaml.org,2002:seq", value=args,
                                     flow_style=all(not isinstance(x, (str, list, range, range2)) for x in data))
        # Represent range as: !!python/object/new:range [start, stop]
        def represent_range(self, data):
            args = [yaml.ScalarNode(tag="tag:yaml.org,2002:int", value=str(data.start)),
                    yaml.ScalarNode(tag="tag:yaml.org,2002:int", value=str(data.stop))]
            if data.step != 1:
                args.append(yaml.ScalarNode(tag="tag:yaml.org,2002:int", value=str(data.step)))
            return yaml.SequenceNode(tag="tag:yaml.org,2002:python/object/new:range", value=args, flow_style=True)
        # Represent range2 as: !!python/object/apply:range2 [start, stop, step]
        def represent_range2(self, data):
            # Represent range as a flow-style list (one line)
            ff = lambda v: f"{v:.16g}" if '.' in str(v) or 'e' in str(v) else f"{v}."
            args = [yaml.ScalarNode(tag="tag:yaml.org,2002:float", value=str(arg)) for arg in \
                    [ff(data.start), ff(data.stop), ff(data.step)]]
            return yaml.SequenceNode(tag="tag:yaml.org,2002:python/object/apply:range2", value=args, flow_style=True)
        # inspired by https://stackoverflow.com/a/44284819/3786245
        def write_line_break(self, data=None):
            super().write_line_break(data)
            if len(self.indents) == 1:
                super().write_line_break()
    yaml.add_representer(list, CustomDumper.represent_list, Dumper=CustomDumper)
    yaml.add_representer(range, CustomDumper.represent_range, Dumper=CustomDumper)
    yaml.add_representer(range2, CustomDumper.represent_range2, Dumper=CustomDumper)
    yaml.CustomDumper = CustomDumper
    return yaml
lazy_load_module("yaml", postload=__init_yaml)


_concatn  = lambda l, n: reduce(lambda a, b: a[:n]+b[:n], l, stop=lambda x: len(x) > n)
_ent      = lambda s, *a, **kw: entropy(s.encode() if isinstance(s, str) else s, *a, **kw)
_isin     = lambda s, l: s in l  # used with expressions, for avoiding error with '"..." in [...]' when loading as YAML
_last     = lambda o, sep=".": str(o).split(sep)[-1]
_len      = lambda i: sum(1 for _ in i) if is_generator(i) else len(i)
_max      = lambda l, *a, **kw: None if len(l2 := [x for x in l if x is not None]) == 0 else max(l2, *a, **kw)
_min      = lambda l, *a, **kw: None if len(l2 := [x for x in l if x is not None]) == 0 else min(l2, *a, **kw)
_repeatn  = lambda s, n: (s * (n // len(s) + 1))[:n]
_sec_name = lambda s: getattr(s, "real_name", getattr(s, "name", s))
_size     = lambda exe, ratio=.1, blocksize=512: round(int(exe['size'] * ratio) / blocksize + .5) * blocksize
_val      = lambda o: getattr(o, "value", o)


def _fail_safe(f):
    # useful e.g. when using a function like 'avg' or 'max' and the input is an empty list
    @functools.wraps(f)
    def __wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return
    return __wrapper


def _randbytes(n, unique=True):
    if not unique:
        return random.randbytes(n)
    if n > 256:
        raise ValueError("Cannot produce more than 256 distinct bytes")
    s, alphabet = b"", bytearray([i for i in range(256)])
    for i in range(n):
        c = random.choice(alphabet)
        s += bytes([c])
        alphabet.remove(c)
    return s


def _select(apply_func=None):
    def _wrapper(lst=(), random_lst=(), inclusions=(), exclusions=()):
        """ Helper for selecting the first argument of a list given inclusions or exclusions, then choosing randomly
             among a given list when the first list is consumed. """
        _list = lambda l: list(l()) if callable(l) else list(l) if is_iterable(l) else [l]
        _map = lambda l: list(map(apply_func, l)) if isinstance(apply_func, type(lambda: 0)) else l
        # ensure that inputs are lists, allowing to make them from a function if desired
        exc, inc, lst, rlst = _list(exclusions), _list(inclusions), _list(lst), _list(random_lst)
        # map the function to be applied to all input lists
        lst, rlst = _map(lst), _map(rlst)
        exc, inc = _map(exc), _map(inc) or lst + rlst
        # now iterate over the list of choices given first exclusions then inclusions
        for x in lst:
            if x in exc or x not in inc:
                continue
            return x
        # if no entry returned yet, return an element from the random list given exclusions
        return random.choice(rlst, exc, False)
    return _wrapper


class dict2(dict):
    """ Simple extension of dict for defining callable items. """
    def __init__(self, idict=_EMPTY_DICT, **kwargs):
        self.setdefault("name", UNDEF_RESULT)
        self.setdefault("description", "")
        self.setdefault("result", self['name'])
        for f, v in getattr(self.__class__, "_fields", {}).items():
            self.setdefault(f, v)
        logger = idict.pop('logger', kwargs.pop('logger', null_logger))
        super().__init__(idict, **kwargs)
        self.__dict__ = self
        dict2._logger = logger
        if self.result == UNDEF_RESULT:
            raise ValueError(f"{self.name}: 'result' shall be defined")
    
    def __call__(self, data, silent=False, **kwargs):
        l = dict2._logger
        d = {k: getattr(random, k) for k in ["choice", "randint", "randrange", "randstr"]}
        d.update({'apply': _apply, 'concatn': _concatn, 'ent': _ent, 'failsafe': _fail_safe, 'find': str.find,
                  'hex2bytes': bytearray.fromhex, 'isin': _isin, 'joinstr': "".join, 'last': _last, 'len': _len,
                  'lower': lambda s: s.lower(), 'max': _max, 'min': _min, 'printable': string.printable,
                  'randbytes': _randbytes, 'repeatn': _repeatn, 'select': _select(),
                  'select_section_name': _select(_sec_name), 'size': _size, 'value': _val, 'zeropad': zeropad})
        d.update(_EVAL_NAMESPACE)
        d.update(data)
        kwargs.update(getattr(self, "parameters", {}))
        bm, bmt = kwargs.pop('benchmark', False), kwargs.pop('benchmark_threshold', .0)
        # execute an expression from self.result (can be a single expression or a list of expressions to be chained)
        def _exec(expr):
            try:
                d.pop('__builtins__', None)  # security issue ; ensure builtins are removed !
                # replace marker "*" before function names by failsafe(...)
                expr = re.sub(r"([a-zA-Z][a-zA-Z0-9_-]*)\*\(", r"failsafe(\1)(", expr)
                # now execute
                r = eval2(expr, d, {}, whitelist_nodes=WL_NODES + _WL_EXTRA_NODES)
                if len(kwargs) == 0:  # means no parameter provided
                    return r
            except ForbiddenNodeError as e:  # this error type shall always be reported
                l.warning(f"[{self.name}] Bad expression:\n{expr}")
                l.error(f"{e}")
                raise
            except NameError as e:
                name = str(e).split("'")[1]
                l.debug(f"[{self.name}] '{name}' is either not computed yet or mistaken")
                raise
            except Exception as e:
                getattr(l, ["warning", "debug"][silent])(f"[{self.name}] Bad expression:\n{expr}")
                if not silent:
                    l.exception(e)
                w = get_terminal_size()[0]
                l.debug("Variables:\n- %s" % \
                        "\n- ".join(string.shorten(f"{k}({type(v).__name__})={v}", w - 2) for k, v in d.items()))
                raise
            try:
                return r(**kwargs)
            except Exception as e:
                getattr(l, ["warning", "debug"][silent])(f"Bad function: {self.result}")
                if not silent:
                    l.error(str(e))
        # now execute expression(s) ; support for multiple expressions must be explicitely enabled for the class
        if not getattr(self.__class__, "_multi_expr", False) and isinstance(self.result, (list, tuple)):
            raise ValueError(f"List of expressions is not supported for the result of {self.__class__.__name__}")
        results = self.result if isinstance(self.result, (list, tuple)) else [self.result]
        if bm:
            retv, times = list(zip(*[benchmark(_exec)(r) for r in results]))
            if sum(times)*1000 > bmt:
                l.info(f"{self.name}:\n{NEWLINE.join(f'  {1000*t:.03f}ms\t    {r}' for r, t in zip(results, times))}")
        else:
            retv = [_exec(r) for r in results]
        return retv[0] if len(retv) == 1 else tuple(retv)


class MetaBase(type):
    """ This metaclass allows to iterate names over the class-bound registry of the underlying abstraction.
         It also adds a class-level 'config' attribute that can be used to reset the registry. It also provides a few
         methods to select or export items from the registry.
        
        This metaclass is fitted for abstractions for which the child items should not be callable as classes.
            
        By contrast to MetaItem, setting self.config does not trigger registry recomputation ; this is done lazily at
         the first instantiation of the class.
        
        Important note: if a class holds this metaclass and has _has_registry=False, then this class shall be a 
                         dictinoary as it is considered as the registry itself. """
    def __iter__(self):
        self()  # trigger registry's lazy initialization
        temp = []
        for base in (self.registry.values() if getattr(self, "_has_registry", True) else [self.registry]):
            for name in base.keys():
                if name not in temp:
                    yield name
                    temp.append(name)
    
    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            if '_config' not in self.__dict__ or name.startswith("_"):
                raise AttributeError(f"'{self.__name__}' object has no attribute '{name}'") from None
            temp, has_reg = [], self.__dict__.get('_has_registry', True)
            reg = self.__dict__.get('_registry') if has_reg else self()
            for base in (reg.values() if has_reg else [reg]):
                has_field = False
                for data in base.values():
                    if hasattr(data, name) or name in data:
                        has_field = True
                        break
                if not has_field:
                    raise AttributeError(f"'{self.__name__}' object has no attribute '{name}'") from None
                for data in base.values():
                    v = getattr(data, name) if hasattr(data, name) else data.get(name)
                    for sv in (v if isinstance(v, (list, tuple, set)) else [v]):
                        if sv not in [None, ""] and sv not in temp:
                            temp.append(sv)
            return sorted(temp)
    
    def _filter(self, format="All", query=None, fields=None, index="name", merge_deps=False, **kw):
        """ Filter a subset of items based on a Pandas query and return the filtered dataframe. """
        _l, has_deps, has_reg = self.logger, False, getattr(self, "_has_registry", True)
        # from the registry, based on the required formats, collect field names from all items
        field_names = []
        for c in (expand_formats(format) if has_reg else [None]):
            for record in self.registry.get(c, self.registry).values():
                has_deps = has_deps or hasattr(record, "dependencies") and len(record.dependencies) > 0
                for k in dict(record).keys():
                    if k not in field_names:
                        field_names.append(k)
        _l.debug(f"Fields found: {', '.join(field_names)}")
        # from the registry, based on the required formats, build a DataFrame combining all items
        for i, c in enumerate(expand_formats(format) if has_reg else [None]):
            d = {}
            for record in self.registry.get(c, self.registry).values():
                r = dict(record)  # dict(...) converts the dict2 instance to its full version
                for k in field_names:
                    d.setdefault(k, [])
                    d[k].append(r.get(k) if k in _EXPLODE_QUERY_FIELDS else str(r.get(k)))
            df = pd.DataFrame.from_dict(d) if i == 0 else pd.concat([df, pd.DataFrame.from_dict(d)], ignore_index=True)
        if fields:
            if index not in fields:
                fields = [index] + fields
            df = df[fields]
        # explode based on pre-defined fields
        for field in _EXPLODE_QUERY_FIELDS:
            if field in df.columns:
                _l.debug(f"Exploding the dataframe based on the '{field}' field")
                df = df.explode(field)
        # now filter the resulting DataFrame with the given query
        if query is not None and query.lower() != "all":
            try:
                df = df.query(query)
            except Exception as e:
                if isinstance(e, (AttributeError, KeyError, NameError, SyntaxError, TypeError, ValueError)) or \
                   Path(inspect.getfile(e.__class__)).dirname.parts[-2:] == ('pandas', 'errors'):
                    _l.error(f"Bad Pandas query expression: {e}")
                    _l.warning("No entry filtered ; aborting...")
                    return None, {}
                else:
                    raise
        if len(df) == 0:
            _l.warning("No data")
            return None, {}
        df = df.drop_duplicates(index)
        _l.debug(f"Dataframe queried:\n{df.describe()}\n{df}")
        # then collect dependencies too
        deps = {}
        if has_deps:
            for c in (expand_formats(format) if has_reg else [None]):
                for name, record in self.registry.get(c, self.registry).items():
                    if name not in df.name.values:
                        continue
                    deps.setdefault(name, [])
                    for dep in getattr(record, "dependencies", []):
                        if dep not in deps[name]:
                            deps[name].append(dep)
        # if requested, merge the dependencies in the dataframe
        if merge_deps:
            d, reg = {}, self.registry.get(c, self.registry)
            for name in sorted(set(x for l in deps.values() for x in l)):
                r = dict(reg[name])
                for k in field_names:
                    d.setdefault(k, [])
                    d[k].append(r.get(k) if k in _EXPLODE_QUERY_FIELDS else str(r.get(k)))
            df = pd.concat([df, pd.DataFrame.from_dict(d)], ignore_index=True)
        return df, deps
    
    def _select(self, format="All", query=None, fields=None, index="name", text=False, split_on=None, **kw):
        """ Select a subset based on a Pandas query and return it as a dictionary. """
        df, all_deps = self._filter(format=format, query=query, fields=fields, index=index, **kw)
        if df is None:
            return
        deps_list = sorted(set(x for l in all_deps.values() for x in l))
        # only keep the interesting parts of the configuration file based on the filtered DataFrame's list of names
        nmap = getattr(self, "names_map", {})
        values = set(nmap.get(k, k) for k in df[index].values)
        d = {k: v for k, v in load_yaml_config(self._config, auto_tag=False) if k in values}
        d2 = {k: v for k, v in load_yaml_config(self._config, auto_tag=False) if k in deps_list and k not in d.keys()}
        # if split_on is defined (e.g. split_on="category" to make a "[...].yml" folder with one "[value].yml" file per
        #  'catgegory' value), collect possible values
        if split_on is not None:
            try:
                splitv = set(v[split_on] for v in d.values())
                all_deps = {}  # reset collected dependencies as, in this case, they are handled through 'd2' as a
                               #  separate config
            except KeyError:
                _l.warning(f"No field '{split_on}' ; cannot split")
                split_on = None
        for config in ([d] if split_on is None else \
                       [{k: v for k, v in d.items() if v[split_on] == s} for s in splitv] + [d2]):
            # collect values in the selected items for rebuilding the defaults
            attrs = {}
            for k, v in config.items():
                for attr, val in v.items():
                    attrs.setdefault(attr, [])
                    if val not in attrs[attr]:
                        attrs[attr].append(val)
            # compute the defaults based on collected values
            defaults = {}
            for attr, vals in attrs.items():
                if len(vals) >= 1:
                    for val in vals:
                        if (nval := sum(1 for v in config.values() if val in v.values())) >= .7 * len(config):
                            defaults[attr] = val
                            for v in config.values():
                                if nval == len(config) or v.get(attr) == val:
                                    v.pop(attr, None)
            # compute dependencies if no split required ; otherwise, 'd2' holds the dependencies to be handled as a
            #  separate config
            deps = {}
            if len(all_deps) > 0:
                for k in config.keys():
                    for dep in all_deps.get(k, []):
                        try:
                            deps_list.remove(dep)
                        except ValueError:
                            continue
                        try:
                            deps[dep] = {k: v for k, v in d2[dep].items() if k not in defaults.keys() or \
                                                                             v != defaults[k]}
                            deps[dep]['keep'] = False
                        except KeyError:
                            continue
            deps = {k: deps[k] for k in sorted(deps.keys())}
            # special case: config has only 1 item, hence its attributes get moved to defaults ; set it back
            if len(config) == 1:
                config, defaults = {list(config.keys())[0]: defaults}, {}
            # return a string if text is requested, otherwise yield items
            if not text:
                yield defaults, config, deps
            from yaml import dump, CustomDumper
            _h = lambda s: f"{'#'*120}\n#{s.upper().center(118)}#\n{'#'*120}\n"
            # compose the YAML configuration
            l1, l2 = len(config), len(df)
            count = [l1, f"{l1} definitions, {l2} {self.__name__.lower()}"][l1 != l2]
            cfg = _h(f"{self.__name__} set ({count})")
            # - put the defaults first, if any
            cfg += dump({'defaults': defaults}, Dumper=CustomDumper, width=float("inf")).rstrip() + "\n\n\n" \
                   if len(defaults) > 0 else ""
            # - then write the collected items
            cfg += dump(config, Dumper=CustomDumper, default_style=None, width=float("inf")).rstrip()
            # - finally, add dependencies, if any
            cfg += "\n\n\n" + _h("dependencies") + dump(deps, Dumper=CustomDumper, width=float("inf")).rstrip() \
                   if split_on is None and len(deps) > 0 else ""
            # if data was split (e.g. on 'category'), then yield the name of the config as the value from the split
            #  field (e.g. defaults['category'] => "header")
            yield cfg, defaults.get(split_on)
    
    def export(self, output="output.csv", format="All", query=None, fields=None, index="name", **kw):
        """ Export items from the registry based on the given executable format, filtered with the given query, with
             only the selected fields, to the given output format. """
        if output in EXPORT_FORMATS.keys():
            output = f"{self.__name__.lower()}-export.{output}"
        kw = {}
        df, _ = self._filter(format=format, query=query, fields=fields, index=index, merge_deps=True, **kw)
        if (ext := Path(output).extension[1:]) != "pickle":
            kw['index'] = False
        elif ext in ["json", "yml"]:
            kw.update(orient="records", indent=2)
        elif ext == "csv":
            kw.update(sep=";", index=False)
        getattr(df.reset_index(), "to_" + EXPORT_FORMATS.get(ext, ext))(output, **kw)
    
    def list(self, **kw):
        """ List all the available items. """
        from tinyscript.report import Section, Table
        from .rendering import render
        self()  # trigger registry's lazy initialization
        temp, d = [], []
        for base in (self.registry.values() if getattr(self, "_has_registry", True) else [self.registry]):
            for name, data in base.items():
                if name not in temp:
                    d.append([name, data.get('description', "")])
                    temp.append(name)
        render(Section(f"{self.__name__}{['','s'][len(d) > 1]} ({len(d)})"),
               Table(sorted(d, key=lambda x: string.natural_key()(x[0])), column_headers=["Name", "Description"]),
               force=True)
    
    def select(self, output=None, format="All", query=None, fields=None, split_on=None, **kw):
        """ Select a subset based on a Pandas query and either display or dump it. """
        cname = self.__name__.lower()
        if split_on is not None and output is None:
            output = f"{cname}.yml"
        if output is not None:
            dst = Locator(f"conf://{output}", new=True)
            if dst.is_samepath(self._config):
                self.logger.warning("Destination is identical to source ; aborting...")
                return
            if split_on is not None:
                dst.mkdir()
        for cfg, name in self._select(format=format, query=query, fields=fields, text=True, split_on=split_on, **kw):
            if output is None:
                print(txt_terminal_render(cfg, pad=(1, 2), syntax="yaml"))
            else:
                with (dst if split_on is None else dst.joinpath(f"{name or cname}.yml")).open('wt') as f:
                    f.write(cfg)
    
    def test(self, files=None, keep=False, **kw):
        """ Tests on some executable files. """
        from tinyscript.helpers import execute_and_log as run
        from ..core.executable import Executable
        d = TempPath(prefix=f"{self.__name__.lower()}-tests-", length=8)
        self.__disp = []
        self()  # force registry computation
        for fmt in expand_formats("All"):
            if fmt not in self.registry.keys():
                self.logger.warning(f"no {self.__name__.lower().rstrip('s')} defined yet for '{fmt}'")
                continue
            l = [f for f in files if Executable(f).format in self._formats_exp] if files else TEST_FILES.get(fmt, [])
            if len(l) == 0:
                continue
            self.logger.info(fmt)
            for exe in l:
                exe = Executable(exe, expand=True)
                tmp = d.joinpath(exe.filename)
                self.logger.debug(exe.filetype)
                run(f"cp {exe} {tmp}")
                n = tmp.filename
                try:
                    self(Executable(tmp))
                    self.logger.success(n)
                except Exception as e:
                    if isinstance(e, KeyError) and exe.format is None:
                        self.logger.error(f"unknown format ({exe.filetype})")
                        continue
                    self.logger.failure(n)
                    self.logger.exception(e)
        if not keep:
            self.logger.debug(f"rm -f {d}")
            d.remove()
    
    @property
    def config(self):
        if not hasattr(self, "_config"):
            self.config = None  # use the default config from 'config'
        return self._config
    
    @config.setter
    def config(self, path):
        p = Path(str(path or config[self.__name__.lower()]), expand=True)
        if hasattr(self, "_config") and self._config == p:
            return
        self._config = p
        if getattr(self, "_has_registry", True):
            self._registry = None  # reset the registry
    
    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            n = self.__name__.lower()
            self._logger = logging.getLogger(n)
            logging.setLogger(n)
        return self._logger
    
    @property
    def names(self):
        try:
            self.__names
        except AttributeError:
            self.__names = string.sorted_natural(list(set(x for x in self)))
        return self.__names
    
    @property
    def registry(self):
        return self.__dict__.get('_registry') if self.__dict__.get('_has_registry', True) else self()


class References(dict, metaclass=MetaBase):
    _has_registry = False
    
    def __init__(self):
        try:
            self.__initialized
        except AttributeError:
            for name, params in load_yaml_config(self.__class__.config):
                self[name] = params
            self.__initialized = True
    
    def items(self):
        for name, data in super().items():
            d = {'name': name}
            for k, v in data.items():
                d[k] = v
            yield name, d
    
    def values(self):
        for _, data in self.items():
            yield data
    
    @classmethod
    def show(cls, **kw):
        """ Show an overview of the references. """
        from ...helpers.utils import pd
        cls.logger.debug(f"computing references overview...")
        #TODO
        #render(Section(f"Counts"), Table([list(counts.values())], column_headers=formats))


def _apply(f):
    """ Simple decorator for applying an operation to the result of the decorated function. """
    def _wrapper(op):
        def _subwrapper(*a, **kw):
            return op(f(*a, **kw))
        return _subwrapper
    return _wrapper


def _init_metaitem():
    class MetaItem(type):
        """ This metaclass uses a list-based registry that contains the child items from the target abstraction.
             For instance, Detector.registry will hold the list of all the classes that inherit from Detector, by
             contrast to MetaBase that handles a dictionary-based registry holding subdictionaries of child items that
             are not directly callable as classes.
            
            This metaclass is fitted for abstractions for which the child items should not be callable as classes.
            
            By contrast to MetaBase, setting self.config immediately triggers registry recomputation. """
        __fields__ = {}
        
        def __getattribute__(self, name):
            if name.startswith("_"):
                return super(MetaItem, self).__getattribute__(name)
            # this masks some attributes for child classes (e.g. Packer.registry can be accessed, but when the registry
            #  of child classes is computed, the child classes, e.g. UPX, won't be able to get UPX.registry)
            if name in ["get", "iteritems", "mro", "registry"] and self._instantiable:
                raise AttributeError(f"'{self.__name__}' object has no attribute '{name}'")
            if not self._instantiable and name in [n.lstrip("_") for n in self.__fields__]:
                temp = []
                for item in self.registry:
                    v = getattr(item, name, None)
                    for sv in (v if isinstance(v, (list, tuple, set)) else [v]):
                        if isinstance(sv, dict):
                            for ssv in [f"{k}: {va}" for k, va in sv.items()]:
                                if ssv not in [None, ""] and ssv not in temp:
                                    temp.append(ssv)
                            continue
                        if sv not in [None, ""] and sv not in temp:
                            temp.append(sv)
                return sorted(temp)
            return super(MetaItem, self).__getattribute__(name)
        
        def _filter(self, query=None, fields=None, index="cname", **kw):
            """ Filter a subset of items based on a Pandas query and return the filtered dataframe. """
            _l = self.logger
            # from the registry, based on the required formats, build a DataFrame combining all items
            for i, record in enumerate(self.registry):
                d = {index: getattr(record, index)}
                if index == "cname":
                    d['name'] = record.name
                for k in self.__fields__.keys():
                    k2 = k.lstrip("_") if k.startswith("_") else k
                    d.setdefault(k2, [])
                    v = getattr(record, k, None) if k in _EXPLODE_QUERY_FIELDS else str(getattr(record, k, None))
                    if k == "formats":
                        v = list(set(x for l in [expand_formats(x) for x in v] for x in l))
                    d[k2].append(v)
                df = pd.DataFrame.from_dict(d) if i == 0 else \
                     pd.concat([df, pd.DataFrame.from_dict(d)], ignore_index=True)
            if fields:
                if index not in fields:
                    fields = [index] + ([[], ["name"]][index == "cname"]) + fields
                df = df[fields]
            # explode based on pre-defined fields
            for field in _EXPLODE_QUERY_FIELDS:
                if field in df.columns:
                    _l.debug(f"Exploding the dataframe based on the '{field}' field")
                    df = df.explode(field)
            # now filter the resulting DataFrame with the given query
            if query is not None and query.lower() != "all":
                try:
                    df = df.query(query)
                except Exception as e:
                    if isinstance(e, (AttributeError, KeyError, NameError, SyntaxError, TypeError, ValueError)) or \
                       Path(inspect.getfile(e.__class__)).dirname.parts[-2:] == ('pandas', 'errors'):
                        _l.error(f"Bad Pandas query expression: {e}")
                        _l.warning("No entry filtered ; aborting...")
                        return None
                    else:
                        raise
            if len(df) == 0:
                _l.warning("No data")
                return None
            df = df.drop_duplicates(index)
            _l.debug(f"Dataframe queried:\n{df.describe()}\n{df}")
            return df
        
        def _select(self, query=None, fields=None, index="cname", text=False, **kw):
            """ Select a subset based on a Pandas query and return it as a dictionary. """
            if (df := self._filter(query=query, fields=fields, index=index, **kw)) is None:
                return
            # only keep the interesting parts of the configuration file based on the filtered DataFrame's list of names
            config = {k: v for k, v in load_yaml_config(self._config, auto_tag=False) if k in set(df[index].values)}
            # collect values in the selected items for rebuilding the defaults
            attrs = {}
            for k, v in config.items():
                for attr, val in v.items():
                    attrs.setdefault(attr, [])
                    if val not in attrs[attr]:
                        attrs[attr].append(val)
            # compute the defaults based on collected values
            defaults = {}
            for attr, vals in attrs.items():
                if len(vals) >= 1:
                    for val in vals:
                        if (nval := sum(1 for v in config.values() if val in v.values())) >= .7 * len(config):
                            defaults[attr] = val
                            for v in config.values():
                                if nval == len(config) or v.get(attr) == val:
                                    v.pop(attr, None)
            # special case: config has only 1 item, hence its attributes get moved to defaults ; set it back
            if len(config) == 1:
                config, defaults = {list(config.keys())[0]: defaults}, {}
            # return a string if text is requested, otherwise return items
            if not text:
                return defaults, config
            from yaml import dump, CustomDumper
            _h = lambda s: f"{'#'*120}\n#{s.upper().center(118)}#\n{'#'*120}\n"
            # compose the YAML configuration
            l1, l2 = len(config), len(df)
            count = [l1, f"{l1} definitions, {l2} {self.__name__}{['','s'][l2 > 1]}"][l1 != l2]
            cfg = _h(f"{self.__name__}s set ({count})")
            # - put the defaults first, if any
            cfg += dump({'defaults': defaults}, Dumper=CustomDumper, width=float("inf")).rstrip() + "\n\n\n" \
                   if len(defaults) > 0 else ""
            # - then write the collected items
            cfg += dump(config, Dumper=CustomDumper, default_style=None, width=float("inf")).rstrip()
            # if data was split (e.g. on 'category'), then yield the name of the config as the value from the split
            #  field (e.g. defaults['category'] => "header")
            return cfg
        
        def export(self, output="output.csv", query=None, fields=None, index="name", **kw):
            """ Export items from the registry filtered with the given query, with only the selected fields, to the
                 given output format. """
            if output in EXPORT_FORMATS.keys():
                output = f"{self.__name__.lower()}-export.{output}"
            kw = {}
            df = self._filter(query=query, fields=fields, index=index, **kw)
            if (ext := Path(output).extension[1:]) != "pickle":
                kw['index'] = False
            elif ext in ["json", "yml"]:
                kw.update(orient="records", indent=2)
            elif ext == "csv":
                kw.update(sep=";", index=False)
            getattr(df.reset_index(), "to_" + EXPORT_FORMATS.get(ext, ext))(output, **kw)
        
        def get(self, item, error=True):
            """ Simple class method for returning the class of an item based on its name (case-insensitive). """
            for i in self.registry:
                if i.name == (item.name if isinstance(item, Item) else format_shortname(item, "-")):
                    return i
            if error:
                raise ValueError(f"'{item}' is not defined")
        
        def iteritems(self):
            """ Class-level iterator for returning enabled items. """
            for i in self.registry:
                try:
                    if i.status in i.__class__._enabled:
                        yield i
                except AttributeError:
                    yield i
        
        def list(self, **kw):
            """ List all the available items. """
            from tinyscript.report import Section, Table
            from .rendering import render
            d = [(a.cname, getattr(a, "description", "")) for a in self.registry]
            render(Section(f"{self.__name__}{['','s'][len(d) > 1]} ({len(d)})"),
                   Table(sorted(d, key=lambda x: string.natural_key()(x[0])), column_headers=["Name", "Description"]),
                   force=True)
        
        def select(self, output=None, query=None, fields=None, **kw):
            """ Select a subset based on a Pandas query and either display or dump it. """
            cname = self.__name__.lower()
            if output is not None:
                dst = Locator(f"conf://{output}", new=True)
                if dst.is_samepath(self._config):
                    self.logger.warning("Destination is identical to source ; aborting...")
                    return
            if cfg := self._select(query=query, fields=fields, text=True, **kw):
                if output is None:
                    print(txt_terminal_render(cfg, pad=(1, 2), syntax="yaml"))
                else:
                    with dst.open('wt') as f:
                        f.write(cfg)
        
        @property
        def config(self):
            if not hasattr(self, "_config"):
                self.config = None
            return self._config
        
        @config.setter
        def config(self, path):
            cfg_key, l = f"{self.__name__.lower()}s", self.logger
            # case 1: self is a parent class among Analyzer, Detector, ... ;
            #          then 'source' means the source path for loading child classes
            try:
                p = Path(str(path or config[cfg_key]), expand=True)
                if hasattr(self, "_config") and self._config == p:
                    return
                self._config = p
            # case 2: self is a child class of Analyzer, Detector, ... ;
            #          then 'source' is an attribute that comes from the YAML definition
            except KeyError:
                return
            # now make the registry from the given source path
            def _setattr(i, d):
                for k, v in d.items():
                    if k == "status":
                        setattr(i, k := f"_{k}", v)
                    elif hasattr(i, "parent") and k in ["install", "references", "steps"]:
                        nv = []
                        for l in v:
                            if l == "<from-parent>":
                                for l2 in getattr(glob[i.parent], k, []):
                                    nv.append(l2)
                            else:
                                nv.append(l)
                        setattr(i, k, nv)
                    else:
                        setattr(i, k, v)
                    if k not in self.__fields__.keys():
                        self.__fields__[k] = None
            # open the .conf file associated to the main class (i.e. Detector, Packer, ...)
            glob = inspect.getparentframe().f_back.f_globals
            # remove the child classes of the former registry from the global scope
            for cls in getattr(self, "registry", []):
                glob.pop(cls.cname, None)
            # reset the registry
            self.registry = []
            if not p.exists():
                l.warning(f"'{p}' does not exist ; set back to default")
                p, func = config.DEFAULTS['definitions'][cfg_key]
                p = func(p)
            # start parsing items of the target class
            _cache, cnt = {}, {'tot': 0, 'var': 0}
            l.debug(f"loading {cfg_key} from {p}...")
            for item, data in load_yaml_config(p, ("base", "install", "steps", "variants")):
                # ensure the related item is available in module's globals()
                #  NB: the item may already be in globals in some cases like pbox.items.packer.Ezuri
                if item not in glob:
                    d = dict(self.__dict__)
                    del d['registry']
                    glob[item] = type(item, (self, ), d)
                i = glob[item]
                i._instantiable = True
                # before setting attributes from the YAML parameters, check for 'base' ; this allows to copy all
                #  attributes from an entry originating from another item class (i.e. copying from Packer's equivalent
                #  to Unpacker ; e.g. UPX)
                base = data.get('base')  # detector|packer|unpacker ; DO NOT pop as 'base' is also used for algorithms
                if isinstance(base, str):
                    if (m := re.match(r"(?i)(detector|packer|unpacker)(?:\[(.*?)\])?$", str(base))):
                        data.pop('base')
                        base, bcls = m.groups()
                        base, bcls = base.capitalize(), bcls or item
                        if base == self.__name__ and bcls in [None, item]:
                            raise ValueError(f"{item} cannot point to itself")
                        if base not in _cache.keys():
                            _cache[base] = dict(load_yaml_config(base.lower() + "s"))
                        for k, v in _cache[base].get(bcls, {}).items():
                            # do not process these keys as they shall be different from an item class to another anyway
                            if k in ["steps", "status"]:
                                continue
                            setattr(i, k, v)
                    else:
                        raise ValueError(f"'base' set to '{base}' of {item} discarded (bad format)")
                # check for variants ; the goal is to copy the current item class and to adapt the fields from the
                #  variants to the new classes (note that on the contrary of base, a variant inherits the 'status'
                #  parameter)
                variants, vilist = data.pop('variants', {}), []
                # collect template parameters
                template = variants.pop('_template', {})
                if len(template) > 0:
                    attrs = [k[1:] for k in list(template.keys()) if k.startswith("_")]
                    iterables = [template.pop(f"_{k}") for k in attrs]
                    for values in itertools.product(*iterables):
                        vitem = "-".join(map(str, (item, ) + values))
                        # this create the new variant item if not present
                        variants.setdefault(vitem, {k: v for k, v in template.items()})
                        # this updates existing variant item with template's content
                        for k, v in itertools.chain(zip(attrs, values), template.items()):
                            variants[vitem].setdefault(k, v)
                # create variant classes in globals
                for vitem in variants.keys():
                    d = dict(self.__dict__)
                    del d['registry']
                    vi = glob[vitem] = type(vitem, (self, ), d)
                    vi._instantiable = True
                    vi.parent = item
                    vilist.append(vi)
                # now set attributes from YAML parameters
                for it in [i] + vilist:
                    _setattr(it, data)
                self.registry.append(i())
                cnt['tot'] += 1
                # overwrite parameters specific to variants
                for vitem, vdata in variants.items():
                    vi = glob[vitem]
                    _setattr(vi, vdata)
                    self.registry.append(vi())
                    cnt['var'] += 1
            varcnt = ["", f" ({cnt['var']} variants)"][cnt['var'] > 0]
            l.debug(f"{cnt['tot']} {cfg_key} loaded{varcnt}")
        
        @property
        def logger(self):
            if not hasattr(self, "_logger"):
                n = self.__name__.lower()
                self._logger = logging.getLogger(n)
                logging.setLogger(n)
            return self._logger
        
        @property
        def names(self):
            return string.sorted_natural(list(set(x.name for x in self.registry)))
    return MetaItem
lazy_load_object("MetaItem", _init_metaitem)


def _init_item():
    global MetaItem
    MetaItem.__name__  # force the initialization of MetaItem
    class Item(metaclass=MetaItem):
        """ Item abstraction. """
        _instantiable = False
        
        def __init__(self, **kwargs):
            cls = self.__class__
            self.cname = cls.__name__
            self.name = format_shortname(cls.__name__, "-")
            self.type = cls.__base__.__name__.lower()
        
        def __new__(cls, *args, **kwargs):
            """ Prevents Item from being instantiated. """
            if cls._instantiable:
                return object.__new__(cls, *args, **kwargs)
            raise NotInstantiable(f"{cls.__name__} cannot be instantiated directly")
        
        def __getattribute__(self, name):
            # this masks some attributes for child instances in the same way as for child classes
            if name in ["get", "iteritems", "mro", "registry"] and self._instantiable:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            return super(Item, self).__getattribute__(name)
        
        def __repr__(self):
            """ Custom string representation for an item. """
            return f"<{self.__class__.__name__} {self.type} at 0x{id(self):02x}>"
        
        def help(self):
            """ Returns a help message in Markdown format. """
            from tinyscript.report import Blockquote, Report, Section, Text
            md = Report()
            if getattr(self, "description", None):
                md.append(Text(self.description))
            if getattr(self, "comment", None):
                md.append(Blockquote("**Note**: " + self.comment))
            if getattr(self, "link", None):
                md.append(Blockquote("**Link**: " + self.link))
            if getattr(self, "references", None):
                md.append(Section("References"), List(*self.references, **{'ordered': True}))
            return md.md()
        
        @property
        def config(self):
            if self._instantiable and not isinstance(self._config, str):
                self.__class__._config = ""
            return self.__class__._config
        
        @property
        def logger(self):
            return self.__class__.logger
    return Item
lazy_load_object("Item", _init_item)


def load_yaml_config(cfg, no_defaults=(), parse_defaults=True, auto_tag=True):
    """ Load a YAML configuration, either as a single file or a folder with YAML files (in this case, loading
         defaults.yml in priority to get the defaults first). """
    from copy import deepcopy
    # local function for expanding defaults from child configurations
    def _set(config, defaults_=None):
        if parse_defaults:
            defaults = deepcopy(defaults_ or {})
            defaults.update(config.pop('defaults', {}))
            for params in [v for v in config.values()]:
                for default, value in defaults.items():
                    if default in no_defaults:
                        raise ValueError(f"default value for parameter '{default}' is not allowed")
                    if isinstance(value, dict):
                        # example advanced defaults configuration:
                        #   defaults:
                        #     keep:
                        #       match:
                        #         - ^entropy*   (do not keep entropy-based features)
                        #         - ^is_*       (do not keep boolean features)
                        #       value: false
                        value.pop('comment', None)
                        if set(value.keys()) == {'match', 'value'}:
                            v = value['value']
                            for pattern in value['match']:
                                for name2 in config.keys():
                                    # special case: boolean value ; in this case, set value for matching names and set
                                    #                                non-matching names to its opposite
                                    if isinstance(v, bool):
                                        # in the advanced example here above, entropy-based and boolean features will
                                        #  not be kept but all others will be
                                        config[name2].setdefault(default, v if re.search(pattern, name2) else not v)
                                        # this means that, if we want to keep additional features, we can still force
                                        #  keep=true per feature declaration
                                    elif re.search(pattern, name2):
                                        config[name2].setdefault(default, v)
                    else:
                        # example normal defaults configuration:
                        #   defaults:
                        #     keep:   false
                        #     source: <unknown>
                        params.setdefault(default, value)
        return config
    # local function for merging child configurations
    def _merge(config, k, v, keep=False):
        if k not in config.keys():
            config[k] = deepcopy(v)
        else:
            v1 = config[k]
            if isinstance(v1, dict):
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        _merge(v1, sk, sv)
                elif not keep:
                    config[k] = v
            elif isinstance(v1, (list, set, tuple)):
                if isinstance(v, (list, set, tuple)):
                    for sv in v:
                        if sv not in v1:
                            v1.append(sv)
                else:
                    v1.append(v)
            elif not keep:
                config[k] = v
    # local function for updating the main configuration
    def _update(base, folder, defaults, parent=()):
        defaults_, dflt = deepcopy(defaults), folder.joinpath("defaults.yml")
        if dflt.is_file():
            with dflt.open() as f:
                defaults_.update(yaml.load(f, Loader=yaml.Loader) or {})
        for cfg in folder.listdir(lambda p: p.extension == ".yml"):
            if cfg.stem == "defaults":
                continue
            with cfg.open() as f:
                config = yaml.load(f, Loader=yaml.Loader) or {}
            _set(config, defaults_)
            for k, v in config.items():
                _merge(base, k, v)
        for cfg_folder in folder.listdir(lambda p: p.is_dir()):
            _update(base, cfg_folder, defaults_, parent)
    # get the list of configs ; may be:
    #  - single YAML file
    #  - folder (and subfolders) with YAML files
    p = cfg if isinstance(cfg, Path) else Path(config[cfg])
    if p.is_dir():
        # parse YAML configurations according to the following rules:
        #  - when encountering a defaults.yml, its values become the defautls for the current folder and subfolders
        #  - when finding a defaults key in a .yml, these default values are merged with higher level defaults
        d = {}
        _update(d, p, {})
    else:
        try:
            with p.open() as f:
                d = _set(yaml.load(f, Loader=yaml.Loader) or {})
        except FileNotFoundError:
            raise OSError(f"Did you forget to prepend \"./\" to force a relative path ?")
    # collect properties that are applicable for all the other features
    for name, params in d.items():
        # handle the references attributes by checking if the "<...>" pattern is present and replace "..." with the
        #  related reference dictionary from References()
        if auto_tag:
            tag_from_references(params)
        yield name, params


def tag_from_references(data):
    if 'references' in data:
        tags = data.get('tags', [])
        for i, ref in enumerate(lst := data['references']):
            try:
                key = re.match(r"<(.*)?>$", ref).group(1)
                lst[i] = References().get(key, ref)
                tags.extend(lst[i].get('tags', []))
            except (AttributeError, TypeError):
                pass
        if len(tags) > 0:
            data['tags'] = string.sorted_natural(list(set(tags)))

