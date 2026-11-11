# -*- coding: UTF-8 -*-
""" Snapshot of the abstractions of the current workspace, as plain data (JSON-serializable).

The TUI runs this module in a subprocess ('python -m pboxtools.tui.snapshot OUTPUT'), so that each refresh reflects
 the current state of the workspace (a newly opened experiment, an edited definition, ...): pbox loads its
 configuration and registries once per process.
"""
import json
import sys
from pathlib import Path


__all__ = ["collect", "ITEM_TYPES", "DEFINITION_TYPES"]

ITEM_TYPES = ["analyzer", "detector", "packer", "unpacker"]
DEFINITION_TYPES = ["algorithms", "alterations", "features", "scenarios"]


def _s(v):
    """ Make a value JSON-friendly (paths, sets, custom objects). """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _s(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_s(x) for x in v]
    return str(v)


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _items(cls, type_):
    from pbox import collapse_formats, expand_formats
    rows = []
    for i in cls.registry:
        try:
            formats = ",".join(collapse_formats(*expand_formats(*getattr(i, "formats", ["All"]))))
        except Exception:
            formats = ",".join(map(str, getattr(i, "formats", [])))
        # NB: the details are rendered by the TUI from these fields ; the items' help() (tinyscript reports) inspects
        #     the call stack for each element, which takes seconds for the whole registry
        extra = {}
        for attr in ("categories", "multiclass", "vote", "weak", "maintained", "version", "author", "license",
                     "alterations", "packers", "references"):
            if (v := getattr(i, attr, None)) not in (None, "", []):
                extra[attr] = _s(v)
        rows.append({
            'name': i.name, 'key': i.cname, 'status': i.status, 'formats': formats,
            'source': _s(getattr(i, "source", "")) or "", 'description': _s(getattr(i, "description", "")) or "",
            'comment': _s(getattr(i, "comment", "")) or "", 'parent': getattr(i.__class__, "parent", None),
            'enabled': i.status in cls._enabled, 'extra': extra,
        })
    return {'file': str(cls.config), 'type': type_, 'rows': rows}


def _algorithms():
    from pbox import Algorithm
    rows = []
    for a in Algorithm.registry:
        extra = {k: _s(getattr(a, k)) for k in ("base", "multiclass", "labelling", "parameters", "boolean")
                 if getattr(a, k, None) not in (None, "", [], {})}
        rows.append({'name': a.name, 'key': a.cname, 'category': _s(getattr(a, "category", "")) or "",
                     'description': _s(getattr(a, "description", "")) or "", 'extra': extra})
    return {'file': str(Algorithm.config), 'rows': rows}


def _scenarios():
    from pbox import Scenario
    rows = [{'name': s.name, 'key': s.cname, 'description': _s(getattr(s, "description", "")) or ""}
            for s in Scenario.registry]
    return {'file': str(Scenario.config), 'rows': rows}


def _per_format(cls, flag):
    """ Features and alterations are registered per executable format ; merge them by name. """
    cls()  # trigger the lazy loading of the registry
    merged = {}
    for fmt, entries in (cls.registry or {}).items():
        for name, e in entries.items():
            r = merged.setdefault(name, {'name': name, 'key': name, 'formats': [], flag: False,
                                         'description': "", 'result': {}, 'extra': {}})
            r['formats'].append(fmt)
            r[flag] = r[flag] or bool(getattr(e, flag, False))
            r['description'] = r['description'] or _s(getattr(e, "description", "")) or ""
            r['result'][fmt] = _s(getattr(e, "result", ""))
            for attr in ("comment", "source", "significant", "loop", "parameters"):
                if (v := getattr(e, attr, None)) not in (None, "", [], {}):
                    r['extra'][attr] = _s(v)
    from pbox import collapse_formats
    rows = sorted(merged.values(), key=lambda r: r['name'])
    for r in rows:
        try:
            r['formats'] = ",".join(collapse_formats(*r['formats']))
        except Exception:
            r['formats'] = ",".join(r['formats'])
    return {'file': str(cls.config), 'flag': flag, 'rows': rows}


def _datasets():
    from pbox import Dataset
    rows = []
    for p in Dataset.iteritems(instantiate=False):
        p, md = Path(str(p)), _json(Path(str(p), "metadata.json"))
        rows.append({'name': p.name, 'path': str(p), 'formats': ",".join(md.get('formats', [])),
                     'executables': md.get('executables', 0), 'counts': md.get('counts', {}),
                     'sources': md.get('sources', []), 'fileless': not p.joinpath("files").is_dir(),
                     'altered': p.joinpath("alterations.json").is_file(), 'metadata': md})
    return {'folder': str(_cfg('datasets')), 'rows': sorted(rows, key=lambda r: r['name'])}


def _models():
    from pbox import Model
    rows = []
    for p in Model.iteritems(instantiate=False):
        p, md = Path(str(p)), _json(Path(str(p), "metadata.json"))
        algo, ds = md.get('algorithm', {}), md.get('dataset', {})
        perf = []
        try:
            perf = [l for l in Path(p, "performance.csv").read_text().splitlines() if l.strip()]
        except OSError:
            pass
        rows.append({'name': p.name, 'path': str(p), 'algorithm': algo.get('name', ""),
                     'dataset': ds.get('name', ""), 'formats': ",".join(ds.get('formats', [])),
                     'executables': ds.get('executables', 0), 'performance': perf, 'metadata': md})
    return {'folder': str(_cfg('models')), 'rows': sorted(rows, key=lambda r: r['name'])}


def _experiments(current):
    from pbox import Experiment
    rows = []
    try:
        paths = list(Experiment.iteritems(instantiate=False))
    except Exception:
        paths = []
    for p in paths:
        p = Path(str(p))
        readme = ""
        try:
            readme = p.joinpath("README.md").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass
        count = lambda f: len([x for x in p.joinpath(f).iterdir() if x.is_dir()]) if p.joinpath(f).is_dir() else 0
        conf = sorted(x.name for x in p.joinpath("conf").iterdir()) if p.joinpath("conf").is_dir() else []
        rows.append({'name': p.name, 'path': str(p), 'current': current is not None and p == Path(current),
                     'datasets': count("datasets"), 'models': count("models"), 'conf': conf, 'readme': readme})
    return {'folder': str(_cfg('experiments')), 'rows': sorted(rows, key=lambda r: r['name'])}


def _config():
    rows = []
    for section in config.sections():
        for opt, _, value, metavar, descr in config.iteroptions(section):
            rows.append({'name': opt, 'key': opt, 'section': section, 'value': _s(value), 'metavar': metavar or "",
                         'description': descr or "", 'option': "--" + opt.replace("_", "-")})
    return {'rows': rows}


def _cfg(key):
    return config[key]


def collect():
    """ Collect every abstraction of the current workspace. Errors are recorded per category, so that one broken
         definition file does not hide the others. """
    import pbox  # noqa: F401  (sets the 'config' builtin)
    from pbox import Analyzer, Alterations, Detector, Features, Packer, Unpacker
    exp_env = Path("~/.packing-box/experiment.env").expanduser()
    current = exp_env.read_text().strip() if exp_env.is_file() else None
    current = current or None
    snap = {'workspace': str(config['workspace']), 'experiment': current, 'experiments': str(config['experiments']),
            'home': str(Path("~/.packing-box").expanduser()), 'categories': {}, 'errors': {}}
    tasks = {
        'analyzers': lambda: _items(Analyzer, "analyzer"),
        'detectors': lambda: _items(Detector, "detector"),
        'packers': lambda: _items(Packer, "packer"),
        'unpackers': lambda: _items(Unpacker, "unpacker"),
        'algorithms': _algorithms,
        'alterations': lambda: _per_format(Alterations, "apply"),
        'features': lambda: _per_format(Features, "keep"),
        'scenarios': _scenarios,
        'datasets': _datasets,
        'models': _models,
        'experiments': lambda: _experiments(current),
        'config': _config,
    }
    for name, f in tasks.items():
        try:
            snap['categories'][name] = f()
        except Exception as e:
            snap['errors'][name] = f"{e.__class__.__name__}: {e}"
    return snap


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python -m pboxtools.tui.snapshot OUTPUT.json")
    import logging
    logging.disable(logging.CRITICAL)  # pbox logs to the console ; the TUI only needs the data
    snap = collect()
    Path(sys.argv[1]).write_text(json.dumps(snap, default=str))


if __name__ == "__main__":
    main()
