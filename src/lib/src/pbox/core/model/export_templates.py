# -*- coding: UTF-8 -*-
""" Templates of the files of a tool exported from a trained model (see pbox.core.model.export). They are filled with
     string.Template ($name placeholders ; $$ for a literal $). """

INIT = '''# -*- coding: UTF-8 -*-
""" $title: $description.

Exported from the Packing Box (https://github.com/packing-box/docker-packing-box) on $date, from the model '$model'.
 It embeds the computation of the $n_features features the model uses and the model itself, with no dependency outside
 the Python standard library.
"""
import math

from . import features as _F
from . import model as _M
from .portable.expressions import compute, namespace, ExtractionError
from .portable.formats import detect


__all__ = ["extract", "predict", "vector", "ExtractionError", "UnsupportedFormat", "__version__"]
__version__ = "$version"

FEATURES, FORMATS = _F.FEATURES, sorted(_F.DEFINITIONS)
_HELPERS = namespace(_F.HELPERS)
_DEFINITIONS = {fmt: {name: dict(d, expr=compile(d['expr'], f"<{name}>", "eval")) for name, d in defs.items()}
                for fmt, defs in _F.DEFINITIONS.items()}


class UnsupportedFormat(ValueError):
    """ The executable is not of a format the model was trained for. """


def extract(data):
    """ Compute the features of an executable given its bytes. Returns (format, {feature: value}). """
    fmt = detect(data)
    if fmt not in _DEFINITIONS:
        raise UnsupportedFormat(f"unsupported format ({fmt or 'not an executable'}) ; supported: {', '.join(FORMATS)}")
    raw = {}
    if _F.PEFEATS.get(fmt):
        from .portable.extractors.pefeats import pefeats_values, PefeatsError
        try:
            raw['pefeats'] = [None if x == -1. else x for x in pefeats_values(data)]
        except PefeatsError as e:
            raise ExtractionError(f"pefeats: {e}") from None
    def _lazy():
        d = dict(_F.CONSTANTS.get(fmt, {}))
        if _F.PARSER.get(fmt):
            from .portable.parsers.pe import PE, PEError
            try:
                binary = PE(data, *_F.SECTION_NAMES[fmt])
            except PEError as e:
                raise ExtractionError(f"parsing: {e}") from None
            d['binary'] = d[_F.GROUPS[fmt]] = binary
        return d
    return fmt, compute(_DEFINITIONS[fmt], raw, _lazy, _HELPERS, FEATURES)


def vector(values):
    """ Model input from feature values, as the Packing Box builds it: features in alphabetical order, missing values
         (None, NaN) set to -1, booleans as 0/1. """
    x = []
    for name in FEATURES:
        v = values.get(name)
        if v is None or isinstance(v, float) and math.isnan(v):
            v = -1.
        if not isinstance(v, (bool, int, float)):
            raise ExtractionError(f"{name}: non-numeric value {v!r}")
        x.append(float(v))
    return x


def predict(data, with_features=False):
    """ Predict the class of an executable given its bytes (or its path).

    Returns {'format': ..., 'class': ..., 'label': ..., 'probability': ... (or None), 'features': ... (if asked)}. """
    if not isinstance(data, (bytes, bytearray)):
        with open(data, "rb") as f:
            data = f.read()
    from .portable.runtime import predict as _predict
    fmt, values = extract(bytes(data))
    cls, proba = _predict(_M.MODEL, vector(values))
    r = {'format': fmt, 'class': cls, 'label': _M.LABELS.get(str(cls), str(cls)),
         'probability': None if proba is None else proba[cls]}
    if with_features:
        r['features'] = values
    return r
'''

MAIN = '''# -*- coding: UTF-8 -*-
""" Command line of $title. """
import argparse
import json
import sys
from pathlib import Path

from . import __version__, predict, ExtractionError, UnsupportedFormat


def _files(paths, recursive):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in (p.rglob("*") if recursive else p.iterdir()) if f.is_file())
        else:
            yield p


def main(argv=None):
    parser = argparse.ArgumentParser("$tool", description="$description")
    parser.add_argument("path", nargs="+", help="executable or folder of executables")
    parser.add_argument("-f", "--features", action="store_true", help="include the feature values (with --json)")
    parser.add_argument("-j", "--json", action="store_true", help="output one JSON object per executable")
    parser.add_argument("-r", "--recursive", action="store_true", help="look for executables in subfolders")
    parser.add_argument("-V", "--version", action="version", version=f"$tool {__version__}")
    args = parser.parse_args(argv)
    errors = 0
    for f in _files(args.path, args.recursive):
        try:
            r = predict(f, with_features=args.features)
        except (OSError, ExtractionError, UnsupportedFormat) as e:
            errors += 1
            r = {'error': str(e)}
        if args.json:
            print(json.dumps({'path': str(f), **r}, default=str))
        elif 'error' in r:
            print(f"{f}: error: {r['error']}")
        else:
            p = "" if r['probability'] is None else f" ({r['probability']:.3f})"
            print(f"{f}: {r['label']}{p}")
    return 1 if errors else 0


def run():
    """ Entry point of the .pyz file (zipapp ignores the return value of the function it calls). """
    sys.exit(main())


if __name__ == "__main__":
    run()
'''

RUN = '''# -*- coding: UTF-8 -*-
""" Entry point for building a native executable with Nuitka (see the Makefile and .github/workflows/build.yml). """
import sys

from $package.__main__ import main


sys.exit(main())
'''

PYPROJECT = '''[build-system]
requires = ["setuptools>=70"]
build-backend = "setuptools.build_meta"

[project]
name = "$tool"
version = "$version"
description = "$description"
readme = "README.md"
license = "GPL-3.0-or-later"
requires-python = ">=3.10"
dependencies = []

[project.scripts]
$tool = "$package.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
'''

MAKEFILE = '''TOOL := $tool

.PHONY: test pyz native clean

test:
\tpython -m pytest -q

pyz:
\tmkdir -p dist
\tpython -m zipapp src -m "$package.__main__:run" -p "/usr/bin/env python3" -o dist/$$(TOOL).pyz

native:
\tpython -m pip install nuitka
\tPYTHONPATH=src python -m nuitka --onefile --assume-yes-for-downloads --include-package=$package \\
\t  --output-dir=dist --output-filename=$$(TOOL) run.py

clean:
\trm -rf dist build *.build *.dist *.onefile-build
'''

WORKFLOW = '''name: build

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:
  workflow_dispatch:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: python -m pip install pytest
      - run: python -m pytest -q

  pyz:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: make pyz
      - uses: actions/upload-artifact@v4
        with:
          name: $tool-pyz
          path: dist/$tool.pyz

  native:
    needs: test
    strategy:
      fail-fast: false
      matrix:
        # one runner per OS and CPU (Python cannot be cross-compiled) ; add other runners as needed (e.g. an Intel
        #  macOS one if your GitHub plan provides it)
        os: [ubuntu-latest, ubuntu-24.04-arm, windows-latest, macos-latest]
    runs-on: $${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: python -m pip install nuitka
      - name: Build with Nuitka
        shell: bash
        run: |
          PYTHONPATH=src python -m nuitka --onefile --assume-yes-for-downloads --include-package=$package \\
            --output-dir=dist --output-filename=$tool-$${{ runner.os }}-$${{ runner.arch }} run.py
      - name: Smoke test
        shell: bash
        run: ./dist/$tool-$${{ runner.os }}-$${{ runner.arch }}* --version
      - uses: actions/upload-artifact@v4
        with:
          name: $tool-$${{ runner.os }}-$${{ runner.arch }}
          path: dist/$tool-*
'''

GITIGNORE = '''__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
*.build/
*.dist/
*.onefile-build/
.pytest_cache/
'''

TESTS = '''# -*- coding: UTF-8 -*-
""" Tests of $title against the values recorded at export time by the Packing Box (tests/data/expected.json):
     - the model, on feature vectors of the training dataset (predictions identical, probabilities within 1e-9) ;
     - the feature extraction, on the executables checked at export time, if they are available: set SAMPLES to a
        folder holding them (they are matched by SHA256 ; they are not distributed with the tool).
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

import $package
from $package.portable.runtime import predict as _predict
from $package import model as _M

EXPECTED = json.loads(Path(__file__).parent.joinpath("data", "expected.json").read_text())


@pytest.mark.parametrize("row", EXPECTED['model'], ids=lambda r: r['id'])
def test_model(row):
    cls, proba = _predict(_M.MODEL, row['x'])
    assert str(cls) == str(row['class'])
    if row['probability'] is not None:
        assert all(abs(proba[c] - p) <= 1e-9 for c, p in zip(_M.MODEL['steps'][-1]['classes'], row['probability']))


def _samples():
    folder = os.environ.get("SAMPLES")
    if not folder:
        return {}
    found = {}
    for f in Path(folder).rglob("*"):
        if f.is_file():
            found.setdefault(hashlib.sha256(f.read_bytes()).hexdigest(), f)
    return found


@pytest.mark.skipif(not os.environ.get("SAMPLES"), reason="set SAMPLES to a folder with the executables")
def test_features():
    samples, checked = _samples(), 0
    for sha256, exp in EXPECTED['files'].items():
        if sha256 not in samples:
            continue
        r = $package.predict(samples[sha256], with_features=True)
        assert r['format'] == exp['format']
        assert $package.vector(r['features']) == exp['x']
        assert str(r['class']) == str(exp['class'])
        checked += 1
    assert checked > 0, "none of the expected executables was found in SAMPLES"


def test_not_an_executable(tmp_path):
    from $package.__main__ import main
    f = tmp_path / "file.txt"
    f.write_text("not an executable")
    assert main([str(f)]) == 1
'''

README = '''# $title

$description.

This tool was exported from the [Packing Box](https://github.com/packing-box/docker-packing-box) on $date. It embeds the
 computation of the features its model uses and the model itself, in pure Python (standard library only): it runs
 anywhere Python 3.10+ runs, and can be compiled into a single native executable.

## Usage

```console
$$ $tool path/to/executable [more executables or folders...]
path/to/executable: $example_label
```

Options: `--json` (one JSON object per executable), `--features` (with `--json`, include the feature values),
 `--recursive` (look into subfolders). The exit code is 1 if an executable could not be processed.

It can also be used as a library:

```python
>>> import $package
>>> $package.predict("path/to/executable")
{'format': 'PE32', 'class': ..., 'label': ..., 'probability': ...}
```

## Installation and build

- From source: `pip install .`, or `python -m $package ...` from the `src` folder.
- Single file for any platform with Python 3.10+: `make pyz` gives `dist/$tool.pyz`, run with `python $tool.pyz ...`.
- Native executable: `make native` (with [Nuitka](https://nuitka.net)) gives `dist/$tool` for the current platform.
  Python cannot be cross-compiled: the GitHub Actions workflow (`.github/workflows/build.yml`) builds one executable
  per platform (Linux x86-64 and ARM64, Windows, macOS) and the `.pyz` file, after running the tests.

## Model

| | |
| --- | --- |
| Source model | `$model` |
| Algorithm | $algorithm |
| Parameters | $parameters |
| Preprocessing | $preprocessors |
| Classes | $classes |
| Formats | $formats |
| Training dataset | `$dataset` ($executables executables ; labels: $counts) |

$performance
## Features

The $n_features features below are computed for each executable, in this order (missing values are set to -1):

$features_table

$sources

## Validation

$validation

## License

GNU General Public License v3.0 or later (see `LICENSE`), as the Packing Box from which this code comes.
'''
