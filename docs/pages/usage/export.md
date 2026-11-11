# Model Export

A trained model can be exported as a standalone tool: a repository holding the model and the computation of the features it uses, in pure Python (standard library only). The tool runs without the Packing Box and without any dependency, wherever Python 3.10 or later runs, and can be compiled into a single native executable.

```console
┌──[user@packing-box]──[/mnt/share]────────
$ model export my-model -o ./my-tool
[INFO] Model checked on 200 rows of the dataset (0 mismatch) ; feature extraction checked on 30 executables (0 mismatch)
[INFO] Tool exported to /mnt/share/my-tool
```

| Option | Description |
| --- | --- |
| `-o`, `--output` | Folder of the repository (default: `./<tool name>`) ; it must not exist or be empty |
| `-n`, `--tool-name` | Name of the tool (default: derived from the model's name) |
| `-s`, `--samples` | Executables or folders to check the feature extraction on (default: files of the model's dataset, if it has files) |
| `-m`, `--max-samples` | Maximum number of executables to check the feature extraction on (default: 30) |
| `-f`, `--force` | Write into an existing folder, and keep the tool even if the checks fail |

As the other `model` commands, it uses the features definitions of the current workspace: open the experiment in which the model was trained before exporting it.

## What is exported

Only what the model needs is embedded:

- the definitions of the features the model uses, with the features they depend on, for each format of the model ;
- the extractors these features need: the PE parser (`pbox.portable`), the Python port of [`pefeats`](https://github.com/packing-box/pefeats), or both ;
- the evaluation of the feature expressions, with only the helpers they use ;
- the model, as plain data (no pickle), with only the inference code of its algorithm.

```
my-tool/
├── src/my_tool/
│   ├── __init__.py       # extract(), vector(), predict()
│   ├── __main__.py       # command line
│   ├── features.py       # generated: features definitions, constants, section names
│   ├── model.py          # generated: model parameters
│   └── portable/         # the modules of pbox.portable the model needs
├── tests/                # tests against the values recorded at export time
├── .github/workflows/build.yml
├── Makefile              # make test | pyz | native
├── pyproject.toml
├── README.md             # model card: model, dataset, performance, features, validation
└── LICENSE
```

```console
$ my-tool samples/
samples/calc.exe: not packed (0.998)
samples/upx_calc.exe: packed (1.000)
$ my-tool --json --features samples/upx_calc.exe
{"path": "samples/upx_calc.exe", "format": "PE32", "class": 1, "label": "packed", "probability": 1.0, "features": {...}}
```

## What can be exported

| | Supported |
| --- | --- |
| Algorithms | Decision Tree, Random Forest, Extra Trees, SVM (`SVC`: linear, polynomial, RBF and sigmoid kernels), Multi-Layer Perceptron (scikit-learn) |
| Preprocessors | MinMax, Standard, MaxAbs and Robust scalers, Normalizer, PCA, or none |
| Formats | PE (`PE32`, `PE64`, `.NET`) |
| Features | Those computed from `pefeats` and from the properties of the parsed binary that `pbox.portable` provides (sections, entry point, entropies, imports, exports, TLS, ...), with the usual helpers of the expressions |

When a model cannot be exported, the command lists the reasons (algorithm, formats, features or properties of the parsed binary that are not supported).

## Checks

Before keeping the repository, the export checks it against the Packing Box:

- **Model**: on rows of the training dataset, the predictions of the tool are identical to those of scikit-learn, and its probabilities equal up to `1e-9`.
- **Features**: on executables (the given samples, or files of the dataset), the format, the features and the prediction computed by the tool are identical to those of the Packing Box with the `portable` parser (which gives the same values as LIEF and `pefeats`).

The results are written in the README of the tool, and recorded in `tests/data/expected.json` for the tests of the tool. If a check fails, the repository is renamed with the `.failed` suffix, unless `--force` is used.

## Building

- `make pyz` gives a single file, `dist/<tool>.pyz`, that runs on any platform with Python 3.10 or later.
- `make native` compiles a native executable for the current platform with [Nuitka](https://nuitka.net).
- Python cannot be cross-compiled: the GitHub Actions workflow of the repository builds the native executables (Linux x86-64 and ARM64, Windows, macOS) and the `.pyz` file, after running the tests.

## Differences with the Packing Box

- The tool only computes the features its model needs. With the Packing Box, the failure of an extractor that the model does not use (e.g. `pefeats` on a PE file without sections) prevents computing any feature ; the tool computes the features it needs anyway.
- The tool reads the executable's format from its headers, with the same rules as the `file` command (libmagic) that the Packing Box relies on.
