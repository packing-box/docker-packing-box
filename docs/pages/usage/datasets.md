# Datasets

*Datasets* are folders holding samples to be analyzed. They aim to organize input executables according to a [strict structure](#structure), including packing labels, original names, and so forth.

## Structure

A (normal) dataset folder holds the following files/folders:

```
[name]
  +-- files
  |     +-- {executables, renamed to their SHA256 hashes}
  +-- data.csv          # metadata and labels of the executable
  +-- metadata.json     # simple statistics about the dataset
```

A fileless dataset has the following structure:

```
[name]
  +-- data.csv          # metadata and labels of the executable
  +-- features.json     # dictionary of selected features and their descriptions
  +-- metadata.json     # simple statistics about the dataset
```

Fileless datasets can be used when we are sure of the features set to be computed. It can also save substantial disk space (such a dataset will at most take a few megabytes of data while the executables from the original one could occupy gigabytes).

## Tool

A [dedicated tool](https://github.com/dhondta/docker-packing-box/blob/main/files/tools/dataset) called `dataset` is provided with the [*Packing Box*](https://github.com/dhondta/docker-packing-box) to manipulate datasets. Its help message tells everything the user needs to get started.

```session
$ dataset --help
[...]
This tool aims to manipulate a dataset in many ways including its creation, enlargement, update, alteration, selection,
 merging, export or purge.
[...]
    alter               alter the target dataset with a set of transformations
    convert             convert the target dataset to a fileless one
    edit                edit the data file
    export              export packed samples from a dataset or export the dataset to a given format
    features            compute and view features
    fix                 fix a corrupted dataset
    ingest              ingest samples from a folder into new dataset(s)
    list                list all the datasets from the workspace
    make                add n randomly chosen executables from input sources to the dataset
    merge               merge two datasets
    plot                plot the distribution of a given feature or multiple features combined
    purge               purge a dataset
    remove              remove executables from a dataset
    rename              rename a dataset
    revert              revert a dataset to its previous state
    select              select a subset of the dataset
    show                get an overview of the dataset
    update              update a dataset with new executables
    view                view executables filtered from a dataset
[...]
```

## Visualization

The list of datasets in a folder can be viewed by using the command `dataset list [folder]` (`dataset list` will display all the datasets in the current folder).

Among the other commands, we can `show` the metadata of the dataset for describing it, limiting the number of displayed records if required (`-l`/`--limit` ; default is 10 records) and/or sorting the metadata per category (`-c`/`--categories`).

```session
# dataset show test  

 Dataset characteristics
    - Number of executables: 40
    - Categories:            ELF64
    - Packers:               upx, m0dern_p4cker, ezuri

 Executables per size and category
  ──────────  ─────  ─────  ──────  ─────
  Size Range  Total  %      Packed  %
  <20kB       13     32.50  6       30.00
  20-50kB     11     27.50  5       25.00
  50-100kB    7      17.50  1       5.00
  100-500kB   2      5.00   2       10.00
  500kB-1MB   1      2.50   0       0.00
  >1MB        6      15.00  6       30.00
  Total       40            20
  ──────────  ─────  ─────  ──────  ─────

 Executables per label and category
  ────────────────────────────────────────────────────────────────  ──────────────────────────────────────  ────────  ────────────  ─────────────
  Hash                                                              Path                                    Creation  Modification  Label
  5d87ba009d9c8c2aa4c945bb35a5735bb4d655da6f035ed1879240a6602ec866  /sbin/update-passwd                     07/03/21  07/03/21      upx
  160bf8509a43eee893e162a814ac2d6533a11b0c79859c7878c572034d8c6fa1  /usr/bin/look                           07/03/21  07/03/21
  b511fc678e104ae6d097f4e152e27767bb59638053347a497f3cc9352dc73a5c  /sbin/chcpu                             07/03/21  07/03/21      upx
  2c3b26774f1959fd5859f4a0209d5c3cd0344f2c6d0a987f0dc091418d94594b  /usr/bin/X11/ssh-keygen                 07/03/21  07/03/21      ezuri
  2cf63d20084229e01e60047c42827cc34031850e3407a9cd154cd55651514f27  /usr/bin/xz                             07/03/21  07/03/21      upx
  1d8b9dfd71b86f0cde5ebc87b83d1a549092e2a5c7acce1745a4e8a0e5d786bb  /usr/bin/X11/stdbuf                     07/03/21  07/03/21
  5d991b919776bb4bc926fe341492762004a334f301dcd62632bc3d87aa24cbf4  /usr/bin/mv                             07/03/21  07/03/21      m0dern_p4cker
  5298aac043b5b45047cde0c076bbb30ff20b058df11a837c7aba5e904e788971  /sbin/groupmems                         07/03/21  07/03/21
  fb231c12168ce76f749035f80e99d507de8ee00be978e7b87ea468fda3a32c84  /usr/bin/X11/perl5.30-x86_64-linux-gnu  07/03/21  07/03/21
  6bb359e2ce970863a5e2161136879cbbcf763ccf5917c145ef087b3a3234eb4b  /usr/bin/X11/fincore                    07/03/21  07/03/21
  ...                                                               ...                                     ...       ...           ...
  ────────────────────────────────────────────────────────────────  ──────────────────────────────────────  ────────  ────────────  ─────────────

```

While the `show` command displays information about a dataset, it is not aimed to filter and observe records according to criteria. This can be done by using the `view` command which uses [Pandas' query method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html) through the `--query` option.

## Manipulation

The rest of the other commands are aimed to manipulate datasets in different ways. The most trivial commands are:

- `purge`: this trivial command purges the target dataset of all its records and files.
- `rename`: this other trivial command renames the target dataset ; it differs from a simple '`mv [...]`' in that it also adapts backup copies of the dataset.

For adding/removing executables to/from the dataset, a few commands are available:

- `make`: this adds a given number (`-n`/`--number-excutables`) of new executables to the dataset, optionally only from some given categories (`-c`/`--categories`), randomly packing some of them such that the dataset is balanced (using the `-b`/`--balance` option will balance between packer labels, not simply between packed and not packed) and with every applicable packer or only the given ones (`-p`/`--packer`).
- `merge`: this merges the second input dataset into the first one.
- `remove`: this allows to remove executables from the dataset according to criteria relying on [Pandas' query method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html) through the `--query` option.
- `select`: this creates a new dataset with a selected subset of the target dataset, using criteria relying on [Pandas' query method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html) through the `--query` option.
- `update`: this updates the current dataset with executables from the target folders (`-s`/`--source-dir`), either packed or not, labelling them with an input JSON (`-l`/`--labels`) or enabling detection with the integrated detectors (`-d`/`--detect`).

It is also possible to apply corrective/evolutive measures on a dataset:

- `convert`: this makes possible to convert a dataset with files (which offers the possibility to select features multiple times from the same dataset) to a dataset without files (that is, with its features computed once and for all, not keeping the related files and then possibly saving a large amount of disk space).
- `fix`: this allows to fix a corrupted dataset, especially when JSON files do not match the content of the `files` folder anymore, which can be fixed either by relying on the content of this folder (with the `-f`/`--files` option) or on hashes contained in the labels JSON, eventually applying detection (`-d`/`--detect`) to fix labels.
- `revert`: this allows to revert the dataset to its state before the previous operation, with a maximum of 3 

