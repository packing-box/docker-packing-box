# Datasets

*Datasets* are folders holding samples to be analyzed. It aims to organize input executables according to a [strict structure](#structure), including packing labels, original names, and so forth.

## Structure

A dataset folder holds the following files/folders:

```
[name]
  +-- files
  |     +-- {executables, renamed to their SHA256 hashes}
  +-- data.csv (contains the labels)        # features for an executable, formatted for ML
  +-- features.json                         # dictionary of feature name/description pairs
  +-- labels.json                           # dictionary of hashes with their full labels
  +-- metadata.json                         # simple statistics about the dataset
  +-- names.json                            # dictionary of hashes and their real filenames
```

## Tool

A [dedicated tool](https://github.com/dhondta/docker-packing-box/blob/main/files/tools/dataset) is provided with the [Packing Box](https://github.com/dhondta/docker-packing-box) to manipulate datasets. Its help message tells everything the user needs to get started.

```session
# dataset --help
[...]
This tool aims to manipulate a dataset in multiple ways by:
- adding new executables from the packing-box Docker image or from a user-defined source, packed or not with the
   selected packers installed in the image.
- updating it with already-packed or not-packed executables given their labels or not (detection can be applied).
- fixing its content.
[...]
                        command to be executed
    fix                 fix a corrupted dataset
    list                list all the datasets in the given folder
    make                add n randomly chosen executables from input sources to the dataset
    merge               merge two datasets
    update              update a dataset with new executables
    remove              remove executables from a dataset
    rename              rename a dataset
    reset               reset a dataset
    revert              revert a dataset to its previous state
    show                get an overview of the dataset
```

## Visualization

The list of datasets in a folder can be viewed by using the command `dataset list [folder]` (`dataset list` will display all the datasets in the current folder).

Among the other commands, we can `show` the metadata of the dataset for describing it.

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

## Manipulation

The rest of the other commands are aimed to manipulate datasets in different ways:

- `reset`: this trivial command truncates the target dataset.
- `rename`: this other trivial command renames the target dataset ; it differs from a simple '`mv [...]`' as it also adapts backup copies of the dataset.

