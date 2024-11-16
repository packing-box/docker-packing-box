# Dataset Manipulation

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

```console
┌──[user@packing-box]──[/mnt/share]────────
$ dataset --help
[...]
This tool aims to manipulate a dataset in many ways including its creation, enlargement, update, alteration, selection,
 merging, export or purge.
[...]
  CMD           command to be executed
  [create/modify/delete]
    alter       alter the target dataset with a set of transformations
    convert     convert the target dataset to a fileless one
    edit        edit the data file
    export      export packed samples from a dataset or export the dataset to a given format
    fix         fix a corrupted dataset
    ingest      ingest samples from a folder into new dataset(s)
    make        add n randomly chosen executables from input sources to the dataset
    merge       merge two datasets
    purge       purge a dataset
    remove      remove executables from a dataset
    rename      rename a dataset
    revert      revert a dataset to its previous state
    select      select a subset of the dataset
    update      update a dataset with new executables
  [read]
    browse      compute features and browse the resulting data
    list        list all the datasets from the workspace
    plot        plot something about the dataset
    preprocess  preprocess the input dataset given preprocessors
    show        get an overview of the dataset
    view        view executables filtered from a dataset
[...]
```

## Visualization

The list of datasets in a folder can be viewed by using the command `dataset list [folder]` (`dataset list` will display all the datasets in the current folder).

Among the other commands, we can `show` the metadata of the dataset for describing it, limiting the number of displayed records if required (`-l`/`--limit` ; default is 10 records) and/or sorting the metadata per category (`-c`/`--categories`).

```console
┌──[user@packing-box]──[/mnt/share]────────
$ dataset show test-upx

Dataset characteristics

 • #Executables: 10                                                                                                                                            
 • Format(s):    PE32                                                                                                                                          
 • Packer(s):    upx                                                                                                                                           
 • Size:         6MB                                                                                                                                           
 • Labelled:     100.00%                                                                                                                                       
 • With files:   yes                                                                                                                                           

Executables per size
                                               
    Size       Not                             
    Range     Packed     %     Packed     %    
 ───────────────────────────────────────────── 
  <20kB       0        0.00    0        0.00   
  20-50kB     0        0.00    0        0.00   
  50-100kB    2        33.33   1        25.00  
  100-500kB   1        16.67   2        50.00  
  500kB-1MB   2        33.33   1        25.00  
  >1MB        1        16.67   0        0.00   
  Total       6        60.00   4        40.00  
                                               
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Sources:                                                                                                                                                       

[0] ~/.wine64/drive_c/windows [1] ~/.wine32/drive_c/windows                                                                                                    

Executables' metadata and labels
                                                                                                                                                               
                             Hash                                                          Path                               Creation   Modification   Label  
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  1d8bf746ba84e321d1afd29c48a6f152e3195cb0c92d5559d75a455d58…   /home/user/.wine32/drive_c/windows/system32/tzres.dll         01/10/23   01/10/23       -      
  21383926f0b5b3909f03f61f1df8b14c7d8f691136e5000c1612016638…   /home/user/.wine32/drive_c/windows/system32/strmdll.dll       01/10/23   01/10/23       -      
  3d2b2fd0d6f3fcb3a0107caa5679ace01f9d9cdb3c1ed37de66a8eb496…   /home/user/.wine32/drive_c/windows/system32/d3dx9_33.dll      01/10/23   01/10/23       -      
  694b7a2075a9ec584346af7db65aacf55afce314e70753d0c88b5bb59d…   /home/user/.wine32/drive_c/windows/system32/itircl.dll        01/10/23   01/10/23       -      
  695f311494155ba4890a2543d4398f97782f9a542b9ebf0978ff356338…   /home/user/.wine32/drive_c/windows/system32/concrt140.dll     01/10/23   01/10/23       -      
  afa6815e81bd1da93d05465b21698da5d8decd177ac39d8fa2f724bbe4…   /home/user/.wine32/drive_c/windows/system32/dispex.dll        01/10/23   01/10/23       upx    
  b73e7f86951d8f8a4b881cb69f6dbda28950b6015c558949f7a2d97781…   /home/user/.wine32/drive_c/windows/system32/start.exe         01/10/23   01/10/23       upx    
  c6e875d0be29bfefc9a5a517e108b395f1404cdb9e80cb5c1c3604f457…   /home/user/.wine32/drive_c/windows/system32/d3dcompiler_41…   01/10/23   01/10/23       -      
  e88f4aedd45410f2a44b94ff928529d9760bd1d35c09a818aa30395790…   /home/user/.wine32/drive_c/windows/system32/traffic.dll       01/10/23   01/10/23       upx    
  ecd981b59b54c3e701079e478e908dd6b68f6ee8e8f7d319ba698c10a1…   /home/user/.wine32/drive_c/windows/system32/msvcm80.dll       01/10/23   01/10/23       upx    

```

While the `show` command displays information about a dataset, it is not aimed to filter and observe records according to criteria. This can be done by using the `view` command which uses [Pandas' query method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html) through the `--query` option.

The following commands are aimed to parse a dataset:

- `browse`: this opens dataset's content with [Visidata](https://www.visidata.org), showing data including computed features ; checkout the [`features.yml` configuration file](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/features.yml) configuration file for examples of feature definitions.
- `export`: this allows either to export packed samples only or to export the dataset to another file format (see [`dsff` package](https://github.com/packing-box/python-dsff) for the available formats).
- `list`: this lists all the existing datasets, showing minimal information including the number of executables, the size, whether files are contained, in-scope executable formats and the numbers of packed samples per packer used.
- `plot`: this allows to make plots representing characteristics of the dataset (e.g. the distribution of labels or information gains of the computed features).
- `preprocess`: this preprocesses features data using an input preprocessor and then opens the result with [Visidata](https://www.visidata.org), allowing to inspect preprocessed data that is fed to learning models.
- `show`: this displays dataset information (as illustrated here above).
- `view`: this allows to view samples in the terminal filter out with the input filter if any.

## Manipulation

The rest of the other commands are aimed to manipulate datasets in different ways. The most trivial commands are:

- `purge`: this trivial command purges the target dataset of all its records and files.
- `rename`: this other trivial command renames the target dataset ; it differs from a simple '`mv [...]`' in that it also adapts backup copies of the dataset.

For adding/removing executables to/from the dataset, a few commands are available:

- `ingest`: this allows to ingest samples from an external folder, specifying an input JSON of labels (`-l`/`--labels`) or enabling [(super)detection](detectors.html#superdetection) (`-d`/`--detect` -- that is, using a combination of detectors with voting for determining the final label).
- `make`: this adds a given number (`-n`/`--number`) of new executables to the dataset, optionally only from some given executable formats (`-f`/`--formats`), randomly packing some of them such that the dataset is balanced (using the `-b`/`--balance` option will balance between packer labels, not simply between packed and not packed) and with every applicable packer or only the given ones (`-p`/`--packer`).
- `merge`: this merges the second input dataset into the first one.
- `remove`: this allows to remove executables from the dataset according to criteria relying on [Pandas' query method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html) through the `--query` option.
- `select`: this creates a new dataset with a selected subset of the target dataset, using criteria relying on [Pandas' query method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html) through the `--query` option.
- `update`: this updates the current dataset with executables from the target folders (`-s`/`--source-dir`), either packed or not, labelling them with an input JSON (`-l`/`--labels`) or enabling [(super)detection](detectors.html#detection) (`-d`/`--detect`).

It is also possible to apply corrective/evolutive measures on a dataset:

- `alter`: this will apply the selected alterations in order to modify samples from the target dataset ; checkout the [`alterations.yml` configuration file](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/alterations.yml) for examples of alteration definitions.
- `convert`: this makes possible to convert a dataset with files (which offers the possibility to select features multiple times from the same dataset) to a dataset without files (that is, with its features computed once and for all, not keeping the related files and then possibly saving a large amount of disk space) ; checkout the [`features.yml` configuration file](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/features.yml) configuration file for examples of feature definitions.
- `edit`: this allows to edit dataset's data contained in its `data.csv` with [Visidata](https://www.visidata.org).
- `fix`: this allows to fix a corrupted dataset, especially when JSON files do not match the content of the `files` folder anymore, which can be fixed either by relying on the content of this folder (with the `-f`/`--files` option) or on hashes contained in the labels JSON, eventually applying detection (`-d`/`--detect`) to fix labels.
- `revert`: this allows to revert the dataset to its state before the previous operation, with a maximum of 3 

