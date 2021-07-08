# Datasets

The `Dataset` class allows to abstract a folder holding samples to be analyzed.

```session
>>> from pbox import Dataset
```

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

## `Dataset` Class

This [class](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/dataset.py#L28) mostly acts as a dictionary for executable entries. When setting a key, it associates the given (real) file and its label and computes its features as available for its category.

```session
>>> ds = Dataset()  # this creates a folder named "dataset" if no name is given

>>> ds['/tmp/executable'] = "upx"
<[ this will update ds._data, ds._features, ds._labels, ds._metadata, ds._names ]>
```

The dataset entries can then be manipulated in many ways.

```session
>>> print(ds['/tmp/executable'])
<[ this will display the data row for the given item from ds._data AS A LIST ]>

>>> print(ds['/tmp/executable', True])
<[ this will display the data row for the given item from ds._data AS A DICT using the headers of ds._data ]>

>>> for row in ds:
        # do something with a row

>>> len(ds)
<[ this will tell the length of ds._labels but also checks for consistency with the lengths of ds._data and ds._names ]>
```

Entries can also be cleaned as with a `dict` instance.

```session
>>> del ds['/tmp/executable']
<[ this will completely remove the item from the dataset ]>
```

**Attributes**:

- `_data`: `pandas.DataFrame` instance holding the data collected from a set of features applicable for the [executable formats](executables.html) selected (loaded from and saved to [`data.csv`](#structure)))
- `_features`: dictionary of included features, with short names as keys and their corresponding descriptions as values (loaded from and saved to [`features.json`](#structure)))
- `_labels`: dictionary with SHA256-filenames as keys and their corresponding labels of packers (or `None` if unpacked) (loaded from and saved to [`labels.json`](#structure)))
- `_metadata`: dictionary of metadata, e.g. holding the list of selected categories of executable format and counts of included executables (loaded from and saved to [`metadata.json`](#structure)))
- `_names`: dictionary with SHA256-filenames as keys and their corresponding original filenames as values (loaded from and saved to [`names.json`](#structure)))
- `categories`: list of applicable categories of executable formats
- `logger`: `logging.Logger` instance for producing debug messages
- `packers`: list of `Packer` instances applicable to the dataset, given the selected categories of executable formats
- `path`: `tinyscript.Path` instance holding the path the the dataset folder
- `sources`: dictionary containing applicable categories as keys and their corresponding lists of source folders for making the dataset

**Properties**:

- `backup` (settable): `Dataset` instance holding the latest backup of the current dataset
- `files`: `tinyscript.Path` instance pointing on dataset's `files` subfolder
- `name`: dataset's name, composed with the folder's name and, between brackets, the comma-separated list of applicable categories of executable formats
- `overview`: string representation of the dataset, for describing it in the terminal

**Methods**:

- `fix`: for making dataset's structure and files match
- `is_valid`: for checking if this Dataset instance has a valid structure
- `list`: for listing all the datasets from the given path
- `make`: for making N new samples in the current dataset among the input binary categories, balanced or not according to the number of distinct packers
- `merge`: for merging another dataset with the current one ; precedence is set by the `update` parameter
- `remove`: for removing executables from the dataset given their hashes
- `rename`: for renaming the current dataset
- `reset`: for truncating and recreating a blank dataset
- `revert`: for reverting to the latest version of the dataset (if a backup copy exists in `/tmp`)
- `select`: for selecting a subset from the current dataset based on multiple criteria
- `show`: for showing an overview of the dataset
- `update`: for updating the dataset with a folder of binaries, detecting used packers if `detect=True`, otherwise packing randomly ; if labels are provided, they are used instead of applying packer detection

**Static methods**:

- `check(folder)`: for checking a `folder` against the required `Dataset` structure ; returns a boolean
- `labels(labels)`: for loading a `labels` dictionary (from a string or a `Path` instance) ; ensures a valid dictionary is returned
- `summarize(path=None, show=False)`: displays the summary of a dataset (if `path=None`, the local folder is the used), showing corrupted data too if `show=True`
- `validate(folder)`: for checking a `folder` against the required `Dataset` structure ; raises `ValueError` if the structure is not respected

