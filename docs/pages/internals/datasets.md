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

This [class](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/common/dataset.py#L23) mostly acts as a dictionary for executable entries but also keeps track of some related metadata and, when computed, the related features. When setting a key, it associates the given (real) file and its label and, if required, computes its features available for its executable format.

```session
>>> ds = Dataset()  # this creates a folder named "dataset" if no name is given

>>> ds['/tmp/executable'] = "upx"
<[ this will update ds._data and ds._metadata ]>
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
<[ this will tell the length of ds._data ]>
```

Entries can also be cleaned as with a `dict` instance.

```session
>>> del ds['/tmp/executable']
<[ this will completely remove the item from the dataset ]>
```

**Attributes**:

- `_data`: `pandas.DataFrame` instance holding the data collected from the sample's metadata (hash, creation and modification time, etc), its packing label and, if it is a fileless dataset (that is, for which the features were precomputed), a set of features applicable for the [executable formats](executables.html) selected (loaded from and saved to [`data.csv`](#structure))
- `_features`: dictionary of included features, with short names as keys and their corresponding descriptions as values (loaded from and saved to [`features.json`](#structure))
- `_metadata`: dictionary of metadata, e.g. holding the list of selected executable formats and counts of included samples (loaded from and saved to [`metadata.json`](#structure))
- `formats`: list of applicable categories of executable formats
- `logger`: `logging.Logger` instance for producing debug messages
- `packers`: list of `Packer` instances applicable to the dataset, given the selected categories of executable formats
- `path`: `tinyscript.Path` instance holding the path the the dataset folder
- `sources`: dictionary containing applicable categories as keys and their corresponding lists of source folders for making the dataset

**Properties**:

- `backup` (settable): `Dataset` instance holding the latest backup of the current dataset
- `basename`: the name of the dataset
- `files`: `tinyscript.helpers.path.Path` instance pointing on dataset's `files` subfolder
- `labelling`: the labelling rate of the dataset (`.0` will mean it is only usable with unsupervised learning algorithms while `1.` allows for supervised learning)
- `labels`: the series of labels from `_data`
- `name`: dataset's name, composed with the folder's name and, between brackets, the comma-separated list of applicable executable formats
- `overview`: string representation of the dataset, for describing it in the terminal

**Methods**:

- `exists`: for simply checking if the dataset exists
- `fix` \*: for making dataset's structure and files match
- `is_empty`: for checking if this Dataset instance has no sample
- `is_valid`: for checking if this Dataset instance has a valid structure
- `list`: for listing all the datasets from the given path
- `make` \*: for making N new samples in the current dataset among the input binary categories, balanced or not according to the number of distinct packers
- `merge`: for merging another dataset with the current one ; precedence is set by the `update` parameter
- `purge`: for removing a whole dataset and its backup copies
- `remove(query)` \*: for removing executables from the dataset based on a Pandas Dataframe filtering query
- `rename`: for renaming the current dataset
- `reset`: for truncating and recreating a blank dataset
- `revert`: for reverting to the latest version of the dataset (if a backup copy exists in `/tmp`)
- `select`: for selecting a subset from the current dataset based on multiple criteria
- `show`: for showing an overview of the dataset
- `update` \*: for updating the dataset with a folder of binaries, detecting used packers if `detect=True` otherwise considering samples as not labelled unless labels are provided (in JSON format with hashes as keys and labels as values)

    \* generates a backup copy

**Class methods**:

- `check(folder)`: for checking a `folder` against the required `Dataset` or `FilelessDataset` structures ; returns a boolean
- `validate(folder)`: for checking a `folder` against the required `Dataset` or `FilelessDataset` structures ; raises `ValueError` if no structure is respected

**Static methods**:

- `labels_from_file(path)`: for loading a `labels` dictionary (from a string path or a `Path` instance) ; ensures a valid dictionary is returned
- `summarize(path=None, show=False)`: displays the summary of a dataset (if `path=None`, the local folder is the used), showing corrupted data too if `show=True`

