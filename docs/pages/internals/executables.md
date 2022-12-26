# Executables

The `Executable` class allows to abstract an executable. It inherits from the `tinyscript.Path` class, based on `pathlib`'s one.

```session
>>> from pbox import Executable
```

## Use Cases

This class can be used in four different ways:

1. As a classical `Path` instance
2. As a classical `Path` instance with a [`Dataset`](datasets.html) instance bound
3. With no positional arguments for describing a path but a [`Dataset`](datasets.html) instance and a *hash* as keyword-arguments ; this will bind the `Executable` instance to the dataset and make the path point to the executable with the given *hash* from within the dataset
4. From an `Executable` instance as positional argument with a [`Dataset`](datasets.html) instance as keyword-argument ; in this case, the new `Executable` will have the properties of the input one and the file will be copied to the bound dataset

## Supported Formats

This abstraction handles multiple executable formats sorted in categories:

```
All
  +-- ELF
  |     +-- ELF32           ^(set[gu]id )?ELF 32-bit
  |     +-- ELF64           ^(set[gu]id )?ELF 64-bit
  +-- Mach-O
  |     +-- Mach-O32        ^Mach-O 32-bit
  |     +-- Mach-O64        ^Mach-O 64-bit
  |     +-- Mach-Ou         ^Mach-O universal binary
  +-- MSDOS                 ^MS-DOS executable\s*
  +-- PE
        +-- .NET            ^PE32\+? executable (.+?)\.Net assembly
        +-- PE32            ^PE32 executable
        +-- PE64            ^PE32\+ executable
```

Each processing depending on categories flattens its list from this tree structure ; e.g. `["PE", "ELF64"]` will be expanded to `[".NET", "PE32", "PE64", "ELF64"]`

## `Executable` Class

This [class](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/executable.py#L28) subclasses [`ts.Path`](https://python-tinyscript.readthedocs.io/en/latest/helpers.html#extended-pathlib-like-classes) (from [Tinyscript](https://python-tinyscript.readthedocs.io/en/latest/)), itself extending [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html) with additional methods.

```session
>>> exe = Executable("hello-world.exe")

>>> exe.category
'.NET'

>>> exe.ctime
datetime.datetime(2021, 7, 8, 7, 41, 4, 875819)

>>> exe.hash
'889ce94c1f7f909c045247adf1f883928e7760cb9e49f2340a233a361f690d28'

>>> exe.data
{'dll_characteristics_1': 0, 'dll_characteristics_2': 0, 'dll_characteristics_3': 0, [...]
```

This abstraction facilitates the retrieval of important attributes and the integration of new [features](https://github.com/dhondta/docker-packing-box/tree/main/files/lib/pbox/learning/features).

**Attributes**:

- `_dataset`: parent [`Dataset`](datasets.html) instance (if any)
- `label`: packer label (if any)

**Properties**:

- `ctime` \*: creation time as a [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime) instance
- `data` \*: set of features computed based on the `format`
- `destination` \*: destination path for integrating the executable into a dataset (only works if a [`Dataset`](datasets.html) instance is bound)
- `features`: dictionary of features (key: feature name, value: feature description)
- `filetype` \*: file type description (based on [`python-magic`](https://github.com/ahupp/python-magic))
- `format` \*: executable format (e.g. *PE*, *ELF32*, *.NET*)
- `hash` \*: file hash (based on [`hashlib`](https://docs.python.org/3/library/hashlib.html))
- `metadata`: dictionary with properties (see hereafter) `realpath`, `format`, `size`, `ctime` and `mtime`
- `mtime` \*: last modification time as a [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime) instance
- `realpath` \*: real path the executable comes from (only works if a [`Dataset`](datasets.html) instance is bound)
- `size`: size of the executable as an integer

    \* [`cached_property`](https://docs.python.org/3/library/functools.html#functools.cached_property)

**Methods**:

- `copy()`: copy the file to `self.destination`, that is, to the dataset it is bound to (note that its permissions are restricted to READ for the owner, that is `user`)
- `update()`: triggers the removal of the cached properties `filetype`, `format` and `hash` for further recomputation

