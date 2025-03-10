# Executables

The `Executable` class allows to abstract an executable. It inherits from the `tinyscript.Path` class, based on `pathlib`'s one.

```session
>>> from pbox import Executable
```

## Use Cases

This class can be used in four different ways:

1. As a classical `Path` instance, not bound to a [`Dataset`](datasets.md) instance

    ```console
    >>> exe = Executable("test.exe")
    >>> exe.filetype
    'PE32 executable (GUI) Intel 80386, for MS Windows'
    ``

2. As a classical `Path` instance with a [`Dataset`](datasets.md) instance specified, to be bound

    ```console
    >>> exe = Executable("test.exe", dataset=Dataset("my-dataset"))
    >>> exe.signature    # data gets retrieved from "my-dataset"
    'PE32 executable (GUI) Intel 80386, for MS Windows, UPX compressed'
    ``
    
    Note that, in this case, the file is required to compute attributes.

3. With exactly one positional argument being the data row with all the attributes to be added to the bound dataset

    ```console
    >>> exe = Executable("test.exe", dataset=Dataset("my-dataset"))
    >>> exe.signature    # data gets retrieved from "my-dataset"
    'PE32 executable (GUI) Intel 80386, for MS Windows, UPX compressed'
    ``
    
    Note that, in this case, the file is not required as attributes come from the input data row.

4. With no positional argument but a [`Dataset`](datasets.md) instance and a *hash* as keyword-arguments ; this will bind the `Executable` instance to the dataset, getting its attributes with data coming from the dataset, and make its path point to the executable with the given *hash* from within the dataset

    ```console
    >>> exe = Executable(hash="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", dataset=Dataset("my-dataset"))
    >>> exe.signature    # data gets retrieved from "my-dataset"
    'PE32 executable (GUI) Intel 80386, for MS Windows, UPX compressed'
    ``

    Note that, in this case, the file is not required as attributes are retrieved from dataset's data.

5. With no positional argument but a source [`Dataset`](datasets.md) instance as *dataset*, a destination [`Dataset`](datasets.md) instance as *dataset2* and a *hash* as keyword-arguments ; this will bind the `Executable` instance to the source dataset and copy its attributes to the destination dataset

    ```console
    >>> exe = Executable(hash="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", dataset=Dataset("my-dataset"), dataset2=Dataset("my-new-dataset"))
    ``


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

This [class](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/core/executable/__init__.py#L22) subclasses [`ts.Path`](https://python-tinyscript.readthedocs.io/en/latest/helpers.html#extended-pathlib-like-classes) (from [Tinyscript](https://python-tinyscript.readthedocs.io/en/latest/)), itself extending [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html) with additional methods.

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

This abstraction facilitates the retrieval of important attributes and the integration of new [features](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/features.yml).

**Attributes**:

- `_dataset`: parent [`Dataset`](datasets.md) instance (if any)
- `label`: packer label (if any)

**Properties**:

- `ctime` \*: creation time as a [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime) instance
- `data` \*: set of features computed based on the `format`
- `destination` \*: destination path for integrating the executable into a dataset (only works if a [`Dataset`](datasets.md) instance is bound)
- `features`: dictionary of features (key: feature name, value: feature description)
- `filetype` \*: file type description (based on [`python-magic`](https://github.com/ahupp/python-magic))
- `format` \*: executable format (e.g. *PE*, *ELF32*, *.NET*)
- `hash` \*: file hash (based on [`hashlib`](https://docs.python.org/3/library/hashlib.html))
- `metadata`: dictionary with properties (see hereafter) `realpath`, `format`, `size`, `ctime` and `mtime`
- `mtime` \*: last modification time as a [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime) instance
- `realpath` \*: real path the executable comes from (only works if a [`Dataset`](datasets.md) instance is bound)
- `size`: size of the executable as an integer

    \* [`cached_property`](https://docs.python.org/3/library/functools.html#functools.cached_property)

**Methods**:

- `alter()`: apply [alterations](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/alterations.yml) to the executable
- `copy()`: copy the file to `self.destination`, that is, to the dataset it is bound to (note that its permissions are restricted to READ for the owner, that is `user`)
- `modify(name)`: apply a modifier by `name` to the executable
- `objdump(n)`: dump `n` disassembled bytes from the executable
- `parse(name)`: parse the binary with a given parser by `name` (by default, the one defined in `~/.packing-box.conf` or, if not defined, the [default from the `pbox` package](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/__conf__.py))
- `plot(...)`: plot the executable's sections with colors and entropy levels
- `show(...)`: show information about the executable

