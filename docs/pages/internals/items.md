# Items

Items are integrated with many [common features coming from a `Base` class](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/__common__.py). This provides for instance the `setup` and `test` methods.

## Initialization

The [`Base` class](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/__common__.py#L127), depending on the subclass inheriting it, will take attributes from the corresponding YAML definition as defined at the [root of the Packing Box](https://github.com/dhondta/docker-packing-box).

It means that, for instance, for a detector called "*Bintropy*" that shall be installed using Pip, its definition can be written in [`detectors.yml`](https://github.com/dhondta/docker-packing-box/blob/main/detectors.yml):

```yaml
Bintropy:
  categories:
    - ELF
    - PE
  command: bintropy {path}
  description: Bintropy is an analysis tool that estimates the likelihood that a binary file contains compressed or encrypted bytes.
  install:
    pip: bintropy
  references:
    - https://ieeexplore.ieee.org/document/4140989
  source: https://github.com/dhondta/docker-packing-box
  status: ok
  vote:   false
```

While being parsed at startup by the [`make_registry` function](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/__common__.py#L101), the `Bintropy` class will be inheriting the [`Detector` class](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/detector.py#L10), itself inheriting the [`Base` class](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/__common__.py#L127) and get its class attributes `categories`, `command`, `description`, `install`, ... set, ready for use by its utility methods.

## `Base` Class

This [class](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/__common__.py#L127) is the base abstraction for items integrated in the box. This holds the common machinery for the [`Detector`](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/detector.py#L10), [`Packer`](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/packer.py#L13) and [`Unpacker`](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/unpacker.py#L12) classes.

This abstraction facilitates the integration of new items as it provides just enough description language in the YAML configuration to do so.

**Attributes**:

- `_bad`: flag for setting the tool as bad (once flagged, it won't run anymore on further inputs during the whole execution)
- `_categories_exp`: expanded list of categories of executable formats (e.g. from `["PE"]`, while expanded, it becomes `[".NET", "PE32", "PE64"]`)
- `_error`: flag for putting the tool in an error state (after an error, the tool will continue running with next inputs)
- `_params`: dictionary for holding on-the-fly generated parameters (e.g. see when [`Packer`'s `run` method](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/packer.py#L46) is used)
- `logger`: `logging.Logger` instance for producing debug messages
- `name`: tool's name (e.g. `bintropy` if defined as *Bintropy* in the YAML file) ; always completely lowercase
- `type`: tool's type, that is, its base class name (e.g. `detector` if the tool inherits from `Detector`) ; lowercase

**Dynamic attributes** (set in the corresponding YAML file):

- `categories` \*: categories of executable formats to be considered
- `command`: the shell command for running the tool (currently only used in the lightweight package [`pboxtools`](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pboxtools/__init__.py#L25))
- `comment`: comment string related to the current tool
- `description`: description of the current tool
- `install` \*: installation steps (see [*Installation*](#installation) for more details)
- `references`: list of hyperlinks to useful documents
- `source`: source where the tool was found
- `status`: status of the tool, amongst the values listed in the [*Status*](#status) section
- `steps`: list of shell commands to be used to come to the desired result from the tool's output (this way, it can for instance include parsing steps)
- `vote` (only for detectors): boolean indicating if the tool shall be used to vote for a packer label

    \* Required (otherwise, an exception will be thrown)

**Methods**:

- `run`: for shaping the command line to run the item on an input executable
- `setup`: for setting up the item according to its install instructions
- `test`: for testing the item on some executable files (embedded with the box)

!!! note "Setup of items with status *broken*"
    
    When an item is marked with the status "*broken*", it won't be installed. Thi means that, if this tool is to be tested afterwards (e.g. if it can finally be integrated), simply changing the status is not sufficient ; it will either require to rebuild the box or, from within the box, run `packing-box setup [detector|packer|unpacker] [item_name]`.


# Status

Not every tool can be easily installed and mechanized in the box. However, for keeping track of what could have been integrated or not, it is useful to keep any tool, working or not, in the YAML configurations. Therefore, we use a *status* to tell if the related tool is usable or not.

The *status* can be set to (*real_value* - *value_to_be_set_in_YAML*):

- `0` - `broken` / `useless`: in this case, the tool is not integrated at all ; it serves for keeping track
- `1` - [*none*]: this status can be obtained when it was not set explicitly in the YAML, but the command with the name of the tool DOES NOT appear in the system's in-`PATH` binaries
- `2` - `gui` / `todo`: temporary status, for flagging the tool as being integrated or stalled for later integration
- `3` - [*none*]: this status can be obtained when it was not set explicitly in the YAML, and the command with the name of the tool DOES appear in the system's in-`PATH` binaries
- `4` - `ok`: this is for tagging the tool as tested and operational

All these statuses are shown with dedicated special colored characters in the help message of the box (see [this screenshot](https://raw.githubusercontent.com/dhondta/docker-packing-box/main/docs/imgs/screenshot.png)).

## Installation

The installation can be mechanized by using various directives in the same style as [Ansible](https://www.ansible.com/). However, it works in a slightly different way as, for the sake of conciness, an install step's state can depend on the previous command's result. The following directives are implemented:

### `apt` / `pip`

- **Argument(s)**: package name
- **Run**: installs the required package through APT/PIP
- **State**: unchanged

### `cd`

- **Argument(s)**: target directory
- **Run**: changes the current directory to the given target
- **State**: target directory

### `copy`

- **Argument(s)**: target | source destination

    - source is absolute or relative to previous *State* if defined or `/tmp`
    - destination is absolute or relative to `/usr/bin` ; if it exists in `/usr/bin`, it is created in `/opt/bin` instead in order to take precedence

- **Run**:

    1. copies a file or a folder (recursively) from the specified *source* to the specified *destination* (equal to *source*, if *destination* is not provided)
    2. if the destination is a file, makes it executable

- **State**: unchanged
- **Examples**:

    - `copy: test` (non-existing file): `cp /tmp/test /usr/bin/test`, `chmod +x /usr/bin/test`
    - `copy: test` (existing file): `cp /tmp/test /opt/bin/test`, `chmod +x /usr/bin/test`
    - `copy: test /opt/test_folder` (folder): `cp -r /tmp/test /opt/test_folder`

### `exec`

- **Argument(s)**: shell command
- **Run**: executes a shell command
- **State**: `None`

### `git` / `gitr`

- **Argument(s)**: URL of the target repository
- **Run**: git-clones the repository, recursively if using `gitr`, to either the current state's directory or, if `None`, `/tmp` joined with the name of the repository
- **State**: target folder of the cloned repository
- **Examples**:

    - `gitr: https://github.com/user/test.git` (no state): `git clone -q --recursive https://github.com/user/test.git /tmp/test`
    - `git: https://github.com/user/test.git` (state is `/opt/packers`): `git clone -q https://github.com/user/test.git /opt/packers/test`

### `ln`

- **Argument(s)**: source

    - absolute
    - relative path of the tool being installed

- **Run**: creates a symlink pointing on either the current state's directory or, if `None`, `/tmp` joined with the argument and targeting `/usr/bin/[item_name_attribute]`
- **State**: `/usr/bin/[item_name_attribute]`
- **Examples**:

    - `ln: test_v1.23` (no state, the tool is defined as "*Test*" in the YAML): `ln -s /tmp/test_v1.23 /usr/bin/test`
    - `ln: /other/path/test_v1.23` (state: `/opt/packers`, same name as before): `ln -s /opt/packers/test_v1.23 /usr/bin/test`

### `lsh`

- **Argument(s)**: tool_name | source tool_name

    - tool's name, for being called from within the shell script
    - source is the folder where the shell script is located

- **Run**:

    1. creates a launcher shell script to `/usr/bin/[item_name_attribute]` that changes its directory to either `/opt/[type]s/[tool_name]` or from the given source, using the binary named as `[tool_name]`
    2. makes the shell script executable

- **State**: `/usr/bin/[item_name_attribute]`
- **Examples**:

    - `lsh: test_v1.23` (the tool is defined as packer named "*Test*" in the YAML): `echo -en "[...]cd /opt/packers/test\n./test_v1.23 $TARGET $2\ncd $PWD" > /usr/bin/test`, `chmod +x /usr/bin/test`
    - `lsh: /other/source test_v1.23` (same as before): `echo -en "[...]cd /other/source\n./test_v1.23 $TARGET $2\ncd $PWD" > /usr/bin/test`, `chmod +x /usr/bin/test`

This is useful when a tool needs to be run from its own folder, e.g. because it depends on relative resources.

### `make`

- **Argument(s)**: new_state make_options
- **Run**: Depending on the build file found in the current *State* folder:

    - `CMakeLists.txt`:
    
        1. run `cmake`
        2. run `make`
    
    - `Makefile`:
    
        1. run `configure.sh` (if exists)
        2. `make`
        3. `make install`
    
    - `make.sh`:
    
        1. `chmod +x make.sh`
        2. `sh -c make.sh`

- **State**: previous state joined with new state

### `move`

- **Argument(s)**: target file
- **Run**: 

    1. moves the target to `/usr/bin[item_name_attribute]` ; if it exists, it is created in `/opt/bin` instead in order to take precedence
    2. if the destination is a file, makes it executable

- **State**: `/usr/bin/[item_name_attribute]`

### `rm`

- **Argument(s)**: file_or_folder
- **Run**: removes the given file or folder
- **State**: unchanged

### `set`

- **Argument(s)**: new_state
- **Run**: sets the current state to the new state
- **State**: new state

### `setp`

- **Argument(s)**: new_state_path
- **Run**: sets the current state as `/tmp` joined to the new state
- **State**: `/tmp` joined to the new state

### `sh` / `wine`

Similar to `lsh`, except that it handles commands in a launcher script that does not change its directory where the target item lies.

- **Argument(s)**: command_(\\n-separated)
- **Run**: 

    1. creates a launcher shell script to `/usr/bin/[item_name_attribute]` (Bash script or Wine launcher), including the given command(s)
    2. makes the shell script executable

- **State**: `/usr/bin/[item_name_attribute]`
- **Examples**:

    - `wine: test_1.exe` (state is `/opt/test_inst`, the tool is named *Test* in the YAML file): `echo -en 'wine /opt/test_inst/test_1.exe \"$@\"' > /usr/bin/test`, `chmod +x /usr/bin/test`
    - `sh: /tmp/test.sh` (same as before): `echo -en '#!/bin/bash\n/tmp/test.sh' > /usr/bin/test`, `chmod +x /usr/bin/test`

### `unrar` / `untar` / `unzip`

- **Argument(s)**: output folder
- **Run**: decompresses an archive at a location based on the previous state (or `/tmp/[tool_name_attribute].[archive_extension]` if not defined) to the given folder, depending on the archive format:
- **State**: deepest single folder from the output folder (e.g. if `/tmp/test` and there s `test_v1.23` decompressed inside, the new state will be `/tmp/test/test_v1.23`)
- **Examples**:

    - `untar: test` (previous state is `/tmp/test_src.tar.gz`): `mkdir -p /tmp/test`, `tar xzf /tmp/test_src.tar.gz -C /tmp/test`
    - `untar: test2` (previous state is `/tmp/test_v1.tar.xz`): `mkdir -p /tmp/test2`, `tar xf /tmp/test_v1.tar.xz -C /tmp/test2`
    - `unzip: /opt/test3` (no state, the tool is named *Test* in the YAML file): `unzip -qqo /tmp/test.zip -d /opt/test3`

### `wget`

- **Argument(s)**: URL
- **Run**: downloads the target URL
- **State**: downloaded file to the `/tmp` folder
- **Examples**:

    - `wget: ĥttps://example.com/test_v1.23.zip`: `wget -q -O /tmp/test_v1.23.zip ĥttps://example.com/test_v1.23.zip`

