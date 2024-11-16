# Items

Items are integrated with many [common features coming from a `Base` class](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/core/items/__common__.py). This provides for instance the `setup` and `test` methods.

## Initialization

The [`Base` class](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/core/items/__common__.py#L88), depending on the subclass inheriting it, will take attributes from the corresponding YAML definition as defined at the [root of the Packing Box](https://github.com/dhondta/docker-packing-box).

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

The `Bintropy` class will be inheriting the [`Detector` class](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/core/items/detector.py#L56), itself inheriting the [`Base` class](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/core/items/__common__.py#L88) and get its class attributes `categories`, `command`, `description`, `install`, ... set, ready for use by its utility methods.

## `Base` Class

This [class](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/core/items/__common__.py#L88) is the base abstraction for items integrated in the box. This holds the common machinery for the [`Detector`](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/core/items/detector.py#L56), [`Packer`](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/core/items/packer.py#L30) and [`Unpacker`](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/core/items/unpacker.py#L17) classes.

This abstraction facilitates the integration of new items as it provides just enough description language in the YAML configuration to do so.

**Attributes**:

- `_bad`: flag for setting the tool as bad (once flagged, it won't run anymore on further inputs during the whole execution)
- `_categories_exp`: expanded list of categories of executable formats (e.g. from `["PE"]`, while expanded, it becomes `[".NET", "PE32", "PE64"]`)
- `_error`: flag for putting the tool in an error state (after an error, the tool will continue running with next inputs)
- `_params`: dictionary for holding on-the-fly generated parameters (e.g. see when [`Packer`'s `run` method](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/core/items/packer.py#L95) is used)
- `logger`: `logging.Logger` instance for producing debug messages
- `name`: tool's name (e.g. `bintropy` if defined as *Bintropy* in the YAML file) ; always completely lowercase
- `type`: tool's type, that is, its base class name (e.g. `detector` if the tool inherits from `Detector`) ; lowercase

**Dynamic attributes** (set in the corresponding YAML file):

- `categories` \*: categories of executable formats to be considered
- `command`: the shell command for running the tool (currently only used in the lightweight package [`pboxtools`](https://github.com/packing-box/docker-packing-box/tree/main/src/lib/src/pboxtools))
- `comment`: comment string related to the current tool
- `description`: description of the current tool
- `install` \*: installation steps (see [*Installation*](#installation) for more details)
- `references`: list of hyperlinks to useful documents
- `source`: source where the tool was found
- `status`: status of the tool, amongst the values listed in the [*Status*](#status) section
- `steps`: list of shell commands to be used to come to the desired result from the tool's output (this way, it can for instance include parsing steps)

    \* Required (otherwise, an exception will be thrown)

**Methods**:

- `check(*formats)`: check if the item applies to the given formats
- `help`: display the item's help message
- `run`: for shaping the command line to run the item on an input executable
- `setup`: for setting up the item according to its install instructions
- `test`: for testing the item on some executable files (embedded with the box)

!!! note "Setup of items with status *broken*"
    
    When an item is marked with the status "*broken*", it won't be installed. This means that, if this tool is to be tested afterwards (e.g. if it can finally be integrated), simply changing the status is not sufficient ; it will either require to rebuild the box or, from within the box, run `packing-box setup [detector|packer|unpacker] [item_name]`.


# Status

Not every tool can be easily installed and mechanized in the box. However, for keeping track of what could have been integrated or not, it is useful to keep any tool, working or not, in the YAML configurations. Therefore, we use a *status* to tell if the related tool is usable or not.

The *status* can be set to (*real_value* - *value_to_be_set_in_YAML*):

- `0` - `broken` / `useless`: in this case, the tool is not integrated at all ; it serves for keeping track
- `1` - [*none*]: this status can be obtained when it was not set explicitly in the YAML, but the command with the name of the tool DOES NOT appear in the system's in-`PATH` binaries
- `2` - `gui` / `todo`: temporary status, for flagging the tool as being integrated or stalled for later integration
- `3` - [*none*]: this status can be obtained when it was not set explicitly in the YAML, and the command with the name of the tool DOES appear in the system's in-`PATH` binaries
- `4` - `ok`: this is for tagging the tool as tested and operational

All these statuses are shown with dedicated special colored characters in the help message of the box (see [this screenshot](https://raw.githubusercontent.com/dhondta/docker-packing-box/main/docs/pages/imgs/screenshot.png)).

## Installation

The installation can be mechanized by using various directives in the same style as [Ansible](https://www.ansible.com/). However, it works in a slightly different way as, for the sake of conciness, an install step's state can depend on the previous command's result.

In the directives hereafter, some "environment variables" (NOT real environment variables as they are only applicable in the scope of the installation tool) can be used in order to shorten arguments:

- *$OPT*: `~/.opt/[ITEM]s` (e.g. `/tmp/packers`)
- *$TMP*: `/tmp/[ITEM]s` (e.g. `/tmp/detectors`)


The following directives are implemented:

### `apt` / `pip`

> Simple install through APT or PIP.

- **Argument(s)**: package name
- **State**: unchanged

### `cd`

> Change to the given directory.

If a relative path is given, it is first joined to the path from the last state if available or to the temporary path (meaning, in this scope, `/tmp/[ITEM]s`).

- **Argument(s)**: target absolute/relative path
- **State**: target directory

### `chmod`

> Add the executable flag on the target.

This basically runs the system command `chmod +x [...]`.

- **Argument(s)**: target absolute/relative path
- **State**: unchanged

### `copy`

> Copy the given source file/directory to the given destination file/directory.

If the source is a relative path, it is first joined to the path from the last state if available or to the temporary path (meaning, in this scope, `/tmp/[ITEM]s`). If the destination is a relative path, it is joined to `~/.local/bin`. If it already exists, it is then joined to `~/.opt/bin`. If the destination is a file, make it executable.

- **Argument(s)**: source absolute/relative path, destination absolute/relative path
- **State**: unchanged
- **Examples**:

    - `copy: test` (non-existing packer): `cp /tmp/packers/test ~/.local/bin/test`, `chmod +x ~/.local/bin/test`
    - `copy: test` (existing file): `cp /tmp/test /opt/bin/test`, `chmod +x ~/.local/bin/test`
    - `copy: test /opt/test_folder` (folder): `cp -r /tmp/test ~/.opt/packers/test_folder`

### `exec`

> Execute the given shell command or the given list of shell commands.

- **Argument(s)**: shell command(s)
- **State**: `None` (resets the state)

### `git` / `gitr`

> Git clone a project (recursively if `gitr`).

The target folder is computed relatively to the last state if any, otherwise to the temporary path (meaning, in this scope, `/tmp/[ITEM]s`).

- **Argument(s)**: URL of the target repository
- **State**: target folder of the cloned repository
- **Examples**:

    - `gitr: https://github.com/user/test.git` (no state): `git clone -q --recursive https://github.com/user/test.git /tmp/packers/test`
    - `git: https://github.com/user/test.git` (state is `/opt/packers`): `git clone -q https://github.com/user/test.git ~/.opt/packers/test`

### `go`

> Build a Go project, moving to the source directory and then coming back to the initial directory.

If no source directory is given, take the directory from the current state.

- **Argument(s)**: (source directory, ) URL of the target repository (without `https://`)
- **State**: target folder of the Go project

### `java` / `mono` / `sh` / `wine`

> Create a shell script to execute the given target with its intepreter/launcher and make it executable.

The launcher script is created at `~/.local/bin/[NAME]`. If relative path is given for the target, it is set relatively to the previous state if any, otherwise to `~/.opt/[ITEM]s`.

- **Argument(s)**: command (\\n-separated)
- **State**: `~/.local/bin/[NAME]`
- **Examples**:

    - `wine: test_1.exe` (state is `~/.opt/analyzers/test_inst`, the tool is named *Test* in the YAML file): `echo -en 'wine ~/.opt/analyzers/test_inst/test_1.exe \"$@\"' > ~/.local/bin/test`, `chmod +x ~/.local/bin/test`
    - `sh: /tmp/test.sh` (same as before): `echo -en '#!/bin/bash\n/tmp/test.sh' > ~/.local/bin/test`, `chmod +x ~/.local/bin/test`

### `ln`

> Create a symbolic link at `~/.local/bin/[NAME]` pointing to the given source.

This is set relatively to the previous state if any, otherwise to `~/.opt/[ITEM]s`.

- **Argument(s)**: source
- **State**: `~/.local/bin/[NAME]`
- **Examples**:

    - `ln: test_v1.23` (no state, the tool is defined as "*Test*" in the YAML): `ln -s /tmp/test_v1.23 /usr/bin/test`
    - `ln: /other/path/test_v1.23` (state: `/opt/packers`, same name as before): `ln -s /opt/packers/test_v1.23 /usr/bin/test`

### `lsh` / `lwine`

> Create a shell script to execute the given target from its source directory with its intepreter/launcher and make it executable.

The launcher script is created at `~/.local/bin/[NAME]`. When executed, the script changes the directory to the source one, executes the target and then comes back to the original directory. If the source directory is not speficied, it is set to `~/.opt/[ITEM]s/[NAME]`.

- **Argument(s)**: (source directory, ) target script/executable
- **State**: `~/.local/bin/[NAME]`
- **Examples**:

    - `lsh: test_v1.23` (the tool is defined as packer named "*Test*" in the YAML): `echo -en "[...]cd /opt/packers/test\n./test_v1.23 $TARGET $2\ncd $PWD" > ~/.local/bin/[NAME]/bin/test`, `chmod +x ~/.local/bin/[NAME]/bin/test`
    - `lsh: /other/source test_v1.23` (same as before): `echo -en "[...]cd /other/source\n./test_v1.23 $TARGET $2\ncd $PWD" > ~/.local/bin/[NAME]/bin/test`, `chmod +x ~/.local/bin/[NAME]/bin/test`

This is useful when a tool needs to be run from its own folder, e.g. because it depends on relative resources.

### `make`

> Compile a project.

Compilation is done based on a build file found in the directory from the current state. The result is the build target joined to the current state. If build options are set as the second argument, they are appended to the `make` command.

3 different formats of build files are supported:

- `CMakeLists.txt`: `cmake` is run first then `make`.
- `Makefile`: `configure.sh` is run first if it exists then `make` then `make install`.
- `make.sh`: the script's flags are set to executable and then run with `sh -c make.sh`.

- **Argument(s)**: build target (, build options)
- **State**: previous state joined with the build target

### `md`

> Rename the current working directory and change to the new one.

The destination folder is computed relatively to the last state if any, otherwise to the temporary path (meaning, in this scope, `/tmp/[ITEM]s`).

- **Argument(s)**: destination folder
- **State**: destination folder

### `rm`

> Remove the target location.

By default, after every installation, `/tmp/[ITEM]s/[NAME]` gets removed. When this directive is used, it overrides this default removal.

- **Argument(s)**: target file/folder
- **State**: unchanged

### `set` / `setp`

> Manually set the result to be used in the next command. 

If `setp` is used, the input argument is joined with `/tmp/[ITEM]s`, therefore giving a `p`ath object.

- **Argument(s)**: new state
- **State**: new state

### `un7z` / `unrar` / `untar` / `unzip`

> Decompress a RAR/TAR/ZIP archive to the given location.

The path to the archive is taken from the state, i.e. when it has been downloaded with a previous command. The destination folder is computed relatively to `/tmp/[ITEM]s`. If the verbose mode is enabled, the archive is also decompressed to a temporary path (i.e. `/tmp/[ITEM]-setup-[RANDOM_8_CHARS]`) for debugging purpose. If the command that downloaded the archive was `wget`, the archive is removed after decompression.

- **Argument(s)**: output folder
- **State**: deepest single folder from the output folder (e.g. if `/tmp/test` and there s `test_v1.23` decompressed inside, the new state will be `/tmp/test/test_v1.23`)
- **Examples**:

    - `untar: test` (previous state is `/tmp/test_src.tar.gz`): `mkdir -p /tmp/test`, `tar xzf /tmp/test_src.tar.gz -C /tmp/test`
    - `untar: test2` (previous state is `/tmp/test_v1.tar.xz`): `mkdir -p /tmp/test2`, `tar xf /tmp/test_v1.tar.xz -C /tmp/test2`
    - `unzip: /opt/test3` (no state, the tool is named *Test* in the YAML file): `unzip -qqo /tmp/test.zip -d /opt/test3`

### `wget`

> Download a resource.

It can possibly download 2-stage generated download links (in this case, the list is handled by downloading the URL from the first element then matching the second element in the URL's found in the downloaded Web page). If the target URL points to GitHub and includes a descriptor for the target release, GitHub's API is used to identify the available releases and the descriptor selects the right one.

Release descriptors:

- `https://github.com/username/repo:TAG{pattern}`: the selected release is the first match for the given `pattern`
- `https://github.com/username/repo:TAG[X]`: the selected release is the `X`th from the list of releases
- `https://github.com/username/repo:TAG`: the selected release is the first one from the list of releases

- **Argument(s)**: URL
- **State**: downloaded file at `/tmp/[ITEM]s/[NAME].[EXT]`
- **Examples**:

    - `wget: ĥttps://example.com/test_v1.23.zip`: `wget -q -O /tmp/test_v1.23.zip ĥttps://example.com/test_v1.23.zip`

