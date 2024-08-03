# Contributing to Packing-Box
We want to make contributing to this project as easy and transparent as possible.

## Our Development Process

We use GitHub to sync code to and from our internal repository. We'll use GitHub to track issues and feature requests, as well as accept pull requests.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you want to **tune container's bash console**, you can add/edit files in this folder: `files/term`. Please test what you tune and be sure not to break other related features.
3. If you want to **add a packer**, please refer to the related subsection hereafter.
4. If you want to **add a tool**, please refer to the related subsection hereafter.

  Please beware that:
  - Already-built packers respectively go to: `files/packers` (project-side) and `/opt/packers/.bin` (within the container).
  - Tools respectively go to the following project and container folders: `files/tools` and `/opt/tools`.

Before submitting your pull requests, please follow the steps below to explain your contribution.

1. Copy the correct template for your contribution
  - 💻 Are you improving the console ? Copy the template from <PR_TEMPLATE_CONSOLE.md>
  - 📦 Are you adding a new packer ? Copy the template from <PR_TEMPLATE_PACKER.md>
  - 🛠️ Are you adding a new tool ? Copy the template from <PR_TEMPLATE_TOOL.md>
2. Replace this text with the contents of the template
3. Fill in all sections of the template
4. Click "Create pull request"

## Add a Packer

Packers are defined in [`packers.yml`](https://github.com/dhondta/docker-packing-box/blob/main/packers.yml). This master file holds the metadata and install instructions of the various packers BUT NOT their specific run method. The idea is to put all the Linux packers in `/usr/bin` and all the Windows packers in `/opt/packers` and get a Shell launcher script created to start `wine` or `mono` from `/usr/bin`.

When adding a packer, consider the following:
- Packer itself:
  - Is it available for download ? Then use the `wget` instruction with the target URL. See [this example](https://github.com/dhondta/docker-packing-box/blob/main/packers.yml#L5).
  - Is it provided with the box ? Then use the `copy` instruction with either an absolute source path or a path relative to `/tmp` OR the last path resulting from a previous instruction (e.g. using `unzip`). This will copy the related file to `/usr/bin` and ensure it has the executable flag. See [this example](https://github.com/dhondta/docker-packing-box/blob/main/packers.yml#L23).
  - Is it compressed in a ZIP archive ? Then use the `unzip` instruction with the *target folder* as its single argument ;
    - Does the archive contain the packer only (nothing else) ? If yes, set the *target folder* folder as `/opt/packer`. See [this example](https://github.com/dhondta/docker-packing-box/blob/main/packers.yml#L31).
    - Does the archive contain a root level with the packer built and other files inside ? If yes, set the *target folder* folder as `/opt/packer` ; if not, use `/opt/packer/[packer_lowercase_name]`. See [this example (first case)](https://github.com/dhondta/docker-packing-box/blob/main/packers.yml#L15) and [this example (second case)](https://github.com/dhondta/docker-packing-box/blob/main/packers.yml#L52).
    - Does the packer need to be built ? Then use the `unzip` instruction with the *target folder* in `/tmp` and use the `make` instruction (use the `cd` instruction if you need to change the directory before running `make`). See [this example](https://github.com/dhondta/docker-packing-box/blob/main/packers.yml#L43).
  - Is packer installable with APT ? Then use the `apt` instruction. See [this example](https://github.com/dhondta/docker-packing-box/blob/main/packers.yml#L95).
  - Is packer's binary a Windows executable ? Then use the `sh` instruction to make a launcher Shell script, specifying `wine` or `mono` as the launcher and the location to packer's binary. See [this example](https://github.com/dhondta/docker-packing-box/blob/main/packers.yml#L16).

By default, packer classes in Python tools are dynamically composed with a run method that simply execute "`[packer_name] [executable_to_pack]`". If you what to add a packer with a specific run machinery, you need to edit [`packers.py`](https://github.com/dhondta/docker-packing-box/blob/main/files/tools/packers.py) (see [example](https://github.com/dhondta/docker-packing-box/blob/main/files/tools/packers.py#L203)).

**Important note**: Everything from `files/packers/` gets copied to `/tmp` and is available using install instructions in [`packers.yml`](https://github.com/dhondta/docker-packing-box/blob/main/packers.yml). **No need to edit the** [**Dockerfile**](https://github.com/dhondta/docker-packing-box/blob/main/Dockerfile).

## Add a Tool

Tools are defined in [`files/tools`](https://github.com/dhondta/docker-packing-box/tree/main/files/tools). A module called `packers.py` implements an abstraction for packers, setting its key-values from [`packers.yml`](https://github.com/dhondta/docker-packing-box/blob/main/packers.yml) as their class attributes. Tools can be made [Tinyscript](https://github.com/dhondta/python-tinyscript) (see [this example](https://github.com/dhondta/docker-packing-box/blob/main/files/tools/packer-installer)) or with whatever you want. If you want to get your new tool referenced with the `help` tool, you need to add the `__description__` dunder.

**Important note**: Everything from `files/tools/` gets copied to `/opt/tools`. **No need to edit the** [**Dockerfile**](https://github.com/dhondta/docker-packing-box/blob/main/Dockerfile).

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.
