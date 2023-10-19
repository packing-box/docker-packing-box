# Design & Architecture

This section is for developpers who want to learn how [*Packing Box*](https://github.com/dhondta/docker-packing-box) is built. It is a Docker image based on Ubuntu and architectured in three layers as depicted hereafter.

<img src="https://raw.githubusercontent.com/dhondta/docker-packing-box/main/docs/pages/imgs/packing-box-architecture.png" alt="Architecture of Packing Box" width="800" align="center"/>

1. The lower layer is the underlying OS, currently Ubuntu 22.04.
2. The middle layer, _Libraries_, shows an underlying set of executable formats. Currently, the Portable Executable (PE) and Executable & Linkable Format (ELF) are supported and Mach Objects are considered for a future addition. This layer shows two other subsets of software ; _Machine Learning_ currently gathers two frameworks ([Scikit-Learn](https://scikit-learn.org) and [Weka](https://www.cs.waikato.ac.nz/ml/weka)) and _Binary Parsing_ regroups parsing libraries for various executable formats (including [LIEF](https://lief-project.github.io) and some others very soon like [pefile](https://github.com/erocarrera/pefile) and [pyelftools](https://github.com/eliben/pyelftools).
3. The upper layer, _Automation Toolset_, contains the logic for abstracting different [entities](entities/index.html) and [items](items/index.html) that are, for some of them, backed by Python tools and/or configurable through YAML files.

These three layers are tied together through the [`Dockerfile`](https://github.com/packing-box/docker-packing-box/blob/main/Dockerfile) of the [Packing Box](https://github.com/dhondta/docker-packing-box).

The upper layer containing the specific logic is disseminated into the Docker image at different locations:

- Code base (`pbox` Python package): `pbox` holds the source code that defines abstractions (see architecture schema here above) ties every pieces together. It gets installed from [`src/lib`](https://github.com/packing-box/docker-packing-box/tree/main/src/lib) into user's home location for Python libraries at `~/.local/lib/python3.11/site-packages/pbox`.
- Toolset (`~/.opt`): Tools, items and utils coming from [`src/files`](https://github.com/packing-box/docker-packing-box/tree/main/src/files) get installed in user's home at `~/.opt`. The `bin` subfolder is a special case and holds Python scripts aimed to be wrappers for detectors so that they use a logic from another package, `pboxtools` (which is a lightweight library separated from `pbox` to keep startup overhead lower), to normalize detection outputs. 
- Main workspace (`~/.packing-box`): Configuration files (coming from [`src/conf`](https://github.com/packing-box/docker-packing-box/tree/main/src/conf)), executable formats' data (from [`src/data`](https://github.com/packing-box/docker-packing-box/tree/main/src/data)), datasets and models are stored in the main workspace at `~/.packing-box`. This can be changed when working experiments using the `experiment` tool by opening a new experiment. Note that, in this case, figures generated from plot functionalities are saved in a dedicated subfolder of the experiment's workspace.
