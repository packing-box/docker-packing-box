<p align="center" id="top"><img src="https://github.com/packing-box/docker-packing-box/raw/main/docs/pages/imgs/logo.png"></p>
<h1 align="center">Packing Box <a href="https://twitter.com/intent/tweet?text=Packing%20Box%20-%20Docker%20container%20featuring%20a%20CLI%20environment%20with%20packers%20and%20detectors%20for%20studying%20executable%20packing%2c%20including%20machine%20learning%20dataset%20generation%20and%20pipeline%20execution%2e%0ahttps%3a%2f%2fgithub%2ecom%2fpacking-box%2fdocker-packing-box&hashtags=docker,container,python,infosec,cybersecurity,malware"><img src="https://img.shields.io/badge/Tweet--lightgrey?logo=twitter&style=social" alt="Tweet" height="20"/></a></h1>
<h3 align="center">Experimental toolkit for static detection of executable packing.</h3>

[![Read The Docs](https://readthedocs.org/projects/docker-packing-box/badge/?version=latest)](http://docker-packing-box.readthedocs.io/en/latest/?badge=latest)
[![Black Hat Arsenal Europe 2022](https://badgen.net/badge/Black%20Hat%20Arsenal/Europe%202022/blue)](https://www.blackhat.com/eu-22/arsenal/schedule/index.html#packing-box-playing-with-executable-packing-29054)
[![Black Hat Arsenal Europe 2023](https://badgen.net/badge/Black%20Hat%20Arsenal/Europe%202023/blue)](https://www.blackhat.com/eu-23/arsenal/schedule/#packing-box-breaking-detectors--visualizing-packing-35678)
[![Black Hat Arsenal Europe 2024](https://badgen.net/badge/Black%20Hat%20Arsenal/Europe%202024/blue)](https://www.blackhat.com/eu-24/arsenal/schedule/#packing-box-improving-detection-of-executable-packing-41931)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)

This Docker container is a CLI environment featuring a toolkit that gathers executable analyzers, packing detectors, packers and unpackers but also many tools for generating and manipulating datasets of packed and not-packed executables of different formats (including PE, ELF and Mach-O) for the sake of evaluating static detection techniques and tools, visualizing executables' layout and automating machine learning pipelines with the support of many algorithms.

See the Black Hat Arsenal presentations for demonstrations:

- [Packing-Box: Playing with Executable Packing](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/material/bheu22-packingbox.pdf)
- [Packing-Box: Breaking Detectors & Visualizing Packing](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/material/bheu23-packingbox.pdf)
- [Packing-Box: Improving Detection of Executable Packing](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/material/bheu24-packingbox.pdf)

Here is what you can see when you start up the Docker container.

<p align="center"><img src="https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/imgs/screenshot.png"></p>

The various items integrated in the Packing-Box are defined in the very declarative and easy-to-use YAML format through different [configuration files](https://github.com/packing-box/docker-packing-box/tree/main/src/conf). This makes shaping the scope for evaluations and machine learning model training straightforward and practical for researchers.

## :fast_forward: Quick Start

Building the image:

```console
# docker build -t dhondta/packing-box .
[...]
<<<wait for a while>>>
[...]
```

Starting it up with the current working directory mounted as `/mnt/share` in the container:

**Windows**

```powershell
PS C:\> docker run -it -h packing-box -v ${pwd}:/mnt/share dhondta/packing-box
```

**Linux**

```bash
# docker run -it -h packing-box -v `pwd`:/mnt/share dhondta/packing-box
```

## :clipboard: Basics

### Items Usage

*Items* are configured through the [YAML configuration files](https://github.com/packing-box/docker-packing-box/tree/main/src/conf). They consist in:
- [`analyzers.yml`](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/analyzers.yml): utilities for analyzing files or more specifically packer trace
- [`detectors.yml`](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/detectors.yml): tools for analyzing and deciding whether an executable is packed or not
- [`packers.yml`](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/packers.yml) and [`unpackers.yml`](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/unpackers.yml): self-explanatory

From within the Packing-Box, the [`packing-box`](https://github.com/packing-box/docker-packing-box/blob/main/src/files/tools/packing-box) tool allows to setup and test items.

**Operation** | **Description** | **Command**
:---:| --- | ---
**`setup`** | Setup an item from its YAML `install` definition | `# packing-box setup detector die`
**`test`** | Test an item using a built-in set of test samples | `# packing-box test packer upx`

Afterwards, items are available from the console.

```console
$ die --help
<<snipped>>
$ upx --help
<<snipped>>
```

### Mass Packing & Detection

Packers and detectors have their respective dedicated tools for mass operations, [`packer`](https://github.com/packing-box/docker-packing-box/blob/main/src/files/tools/packer) and [`detector`](https://github.com/packing-box/docker-packing-box/blob/main/src/files/tools/detector). They work either on a single file, a complete folder or a special dataset instance (as of the abstraction defined in the [`pbox`](https://github.com/packing-box/docker-packing-box/tree/main/src/lib/pbox) package).

```console
$ packer upx path/to/executables --prefix "upx_"
<<snipped>>
```

For the [`detector`](https://github.com/packing-box/docker-packing-box/blob/main/src/files/tools/detector) tool, not selecting any detector will use those selected in [`detectors.yml`](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/detectors.yml) as being part of the "*superdetector*". Moreover, the `--binary` option will consider whether the target executable is packed or not and not is precise packer.

```console
$ detector path/to/single-executable -d die -d pypackerdetect
<<snipped>>
$ detector path/to/executables
<<snipped ; will use "superdetection">>
$ detector path/to/executables -d bintropy --binary
<<snipped ; in this case, as Bintropy only supports binary classification, --binary is necessary>>
```

### Learning Pipeline

Machine Learning models are fine-tuned through the [YAML configuration files](https://github.com/packing-box/docker-packing-box/tree/main/src/conf). They consist in:
- [`algorithms.yml`](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/algorithms.yml): the algorithms that are used with their static or dynamic parameters while training models
- [`features.yml`](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/features.yml): the characteristics to be considered while training and using models

<p align="center"><img src="https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/imgs/machine-learning-pipeline.png"></p>

The *PREPARE* phase, especially *feature engineering*, is fine-tuned with the *features* YAML definition. Note that feature extraction is achieved with the [`pbox`](https://github.com/packing-box/docker-packing-box/tree/main/src/lib/pbox) package of the Packing-Box while feature derivation and transformation is fine-tuned via the *features* YAML file.

The *TRAIN* phase is fine-tuned through the *algorithms* YAML file by setting the static and/or cross-validation parameters.

### Dataset Manipulations

The *PREPARE* phase, especially *dataset generation*, is achieved with the [`dataset`](https://github.com/packing-box/docker-packing-box/blob/main/src/files/tools/dataset) tool.

**Operation** | **Description** | **Command**
:---:| --- | ---
**`make`** ![](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/imgs/dataset-operations-make.png) | Make a new dataset, either fully packed or mixed with not-packed samples | `# dataset make dataset -f PE -n 200 -s /path/to/pe`
**`merge`** ![](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/imgs/dataset-operations-merge.png) | Merge two datasets | `# dataset merge dataset dataset2`
**`select`** ![](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/imgs/dataset-operations-select.png) | Select a subset of a dataset to create a new one | `# dataset select dataset dataset2 -q "format == 'PE32'"`
**`update`** ![](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/imgs/dataset-operations-update.png) | Update a dataset with new samples given their labels | `# dataset update dataset -l labels.json -s folder-of-executables`

### Data Visualization

The *VISUALIZE* phase can be performed with the [`dataset`](https://github.com/packing-box/docker-packing-box/blob/main/src/files/tools/dataset) and [`visualizer`](https://github.com/packing-box/docker-packing-box/blob/main/src/files/tools/visualizer) tools.

In order to visualize feature values:

```console
$ dataset plot test-mix byte_0_after_ep byte_1_after_ep --multiclass
```

<p align="center"><img src="https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/imgs/data-visualization-features.png"></p>

In order to visualize samples (aims to compare the not-packed and some packed versions):

```console
$ visualizer plot "PsExec.exe$" dataset -s -l not-packed -l MEW -l RLPack -l UPX
```

<p align="center"><img src="https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/imgs/data-visualization-psexec.png"></p>

This will work for instance for a structure formatted as such:

```
folder/
  +-- not-packed/PsExec.exe
  +-- packed
        +-- MEW/mew_PsExec.exe
        +-- RLPack/rlpack_PsExec.exe
        +-- UPX/upx_PsExec.exe
```

### Model Manipulations

The *TRAIN* and *PREDICT* phases of the pipeline are achieved with the [`model`](https://github.com/packing-box/docker-packing-box/blob/main/src/files/tools/model) tool. 

**Operation** | **Description** | **Command**
:---:| --- | ---
**`compare`** ![](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/imgs/model-operations-compare.png) | Compare the performance metrics of multiple models | `# model compare model --dataset dataset --model model2`
**`test`** ![](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/imgs/model-operations-test.png) | Test a model on a given dataset | `# model test model --name dataset`
**`train`** ![](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/imgs/model-operations-train.png) | Train a model given an algorithm and input dataset | `# model train dataset --algorithm dt`


## :star: Related Projects

You may also like these:

- [Awesome Executable Packing](https://github.com/packing-box/awesome-executable-packing): A curated list of awesome resources related to executable packing.
- [Bintropy](https://github.com/packing-box/bintropy): Analysis tool for estimating the likelihood that a binary contains compressed or encrypted bytes (inspired from [this paper](https://ieeexplore.ieee.org/document/4140989)).
- [Dataset of packed ELF files](https://github.com/packing-box/dataset-packed-elf): Dataset of ELF samples packed with many different packers.
- [Dataset of packed PE files](https://github.com/packing-box/dataset-packed-pe): Dataset of PE samples packed with many different packers (fork of [this repository](https://github.com/chesvectain/PackingData)).
- [DSFF](https://github.com/packing-box/python-dsff): Library implementing the DataSet File Format (DSFF).
- [PEiD](https://github.com/packing-box/peid): Python implementation of the well-known Packed Executable iDentifier ([PEiD](https://www.aldeid.com/wiki/PEiD)).
- [PyPackerDetect](https://github.com/packing-box/pypackerdetect): Packing detection tool for PE files (fork of [this repository](https://github.com/cylance/PyPackerDetect)).
- [REMINDer](https://github.com/packing-box/reminder): Packing detector using a simple heuristic (inspired from [this paper](https://ieeexplore.ieee.org/document/5404211)).


## :books: Related Readings

- arXiv - [Experimental Toolkit for Manipulating Executable Packing](https://arxiv.org/abs/2302.09286)
- Black Hat Arsenal EU 2022 - [Packing-Box: Playing with Executable Packing](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/material/bheu22-packingbox.pdf)
- Black Hat Arsenal EU 2023 - [Packing-Box: Breaking Detectors & Visualizing Packing](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/material/bheu23-packingbox.pdf)
- Medium Blog Post - [Unpacking the Potential of "Packing Box"](https://medium.com/@elniak/unpacking-the-potential-of-packing-box-dfd765609233)


## :clap:  Supporters

[![Stargazers repo roster for @packing-box/docker-packing-box](https://reporoster.com/stars/dark/packing-box/docker-packing-box)](https://github.com/packing-box/docker-packing-box/stargazers)

[![Forkers repo roster for @packing-box/docker-packing-box](https://reporoster.com/forks/dark/packing-box/docker-packing-box)](https://github.com/packing-box/docker-packing-box/network/members)

<p align="center"><a href="#top"><img src="https://img.shields.io/badge/Back%20to%20top--lightgrey?style=social" alt="Back to top" height="20"/></a></p>

