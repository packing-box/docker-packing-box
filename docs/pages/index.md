# Introduction

The [*Packing Box*](https://github.com/packing-box/docker-packing-box) is a Docker image containing various tools for studying executable packing. It is designed to instrumentalize packers and packing detectors and to create datasets for comparing packing detection techniques.

It aims to provide a single platform for studying packing for more than what we can traditionally see in the literature, that is essentially techniques applied to Windows PE files in the scope of malware analysis. From the [*Packing Box*](https://github.com/packing-box/docker-packing-box) (based on Ubuntu, therefore supporting the ELF format), [Wine](https://www.winehq.org/) is used to handle PE files. Ultimately, it aims to also handle the Mach-O format (using [Darling](https://www.darlinghq.org/)).

Check out the [slideshow presented at Black Hat Arsenal Europe 2022](https://raw.githubusercontent.com/packing-box/docker-packing-box/main/docs/pages/assets/bheu22-presentation.pdf) !

## Setup

Setting up the [*Packing Box*](https://github.com/packing-box/docker-packing-box) requires installing [Docker](https://www.docker.com) and cloning this repository. The [`Dockerfile`](https://github.com/dhondta/docker-packing-box/blob/main/Dockerfile) can then be used to build the image.

```console
# apt install docker
[...]
# git clone https://github.com/packing-box/docker-packing-box
[...]
# docker build -t packing-box .
[...]
```

## Run

Running the [*Packing Box*](https://github.com/packing-box/docker-packing-box) is done by starting it up with [Docker](https://www.docker.com).

```console
# docker run -it -h packing-box -v `pwd`:/mnt/share packing-box
                               .______      ___       ______  __  ___  __  .__   __.   _______        .______     ______   ___   ___
                               |   _  \    /   \     /      ||  |/  / |  | |  \ |  |  /  _____|       |   _  \   /  __  \  \  \ /  /
                               |  |_)  |  /  ^  \   |  ,----'|  '  /  |  | |   \|  | |  |  __   ______|  |_)  | |  |  |  |  \  V  /
                               |   ___/  /  /_\  \  |  |     |    <   |  | |  . `  | |  | |_ | |______|   _  <  |  |  |  |   >   <
                               |  |     /  _____  \ |  `----.|  .  \  |  | |  |\   | |  |__| |        |  |_)  | |  `--'  |  /  .  \
                               | _|    /__/     \__\ \______||__|\__\ |__| |__| \__|  \______|        |______/   \______/  /__/ \__\



  This Docker image is a ready-to-use platform for making datasets of packed and not packed executables, especially for training machine learning models.

[...]

┌──[user@packing-box]──[/mnt/share]──[master|✓]────────                           ────[172.17.0.2]──[12:34:56]──[2.00]────
# 

```

From inside the box, help can be obtained by using the "`?`" tool. This will display the welcome message seen when starting it. It has a few useful options ;

- `-f FORMAT` / `--format FORMAT`: allows to filter out items based on the input executable format

- `-i ITEM` / `--item ITEM`: allows to see the particular help of an item from the box

```console
# ? -i peid    
                                                                 .______    _______  __   _______
                                                                 |   _  \  |   ____||  | |       \
                                                                 |  |_)  | |  |__   |  | |  .--.  |
                                                                 |   ___/  |   __|  |  | |  |  |  |
                                                                 |  |      |  |____ |  | |  '--'  |
                                                                 | _|      |_______||__| |_______/




PEiD detects most common packers, cryptors and compilers for PE files.


Source    : https://github.com/dhondta/peid


Applies to: .NET, MSDOS, PE32, PE64



References

  1 https://www.aldeid.com/wiki/PEiD

  2 www.softpedia.com/get/Programming/Packers-Crypters-Protectors/PEiD-updated.shtml

  3 https://github.com/ynadji/peid/

  4 https://github.com/wolfram77web/app-peid

```

- `-k {detectors,packers,unpackers}` / `--keep {detectors,packers,unpackers}`: allows to filter out items of a single kind

- `--show-all`: will make the tool display the whole items, even those that are marked as *broken*.

