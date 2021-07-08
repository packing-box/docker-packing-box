## Introduction

The *Packing Box* is a Docker image containing various tools for studying executable packing. It is designed to instrumentalize packers and packing detectors and to create datasets for comparing packing detection techniques.

It aims to provide a single platform for studying packing for more than what we can traditionally see in the literature, that is, techniques applied to Windows PE files. From the box (based on Ubuntu), [Wine](https://www.winehq.org/) is used to handle PE files but it also allows to use the ELF format. Ultimately, it aims to also include the Mach-O format (using [Darling](https://www.darlinghq.org/)).

## Setup

Setting up the *Packing Box* requires installing Docker and cloning this repository. The [`Dockerfile`](https://github.com/dhondta/docker-packing-box/blob/main/Dockerfile) can then be used to build the image.

```session
# apt install docker
[...]
# git clone https://github.com/dhondta/docker-packing-box
[...]
# docker build -t dhondta/packing-box .
[...]
Successfully tagged dhondta/packing-box:latest
# 
```

## Run

Running the *Packing Box* is done by starting it up with Docker.

```session
# docker run -it -h packing-box -v `pwd`:/mnt/share dhondta/packing-box
                              .______      ___       ______  __  ___  __  .__   __.   _______ .______     ______   ___   ___
                              |   _  \    /   \     /      ||  |/  / |  | |  \ |  |  /  _____||   _  \   /  __  \  \  \ /  /
                              |  |_)  |  /  ^  \   |  ,----'|  '  /  |  | |   \|  | |  |  __  |  |_)  | |  |  |  |  \  V  /
                              |   ___/  /  /_\  \  |  |     |    <   |  | |  . `  | |  | |_ | |   _  <  |  |  |  |   >   <
                              |  |     /  _____  \ |  `----.|  .  \  |  | |  |\   | |  |__| | |  |_)  | |  `--'  |  /  .  \
                              | _|    /__/     \__\ \______||__|\__\ |__| |__| \__|  \______| |______/   \______/  /__/ \__\



  This Docker image is a ready-to-use platform for making datasets of packed and not packed executables, especially for training machine learning models.

[...]

┌──[root@packing-box]──[/mnt/share]──[main|+6…6]────────                           ────[172.17.0.2]──[12:34:56]──[2.00]────
# 

```

From inside the box, help can be obtained by using the "`?`" tool. This will display the welcome message seen when starting it. It also has some options ;

- `-i ITEM`: allows to see the particular help of an item from the box

```session
# \? -i peid    
                                           .______    _______  __   _______
                                           |   _  \  |   ____||  | |       \
                                           |  |_)  | |  |__   |  | |  .--.  |
                                           |   ___/  |   __|  |  | |  |  |  |
                                           |  |      |  |____ |  | |  '--'  |
                                           | _|      |_______||__| |_______/



PEiD detects most common packers, cryptors and compilers for PE files. It can currently detect more than 600 different signatures in PE files.
Source: https://github.com/wolfram77web/app-peid
Applies to: PE32, PE64

References
  1. https://www.aldeid.com/wiki/PEiD
  2. www.softpedia.com/get/Programming/Packers-Crypters-Protectors/PEiD-updated.shtml
  3. https://github.com/ynadji/peid/

```

- `--show-all`: will make the tool display the whole items, even those that are marked as *broken*.

