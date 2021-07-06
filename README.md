[![Read The Docs](https://readthedocs.org/projects/docker-packing-box/badge/?version=latest)](http://docker-packing-box.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Packing Box

This Docker image aims to regroup multiple common executable packers and make datasets of packed executables.

![](docs/screenshot.png)

## Quick Start

```sh
$ docker build -t dhondta/packing-box .
[...]
<<<wait for a while>>>
[...]
$ docker run -it -h packing-box -v `pwd`:/mnt/share dhondta/packing-box

┌──[root@packing-box]──[/]────────                     ────[172.17.0.2]──[12:34:56]──[0.12]────
# 
```

## TODO

- Use [Xvfb](https://superuser.com/questions/902175/run-wine-totally-headless) to run GUI apps through wine in headless mode
- Check this [link](https://webscene.ir/tools/Packers-and-protectors).
- Check this [link](http://protools.narod.ru/packers.htm).
- Check this [link](https://in4k.github.io/wiki/exe-packers-tweakers-and-linkers) for new ideas.
- Check this [link](https://www.softpedia.com/catList/14,1,3,0,1.html) for new ideas.
- Check this [link](https://storage.ey.md/Technology%20Related/Programming%20%26%20Reversing/Tuts4You%20Collection/Unpacking%20Tutorials/).
- Check this [link](https://storage.ey.md/Technology%20Related/Programming%20%26%20Reversing/Tuts4You%20Collection/UnPackMe%20Collection/).
- Install [EXE Bundle](https://www.softpedia.com/get/Security/Security-Related/EXE-Stealth-Packer.shtml) (could be the same or a further version of EXE Stealth Packer)
- Install [EXE Stealth Packer](https://www.webtoolmaster.com/packer.htm)
- Install [iPackk](http://www.pouet.net/prod.php?which=29185)
- Install [NetShrink](https://www.pelock.com/products/netshrink) ([PELock](https://www.pelock.com/) suite)
- Install [oneKpaq](http://www.pouet.net/prod.php?which=66926)
- Install [PELock](https://www.pelock.com/products/pelock) ([PELock](https://www.pelock.com/) suite)
- Install [PE Packer](https://github.com/czs108/PE-Packer)
- Make Python tool that allows to train ML models (e.g. RF or MLP) and compare their performance
- https://github.com/fireeye/capa-rules/tree/master/anti-analysis/packer
- https://storage.ey.md/Technology%20Related/Programming%20%26%20Reversing/Tuts4You%20Collection/Unpacking%20Tutorials/
- https://reverseengineering.stackexchange.com/questions/3184/packers-protectors-for-linux
- https://reverseengineering.stackexchange.com/questions/1545/linux-protectors-any-good-one-out-there

