<h1 align="center">Packing Box <a href="https://twitter.com/intent/tweet?text=Packing%20Box%20-%20A%20Docker%20container%20featuring%20many%20packers,%20unpackers%20and%20detectors%20for%20studying%20executable%20packing,%20including%20machine%20learning%20dataset%20generation%20and%20algorithms.%0D%0Ahttps%3a%2f%2fgithub%2ecom%2fdhondta%2fdocker-packing-box%0D%0A&hashtags=docker,pe,elf,macho,packer,unpacker,packingdetection,peid,upx"><img src="https://img.shields.io/badge/Tweet--lightgrey?logo=twitter&style=social" alt="Tweet" height="20"/></a></h1>
<h3 align="center">Study executable packing in a dedicated platform.</h3>

[![Read The Docs](https://readthedocs.org/projects/docker-packing-box/badge/?version=latest)](http://docker-packing-box.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)

This Docker image aims to regroup multiple common executable packers and make datasets of packed executables.

![](docs/screenshot.png)

## :fast_forward: Quick Start

```sh
$ docker build -t dhondta/packing-box .
[...]
<<<wait for a while>>>
[...]
$ docker run -it -h packing-box -v `pwd`:/mnt/share dhondta/packing-box

┌──[root@packing-box]──[/]────────                     ────[172.17.0.2]──[12:34:56]──[0.12]────
# 
```

## :bulb: TODO

- Check this [link](http://protools.narod.ru/packers.htm)
- Check this [link](https://in4k.github.io/wiki/exe-packers-tweakers-and-linkers) for new ideas
- Check this [link](https://www.softpedia.com/catList/14,1,3,0,1.html) for new ideas
- Check this [link](https://storage.ey.md/Technology%20Related/Programming%20%26%20Reversing/Tuts4You%20Collection/Unpacking%20Tutorials/)
- Check this [link](https://storage.ey.md/Technology%20Related/Programming%20%26%20Reversing/Tuts4You%20Collection/UnPackMe%20Collection/)
- Check this [link](https://gist.github.com/joenorton8014/bddbbb3c068f7d9e7875d741a8f9b0f6)
- Install [PoCrypt](https://github.com/picoflamingo/pocrypt)
- Install [EXE Bundle](https://www.softpedia.com/get/Security/Security-Related/EXE-Stealth-Packer.shtml) (could be the same or a further version of EXE Stealth Packer)
- Install [EXE Stealth Packer](https://www.webtoolmaster.com/packer.htm)
- Install [iPackk](http://www.pouet.net/prod.php?which=29185)
- Install [NetShrink](https://www.pelock.com/products/netshrink) ([PELock](https://www.pelock.com/) suite)
- Install [oneKpaq](http://www.pouet.net/prod.php?which=66926)
- https://github.com/fireeye/capa-rules/tree/master/anti-analysis/packer
- https://storage.ey.md/Technology%20Related/Programming%20%26%20Reversing/Tuts4You%20Collection/Unpacking%20Tutorials/
- https://reverseengineering.stackexchange.com/questions/3184/packers-protectors-for-linux
- https://reverseengineering.stackexchange.com/questions/1545/linux-protectors-any-good-one-out-there


## :star: Related Projects

You may also like these:

- [Awesome Executable Packing](https://github.com/dhondta/awesome-executable-packing): A curated list of awesome resources related to executable packing.
- [Bintropy](https://github.com/dhondta/bintropy): Analysis tool for estimating the likelihood that a binary contains compressed or encrypted bytes.
- [PEiD](https://github.com/dhondta/peid): Python implementation of the Packed Executable iDentifier (PEiD).
- [PyPackerDetect](https://github.com/dhondta/PyPackerDetect): Packing detection tool for PE files.


## :clap:  Supporters

[![Stargazers repo roster for @dhondta/docker-packing-box](https://reporoster.com/stars/dark/dhondta/docker-packing-box)](https://github.com/dhondta/docker-packing-box/stargazers)

[![Forkers repo roster for @dhondta/docker-packing-box](https://reporoster.com/forks/dark/dhondta/docker-packing-box)](https://github.com/dhondta/docker-packing-box/network/members)

<p align="center"><a href="#"><img src="https://img.shields.io/badge/Back%20to%20top--lightgrey?style=social" alt="Back to top" height="20"/></a></p>

