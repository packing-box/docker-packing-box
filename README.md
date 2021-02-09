# Packing Box

This Docker image aims to regroup multiple common executable packers and make datasets of packed executables.

## Currently installed packers

- [Amber](https://github.com/EgeBalci/amber)
- [APack](https://www.ibsensoftware.com/download.html)
- [Ezuri](https://github.com/guitmz/ezuri)
- [KKrunchy](http://www.farbrausch.de/~fg/kkrunchy/)
- [m0dern_p4cker](https://github.com/n4sm/m0dern_p4cker)
- [PEtite](https://www.un4seen.com/petite/)
- [UPX](https://upx.github.io/)

> **GUI (Windows)**
> 
> - [MEW](https://www.softpedia.com/get/Programming/Packers-Crypters-Protectors/MEW-SE.shtml)
> - [NetCrypt](https://github.com/friedkiwi/netcrypt)
> - [tElock](https://www.softpedia.com/get/Programming/Packers-Crypters-Protectors/Telock.shtml)

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

> **Note**: Build from intermediate step
> 
> This allows not to reinstall from scratch, i.e. when adding a new **packer**.
> 
> ```sh
> $ docker build -t dhondta/packing-box --target packers .
> ```
> 
> This allows not to reinstall from scratch, i.e. when adding a new **tool**.
> 
> ```sh
> $ docker build -t dhondta/packing-box --target tools .
> ```

## TODO

- Check this [link](https://in4k.github.io/wiki/exe-packers-tweakers-and-linkers) for new ideas.
- Check this [link](https://www.softpedia.com/catList/14,1,3,0,1.html) for new ideas.
- Install [ASPack](http://www.aspack.com/) (demo version)
- Install [Crinkler](http://www.crinkler.net/crinkler20.zip)
- Install [EXE Bundle](https://www.softpedia.com/get/Security/Security-Related/EXE-Stealth-Packer.shtml) (could be the same or a further version of EXE Stealth Packer)
- Install [EXE Stealth Packer](https://www.webtoolmaster.com/packer.htm)
- Install [FSG](http://in4k.untergrund.net/packers%20droppers%20etc/xt_fsg20.zip)
- Install [iPackk](http://www.pouet.net/prod.php?which=29185)
- Install [muncho](http://www.pouet.net/prod.php?which=51324)
- Install [NetShrink](https://www.pelock.com/products/netshrink) ([PELock](https://www.pelock.com/) suite)
- Install [oneKpaq](http://www.pouet.net/prod.php?which=66926)
- Install [PELock](https://www.pelock.com/products/pelock) ([PELock](https://www.pelock.com/) suite)
- Install [PE Packer](https://github.com/czs108/PE-Packer)
- Make Python tool that allows to train ML models (e.g. RF or MLP) and compare their performance
- Use [Xvfb](https://superuser.com/questions/902175/run-wine-totally-headless) to run GUI apps through wine in headless mode
- Fix MEW, NetCrypt, Telock, ... GUI issue

