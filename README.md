# Packing Box

This Docker image aims to regroup multiple common executable packers and make datasets of packed executables.

## Currently installed packers

**Tested and functional**:
- [Amber](https://github.com/EgeBalci/amber)
- [Ezuri](https://github.com/guitmz/ezuri)
- [m0dern_p4cker](https://github.com/n4sm/m0dern_p4cker)
- [MidgetPack](https://github.com/arisada/midgetpack)
- [UPX](https://upx.github.io/)

**To be tested**:
- [APack](https://www.ibsensoftware.com/download.html)
- [ASPack](http://www.aspack.com/) (demo version)
- [Crinkler](http://www.crinkler.net)
- [KKrunchy](http://www.farbrausch.de/~fg/kkrunchy/)
- [MPRESS](https://www.softpedia.com/get/Programming/Packers-Crypters-Protectors/MPRESS.shtml)
- [.NetZ](https://www.softpedia.com/get/Programming/Packers-Crypters-Protectors/NETZ.shtml)
- [PEtite](https://www.un4seen.com/petite/)

> **Mac OS X**
> - [muncho](http://www.pouet.net/prod.php?which=51324)

> **GUI (Windows)**
> - [FSG](http://in4k.untergrund.net/packers%20droppers%20etc/xt_fsg20.zip)
> - [MEW](https://www.softpedia.com/get/Programming/Packers-Crypters-Protectors/MEW-SE.shtml)
> - [NetCrypt](https://github.com/friedkiwi/netcrypt)
> - [RLPack](https://www.softpedia.com/get/Programming/Packers-Crypters-Protectors/RLPack-Basic-Edition.shtml)
> - [tElock](https://www.softpedia.com/get/Programming/Packers-Crypters-Protectors/Telock.shtml)

> **Not working**
> - [BurnEye](https://packetstormsecurity.com/files/29691/burneye-1.0-linux-static.tar.gz.html)
> - [ELFuck](https://github.com/timhsutw/elfuck)
> - [Shiva](https://packetstormsecurity.com/files/31087/shiva-0.95.tar.gz.html)

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

