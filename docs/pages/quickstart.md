# Quick Start

## Getting started

The box can be built with the following command (note that it takes a while as it installs many underlying packages) :

```console
$ docker build -t packing-box .
```

When running the [Docker image](index.html#run) ;

```console
$ docker run -it -h packing-box -v `pwd`:/mnt/share packing-box
```

We are presented with a root shell that starts with a help message like this:

![](https://raw.githubusercontent.com/dhondta/docker-packing-box/main/docs/pages/imgs/screenshot.png)


## Getting Help

Help can be obtained by using the "`?`" tool. This will display the welcome message seen when starting it. It has an option `-i ITEM` / `--item ITEM` allowing to see the specific help of an item from the box.

```console
$ ? -i upx
                                                               __    __  .______   ___   ___
                                                              |  |  |  | |   _  \  \  \ /  /
                                                              |  |  |  | |  |_)  |  \  V  /
                                                              |  |  |  | |   ___/    >   <
                                                              |  `--'  | |  |       /  .  \
                                                               \______/  | _|      /__/ \__\




  UPX is a free, portable, extendable, high-performance executable packer for several executable formats.
  Source    : https://upx.github.io
  Applies to: ELF32, ELF64, MSDOS, Mach-O32, Mach-O64, Mach-Ou, PE32, PE64
  Categories: compressor

 References
    1. https://linux.die.net/man/1/upx
    2. https://github.com/fireeye/capa-rules/blob/master/anti-analysis/packer/upx/packed-with-upx.yml
    3. https://storage.ey.md/Technology%%20Related/Programming%%20%%26%%20Reversing/Tuts4You%%20Collection/UnPackMe%%20Collection/PE32/UPX%%201.25.rar
    4. https://storage.ey.md/Technology%%20Related/Programming%%20%%26%%20Reversing/Tuts4You%%20Collection/UnPackMe%%20Collection/PE32/UPX%%203.04.rar
    5. https://storage.ey.md/Technology%%20Related/Programming%%20%%26%%20Reversing/Tuts4You%%20Collection/Unpacking%%20Tutorials/UPX%%20%%28Unpacking%%29.rar
    6. https://www.aldeid.com/wiki/Category:Digital-Forensics/Computer-Forensics/Anti-Reverse-Engineering/Packers/UPX
```

The help tool has also a `-k {detectors|packers|unpackers}` / `--keep {detectors|packers|unpackers}` option to only display specific classes of items. The same applies to formats with the `-f FORMAT` / `--format FORMAT` option. For instance, if we want to display packers for the Portable Executable (PE) format, we can type:

```console
$ ? -k packers -f PE
                                  .______      ___       ______  __  ___  __  .__   __.   _______ .______     ______   ___   ___
                                  |   _  \    /   \     /      ||  |/  / |  | |  \ |  |  /  _____||   _  \   /  __  \  \  \ /  /
                                  |  |_)  |  /  ^  \   |  ,----'|  '  /  |  | |   \|  | |  |  __  |  |_)  | |  |  |  |  \  V  /
                                  |   ___/  /  /_\  \  |  |     |    <   |  | |  . `  | |  | |_ | |   _  <  |  |  |  |   >   <
                                  |  |     /  _____  \ |  `----.|  .  \  |  | |  |\   | |  |__| | |  |_)  | |  `--'  |  /  .  \
                                  | _|    /__/     \__\ \______||__|\__\ |__| |__| \__|  \______| |______/   \______/  /__/ \__\



 
  This Docker image is a ready-to-use platform for making datasets of packed and not packed executables, especially for training machine learning models.

 Packers (10/22)
  ──────────────────  ──────────────────────────  ──────  ──────────────────────────────────────────────────────────────────────────────────────────────────
  Name                Targets                     Status  Source
  Amber               PE                          ☑         https://github.com/EgeBalci/amber
  BeRo                PE32                        ☒         https://blog.rosseaux.net/page/875fbe6549aa072b5ee0ac9cefff4827/BeRoEXEPacker
  DotNetZ             .NET                        ☒         https://www.softpedia.com/get/Programming/Packers-Crypters-Protectors/NETZ.shtml
  EXE32Pack           PE32                        ☒         https://exe32pack.apponic.com
  Enigma_Virtual_Box  PE64,PE32                   ☒         https://www.enigmaprotector.com/en/aboutvb.html
  Eronana_Packer      PE32                        ☑         https://github.com/Eronana/packer
  hXOR-Packer         PE                          ☑         https://github.com/rurararura/hXOR-Packer
  Kkrunchy            PE32                        ☑         http://www.farbrausch.de/~fg/kkrunchy
  MEW                 PE32                        ☑         https://in4k.github.io/wiki/exe-packers-tweakers-and-linkers
  MPRESS              Mach-O,PE                   ☑         https://www.softpedia.com/get/Programming/Packers-Crypters-Protectors/MPRESS.shtml
  NetCrypt            .NET                        ☒         https://github.com/friedkiwi/netcrypt
  PE-Packer           PE32                        ☑         https://github.com/czs108/PE-Packer
  PEtite              PE32                        ☑         https://www.un4seen.com/petite
  PEzor               PE                          ☒         https://github.com/phra/PEzor
  RLPack              PE                          ☒         https://www.softpedia.com/get/Programming/Packers-Crypters-Protectors/RLPack-Basic-Edition.shtml
  Silent_Packer       PE,ELF                      ☒         https://github.com/SilentVoid13/Silent_Packer
  SimpleDpack         PE32                        🗗         https://github.com/YuriSizuku/SimpleDpack
  Telock              PE                          🗗         https://www.softpedia.com/get/Programming/Packers-Crypters-Protectors/Telock.shtml
  TheArk              PE32                        ☒         https://github.com/aaaddress1/theArk
  UPX                 PE64,PE32,MSDOS,Mach-O,ELF  ☑         https://upx.github.io
  Yoda_Crypter        PE32                        ☑         https://sourceforge.net/projects/yodap
  Yoda_Protector      PE32                        🗗         http://yodap.sourceforge.net
  ──────────────────  ──────────────────────────  ──────  ──────────────────────────────────────────────────────────────────────────────────────────────────

 Legend
  🗗  gui ; ☒  not installed ; ☑  ok

```

## Items management

From the help message, the section named *Tools* shows the list of available tools. The main one is `packing-box`. It is aimed to install and test items but also display or edit the workspace or even configure options like the workspace's folder.

```console
$ packing-box --help
PackingBox 1.2.4
Author   : Alexandre D'Hondt (alexandre.dhondt@gmail.com)
Copyright: © 2021-2022 A. D'Hondt
License  : GNU General Public License v3.0

This utility aims to facilitate detectors|packers|unpackers' setup|test according to the related YAML data file.

usage: packing-box [-h] [--help] [-v] [-f LOGFILE] [-r] [-s] {clean,config,list,setup,test,workspace} ...

positional arguments:
  {clean,config,list,setup,test,workspace}
                        command to be executed
    clean               cleanup temporary folders
    config              set a config option
    list                list something
    setup               setup something
    test                test something
    workspace           inspect the workspace

extra arguments:
  -h                    show usage message and exit
  --help                show this help message and exit
  -v, --verbose         verbose mode (default: False)
  -f LOGFILE, --logfile LOGFILE
                        destination log file (default: None)
  -r, --relative        display relative time (default: False)
  -s, --syslog          log to /var/log/syslog (default: False)

Usage examples:
  packing-box config --workspace /home/user/my-workspace --packers path/to/packers.yml
  packing-box list config
  packing-box list packers --name
  packing-box list detectors --all
  packing-box setup packer
  packing-box setup detector peid
  packing-box setup analyzer gettyp
  packing-box test packer upx ezuri midgetpack
  packing-box test -b unpacker upx
  packing-box workspace view
  packing-box workspace edit MyDataset/data.csv
```

The tool's help message provides a few interesting examples of possible actions.

## Playing with datasets

A dedicated tool called `dataset` allows to manipulate datasets.

The following simple example creates a dataset called "`test-pe-upx`" consisting of 100 PE files (including 32-bits and 64-bits) from the source folders of Wine integrated in the box, balanced between not-packed and UPX-packed samples.

```console
$ dataset make test-pe-upx -n 100 --format PE --packer upx
```

The following command lets us see some details of the newly created dataset (including some metadata and its statistics per size ranges for not-packed and packed samples):

```console
$ dataset show test-pe-upx
```

Up to now, we have built a dataset that contains the samples. The next command will convert it to a fileless dataset, that is with the features computed. This has the advantages of precomputing the features and thus speeding up further processings like model training and drastically reducing the size of the dataset (especially useful when planning to exchange it with other researchers). But it has the disadvantage to remove the files, therefore making the dataset unusable with integrated detectors or even preventing from feature modifications.

```console
$ dataset convert test-pe-upx
```

## Playing with models

A dedicated tool called `model` allows to manipulate models.

The following simple example trains a Random Forest on a reference dataset called "`test-pe-upx`".

```console
$ model train test-pe-upx -a rf
```

Once trained, the model receives an auto-generated name that includes the name of the reference dataset, the executable formats it applies to, the number of samples from the reference dataset, the training algorithm and the number of considered features. In order to find back this name, the list command can be used to display all the existing models:

```console
$ model list
```

Let us assume that the newly trained model was named `test-pe-upx_pe32-pe64_100_rf_f111`, we can show its details (including metadata) by using the following command:

```console
$ model show test-pe-upx_pe32-pe64_100_rf_f111
```

If we want to test it against whatever dataset we want, we can use the following command (here, we test on the reference dataset itself):

```console
$ model test test-pe-upx_pe32-pe64_100_rf_f111 test-pe-upx
```

