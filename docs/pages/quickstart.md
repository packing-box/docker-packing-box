When running the [Docker image](index.html#run), we are presented with a root shell that starts with a help message like this:

![](https://raw.githubusercontent.com/dhondta/docker-packing-box/main/docs/screenshot.png)

## Getting Help

Help can be obtained by using the "`?`" tool. This will display the welcome message seen when starting it. It also has an option `-i ITEM` allowing to see the particular help of an item from the box

```session
# \? -i upx
                                                                                           __    __  .______   ___   ___
                                                                                          |  |  |  | |   _  \  \  \ /  /
                                                                                          |  |  |  | |  |_)  |  \  V  /
                                                                                          |  |  |  | |   ___/    >   <
                                                                                          |  `--'  | |  |       /  .  \
                                                                                           \______/  | _|      /__/ \__\



  UPX is a free, portable, extendable, high-performance executable packer for several executable formats.
  Source: https://upx.github.io
  Applies to: ELF, MSDOS, Mach-O, PE

 References
    1. https://linux.die.net/man/1/upx
    2. https://github.com/fireeye/capa-rules/blob/master/anti-analysis/packer/upx/packed-with-upx.yml
    3. https://storage.ey.md/Technology%%20Related/Programming%%20%%26%%20Reversing/Tuts4You%%20Collection/UnPackMe%%20Collection/PE32/UPX%%201.25.rar
    4. https://storage.ey.md/Technology%%20Related/Programming%%20%%26%%20Reversing/Tuts4You%%20Collection/UnPackMe%%20Collection/PE32/UPX%%203.04.rar
    5. https://storage.ey.md/Technology%%20Related/Programming%%20%%26%%20Reversing/Tuts4You%%20Collection/Unpacking%%20Tutorials/UPX%%20%%28Unpacking%%29.rar
    6. https://www.aldeid.com/wiki/Category:Digital-Forensics/Computer-Forensics/Anti-Reverse-Engineering/Packers/UPX

```

## Main Tool

From the help message, the section named *Tools* shows the list of availble tools. The main one is `packing-box`.

```session
PackingBox 1.1.0
Author: Alexandre D'Hondt (alexandre.dhondt@gmail.com)
Copyright: © 2021 A. D'Hondt
License: GNU General Public License v3.0

This utility aims to facilitate detectors|packers|unpackers' setup|test according to the related YAML data file.

usage: packing-box [-h] [--help] [-v] {setup,test} ...

positional arguments:
  {setup,test}  command to be executed
    setup       setup something
    test        test something

extra arguments:
  -h             show usage message and exit
  --help         show this help message and exit
  -v, --verbose  verbose mode (default: False)

Usage examples:
  packing-box setup packer
  packing-box setup detector peid
  packing-box test packer upx ezuri midgetpack
  packing-box test -b unpacker upx

```

The help message of this tool shows some examples for setting up and testing items.

## Specific Tools

From the *Tools* section in the output of the `?` tool, we also see:

- `dataset`: for making datasets of packed and not packed executables ; this can also ingest existing datasets provided their labels
- `detector`: for testing integrated detectors ; this works on various input formats: single executable | folder of executables | dataset
- `model`: for training machine learning models and observing model parameters
