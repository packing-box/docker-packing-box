# Executable Packing

*Packers* are programs that pack executable samples. They can be categorized in different types including compression, encryption, obfuscation, virtualization and anti-reversing techniques.

## Tool

A [dedicated tool](https://github.com/dhondta/docker-packing-box/blob/main/files/tools/packer) called `packer` is provided with the [*Packing Box*](https://github.com/dhondta/docker-packing-box) to pack samples. It is especially useful for **mass-packing** as either a single sample or a folder of samples can be specified as input, as its help message tells.

```console
┌──[user@packing-box]──[/mnt/share]────────
$ packer --help
[...]
This tool simply packs (using Packer.pack) with the selected packer an input executable or folder of executables.
[...]
usage: packer [-p PREFIX] [-h] [--help] [-v] packer executable

positional arguments:
  packer      selected packer
  executable  executable or folder containing executables

options:
  -p PREFIX, --prefix PREFIX
                        string to be prepended to the filename (default: None)
[...]
```

From the optional arguments, we see that it allows to specify a prefix while saving packed samples. This may be useful for avoiding sample name clashes when different packers need to be applied to a same set of not-packed samples so that they get merged into a big dataset afterwards.

