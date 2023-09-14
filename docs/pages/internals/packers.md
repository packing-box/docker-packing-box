# Packers

The `Packer` class allows to abstract packers, based on the [`Base` class](items.md#base-class) and adding a special packing method called `pack`.

```session
>>> from pbox import Packer
```

## `Packer` Class

This [class](https://github.com/packing-box/docker-packing-box/blob/main/src/lib/src/pbox/core/items/packer.py#L30) is the base for abstracted tools loaded from the [`packers.yml`](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/packers.yml) description file. It holds the registry of all the loaded child classes.

```session
>>> Packer.registry
[<pbox.items.packer.Amber object at 0x7f312b953220>, [...]
```

**Special methods**:

- `pack(executable)`: for packing an input executable
- `run(executable)`: overridden `run` method for handling parametrized packers (i.e. generating a password and including it in the label)

!!! note "Packing validation"
    
    When trying to pack an executable, the decision of whether it was successfully packed or not is made based on the change of SHA256 hash. If no change, the executable is included but with an unpacked label.
    
    Currently, it does not support checking whether the executable still runs after packing.

