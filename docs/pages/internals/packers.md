# Packers

The `Packer` class allows to abstract packers, based on the [`Base` class](items.html#base-class) and adding a special packing method called `pack`.

```session
>>> from pbox import Packer
```

## `Packer` Class

This [class](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/packer.py#L13) is the base for abstracted tools loaded from the [`packers.yml`](https://github.com/dhondta/docker-packing-box/tree/main/packers.yml) description file. It holds the registry of all the loaded child classes.

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

