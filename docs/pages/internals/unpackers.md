# Unpackers

The `Unpacker` class allows to abstract unpackers, based on the [`Base` class](items.md#base-class) and adding a special unpacking method called `unack`.

```session
>>> from pbox import Unpacker
```

## `Unpacker` Class

This [class](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/unpacker.py#L12) is the base for abstracted tools loaded from the [`unpackers.yml`](https://github.com/dhondta/docker-packing-box/tree/main/unpackers.yml) description file. It holds the registry of all the loaded child classes.

```session
>>> Unpacker.registry
[<pbox.items.unpacker.UPX object at 0x7f313a0faf70>]
```

**Special methods**:

- `unpack(executable)`: for unpacking an input executable

