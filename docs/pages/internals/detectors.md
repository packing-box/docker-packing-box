# Detectors

The `Detector` class allows to abstract detection tools, based on the [`Base` class](items.md#base-class) and adding a special detection method called `detect`.

```session
>>> from pbox import Detector
```

## `Detector` Class

This [class](https://github.com/dhondta/docker-packing-box/blob/main/files/lib/pbox/items/detector.py#L10) is the base for abstracted tools loaded from the [`detectors.yml`](https://github.com/dhondta/docker-packing-box/tree/main/detectors.yml) description file. It holds the registry of all the loaded child classes.

```session
>>> Detector.registry
[<pbox.items.detector.Bintropy object at 0x7f30e37f37f0>, [...]
```

**Special methods**:

- `check(formats)`: for checking if the detector applies for the input executable formats
- `detect(executable_or_folder_or_dataset)`: for detecting the packer used on an input executable, folder of executables or [`Dataset`](datasets.md) structure
- `test(executable_or_folder_or_dataset)`: for testing the detector(s) on an input executable, folder of executables or [`Dataset`](datasets.md) structure

!!! note "Using as a class or an instance"
    
    The behavior of the detection method is different depending on the object it is called from. If calling it from:
    
    - The `Detector` class: all the available detectors in `Detector.registry` with the attribute `vote=True` are used and the label is determined based on a decision heuristic.
    - A `Detector` instance: the particular detector (e.g. `DIE`) inheriting `Detector` is used.


!!! note "Multiple valid input types"
    
    These functions are decorated with a special function that allows to input either a single executable, a folder of executables or a dataset containing a "`files`" folder with executables.

