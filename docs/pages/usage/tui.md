# Terminal Interface

`pbox-tui` is a terminal user interface for browsing and managing the abstractions of the current workspace: the default one (`~/.packing-box`) or, when an experiment is open, the experiment.

```console
┌──[user@packing-box]──[/mnt/share]────────
$ pbox-tui
```

An optional argument selects the category shown first, e.g. `pbox-tui packers` or `pbox-tui datasets`.

## Layout

- **Sidebar**: the categories, with their number of entries.
    - *Items*: analyzers, detectors, packers and unpackers.
    - *Definitions*: algorithms, alterations, features and scenarios.
    - *Workspace*: datasets, models, experiments and the configuration.
- **Table**: the entries of the selected category. Type `/` to filter them. Every word must match the name, the description, the status or the formats.
- **Details**: the selected entry. For items and definitions, this includes its YAML definition, read from the file in use (the experiment's one if it has its own).

The header shows the workspace in use, and the status line shows the definition file (or folder) of the category.

## Actions

The footer only shows the keys that apply to the selected category.

| Key | Applies to | Action |
| --- | --- | --- |
| `/` | all | Filter the table |
| `a` | items, alterations, features | Show all entries or only the enabled ones (status `ok`, `keep` or `apply` set) |
| `e` | items, definitions | Edit the definition in `$VISUAL`/`$EDITOR` (`vim` by default), at the line of the entry |
| `e` | experiments | Edit the README of the experiment |
| `e` | configuration | Set the value of the option (`packing-box config --option value`) |
| `s` | items | Set the status of the item (`ok`, `gui`, `test`, `todo`, `info`, `useless`, `broken`, `commercial`), or unset it so that the status is derived from the installation |
| `space` | alterations, features | Toggle `apply` (alterations) or `keep` (features) |
| `i` / `t` | items | Set up (`packing-box -v setup`, in verbose mode) or test (`packing-box test`) the item |
| `v` | datasets, models, experiments | Show an overview (`dataset show`, `model show`, `experiment show`) |
| `n` | datasets, models | Rename (`dataset rename`, `model rename`) |
| `Del` | datasets, models, experiments | Purge, after confirmation (`dataset purge`, `model purge`, `experiment purge`) |
| `o` | experiments | Open the experiment (closing the current one first) |
| `c` | all, with an open experiment | Close the experiment |
| `!` | all | Run any command of the toolkit, e.g. `dataset make ...` or `model train ...` |
| `Ctrl+R` | all | Reload the workspace |
| `q` | all | Quit |

Actions that rely on a tool of the toolkit (setup, test, show, rename, purge, open, close and `!`) suspend the interface and run the tool in the terminal, so that its output and prompts stay visible. Press Enter to come back. The workspace is then reloaded.

!!! note "Changes within an experiment"
    When an experiment is open, the definitions it does not have yet come from the default workspace. Before changing such a definition (`e`, `s` or `space`), the interface copies it into the `conf` folder of the experiment, as `experiment edit` does. Changes made in an experiment therefore never affect the default workspace.

The status, `keep` and `apply` flags are changed in place in the YAML files: only the line of the property changes, and the comments and the layout of the files are kept. When a property is missing, it is inserted among the properties of the entry, in alphabetical order.
