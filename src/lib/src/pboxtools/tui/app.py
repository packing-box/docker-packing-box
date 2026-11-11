# -*- coding: UTF-8 -*-
""" Textual application for managing the abstractions of the current workspace (the default one, ~/.packing-box, or
     the currently open experiment).

Reading goes through a snapshot computed by pbox in a subprocess (see snapshot.py). Changes go either through the
 command-line tools of the Packing Box (packing-box, dataset, model, experiment), run in the terminal with the TUI
 suspended so that their output and prompts stay visible, or, for the simple flags of the YAML definitions (status,
 keep, apply), through line-based edits that keep the comments and layout of the files (see yamledit.py).
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.cells import cell_len
from rich.markdown import Markdown
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from . import yamledit


__all__ = ["PackingBoxApp"]

# category id -> (label, group) ; the order is the one of the sidebar
CATEGORIES = {
    'analyzers':   ("Analyzers", "Items"),
    'detectors':   ("Detectors", "Items"),
    'packers':     ("Packers", "Items"),
    'unpackers':   ("Unpackers", "Items"),
    'algorithms':  ("Algorithms", "Definitions"),
    'alterations': ("Alterations", "Definitions"),
    'features':    ("Features", "Definitions"),
    'scenarios':   ("Scenarios", "Definitions"),
    'datasets':    ("Datasets", "Workspace"),
    'models':      ("Models", "Workspace"),
    'experiments': ("Experiments", "Workspace"),
    'config':      ("Configuration", "Workspace"),
}
ITEMS = ("analyzers", "detectors", "packers", "unpackers")
DEFINITIONS = ("algorithms", "alterations", "features", "scenarios")
FLAGS = {'alterations': "apply", 'features': "keep"}
# statuses that can be set in a definition ; 'installed' and 'not installed' are derived from the system when no
#  status is set (see pbox.core.items.__common__.Base.status)
STATUSES = ["ok", "gui", "test", "todo", "info", "useless", "broken", "commercial"]
STATUS_STYLES = {
    'ok': "green", 'installed': "dark_orange", 'not installed': "red", 'broken': "magenta", 'gui': "cyan",
    'commercial': "yellow", 'info': "grey62", 'test': "grey62", 'todo': "grey62", 'useless': "grey62",
}
UNSET = "(unset: derive from installation)"


def _flat(text):
    return " ".join(str(text or "").split())


def _short(text, n=70):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def _status(s):
    return Text(s, style=STATUS_STYLES.get(s, ""))


def _check(flag):
    return Text("●", style="green") if flag else Text("○", style="grey50")


def _counts(counts):
    return ", ".join(f"{k}: {v}" for k, v in sorted((counts or {}).items(), key=lambda x: -x[1]))


def _yaml(v):
    import yaml
    return yaml.safe_dump(v, default_flow_style=False, sort_keys=False, allow_unicode=True).rstrip()


# ------------------------------------------------------------------------------------------------------------- MODALS
class ConfirmScreen(ModalScreen):
    """ Yes/No confirmation ; dismisses with a boolean. """
    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel")]

    def __init__(self, message, yes="Yes", danger=False):
        super().__init__()
        self.message, self.yes, self.danger = message, yes, danger

    def compose(self):
        with Vertical(classes="dialog"):
            yield Static(self.message, classes="message")
            with Horizontal(classes="buttons"):
                yield Button(self.yes, variant="error" if self.danger else "primary", id="yes")
                yield Button("Cancel", id="no")

    def on_mount(self):
        self.query_one("#no" if self.danger else "#yes").focus()

    @on(Button.Pressed)
    def _pressed(self, event):
        self.dismiss(event.button.id == "yes")


class InputScreen(ModalScreen):
    """ Single-line input ; dismisses with the entered string, or None if cancelled. """
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(self, title, value="", placeholder=""):
        super().__init__()
        self.title_, self.value, self.placeholder = title, value, placeholder

    def compose(self):
        with Vertical(classes="dialog"):
            yield Label(self.title_, classes="message")
            yield Input(value=self.value, placeholder=self.placeholder)
            yield Static("Enter to confirm, Escape to cancel", classes="hint")

    @on(Input.Submitted)
    def _submitted(self, event):
        self.dismiss(event.value)


class ChoiceScreen(ModalScreen):
    """ Choice among options ; dismisses with the chosen option, or None if cancelled. """
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(self, title, choices, current=None):
        super().__init__()
        self.title_, self.choices, self.current = title, choices, current

    def compose(self):
        with Vertical(classes="dialog"):
            yield Label(self.title_, classes="message")
            yield OptionList(*[Option(c + ("  (current)" if c == self.current else ""), id=c) for c in self.choices])

    def on_mount(self):
        ol = self.query_one(OptionList)
        if self.current in self.choices:
            ol.highlighted = self.choices.index(self.current)
        ol.focus()

    @on(OptionList.OptionSelected)
    def _selected(self, event):
        self.dismiss(event.option.id)


class ItemsTable(DataTable):
    """ Table of the entries of a category ; its columns are fitted again to its width when it is resized. """
    _last_width = 0
    
    def on_resize(self, event):
        if event.size.width != self._last_width:
            self._last_width = event.size.width
            if getattr(self.app, "snap", None) is not None:
                self.app.call_after_refresh(self.app.populate)


# ---------------------------------------------------------------------------------------------------------------- APP
class PackingBoxApp(App):
    """ Manage the abstractions (items, definitions, datasets, models, experiments, configuration) of the current
         workspace of the Packing Box. """
    TITLE = "Packing Box"
    CSS = """
    #sidebar { width: 26; border-right: solid $panel; }
    #sidebar OptionList { height: 1fr; border: none; }
    #main { width: 1fr; }
    #filter { margin: 0 1; }
    #table { height: 1fr; }
    #details { height: 45%; border-top: solid $panel; padding: 0 1; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    ModalScreen { align: center middle; }
    .dialog { width: 70; height: auto; max-height: 80%; padding: 1 2; border: thick $primary; background: $surface; }
    .dialog OptionList { height: auto; max-height: 16; }
    .message { width: 100%; margin-bottom: 1; }
    .hint { color: $text-muted; margin-top: 1; }
    .buttons { height: auto; align-horizontal: right; }
    .buttons Button { margin-left: 2; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "filter", "Filter"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("a", "toggle_all", "All/enabled"),
        Binding("e", "edit", "Edit"),
        Binding("s", "set_status", "Status"),
        Binding("space", "toggle_flag", "Toggle"),
        Binding("i", "setup", "Setup"),
        Binding("t", "test", "Test"),
        Binding("v", "show", "Show"),
        Binding("n", "rename", "Rename"),
        Binding("delete", "purge", "Purge"),
        Binding("o", "open_experiment", "Open"),
        Binding("c", "close_experiment", "Close experiment"),
        Binding("exclamation_mark", "command", "Command"),
    ]

    def __init__(self, category="detectors", snapshot=None, **kw):
        super().__init__(**kw)
        self.snap = snapshot
        self.category = category if category in CATEGORIES else "detectors"
        self.enabled_only = False
        self._selected = {}    # category -> selected row key, kept across refreshes

    # ------------------------------------------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield OptionList(id="categories")
            with Vertical(id="main"):
                yield Input(placeholder="Filter (/)", id="filter")
                yield ItemsTable(id="table", cursor_type="row", zebra_stripes=True)
                yield Static(id="status")
                with VerticalScroll(id="details"):
                    yield Static(id="detail")
        yield Footer()

    def on_mount(self):
        self.query_one("#table").focus()
        if self.snap is None:
            self.action_refresh()
        else:
            self.apply_snapshot(self.snap)

    # ------------------------------------------------------------------------------------------------------ data
    @property
    def cat(self):
        return (self.snap or {}).get('categories', {}).get(self.category, {})

    @property
    def experiment(self):
        return (self.snap or {}).get('experiment')

    def rows(self):
        """ Rows of the current category after filtering. """
        rows = self.cat.get('rows', [])
        if self.enabled_only:
            if self.category in ITEMS:
                rows = [r for r in rows if r.get('enabled')]
            elif self.category in FLAGS:
                rows = [r for r in rows if r.get(FLAGS[self.category])]
        q = self.query_one("#filter", Input).value.strip().lower()
        if q:
            fields = ("name", "description", "status", "formats", "category", "section", "algorithm", "dataset")
            rows = [r for r in rows if all(any(w in str(r.get(f, "")).lower() for f in fields) for w in q.split())]
        return rows

    def selected(self):
        """ Currently highlighted row (dict), or None. """
        t = self.query_one("#table", DataTable)
        if t.row_count == 0 or t.cursor_row is None or t.cursor_row < 0:
            return
        try:
            key = t.coordinate_to_cell_key((t.cursor_row, 0)).row_key.value
        except Exception:
            return
        for r in self.cat.get('rows', []):
            if r['name'] == key:
                return r

    @work(thread=True, exclusive=True, group="snapshot")
    def action_refresh(self):
        """ Recompute the snapshot of the workspace in a subprocess (pbox loads its state once per process). """
        self.call_from_thread(self._loading, True)
        with tempfile.TemporaryDirectory(prefix="pbox-tui-") as d:
            out = Path(d, "snapshot.json")
            p = subprocess.run([sys.executable, "-m", "pboxtools.tui.snapshot", str(out)], capture_output=True,
                               text=True, stdin=subprocess.DEVNULL)
            if p.returncode != 0 or not out.is_file():
                err = (p.stderr or p.stdout or "unknown error").strip().splitlines()[-5:]
                self.call_from_thread(self._loading, False)
                self.call_from_thread(self.notify, "\n".join(err), title="Cannot load the workspace",
                                      severity="error", timeout=15)
                return
            snap = json.loads(out.read_text())
        self.call_from_thread(self.apply_snapshot, snap)

    def _loading(self, flag):
        self.query_one("#table").loading = flag

    def apply_snapshot(self, snap):
        self.snap = snap
        self._loading(False)
        xp = snap.get('experiment')
        self.sub_title = f"experiment {Path(xp).name} ({xp})" if xp else f"workspace {snap.get('workspace')}"
        ol = self.query_one("#categories", OptionList)
        ol.clear_options()
        group, idx, highlight = None, 0, 0
        for cid, (label, grp) in CATEGORIES.items():
            if grp != group:
                ol.add_option(Option(Text(grp.upper(), style="bold"), disabled=True))
                group, idx = grp, idx + 1
            n = len(snap['categories'].get(cid, {}).get('rows', []))
            err = " ⚠" if cid in snap.get('errors', {}) else ""
            ol.add_option(Option(f"  {label} ({n}){err}", id=cid))
            if cid == self.category:
                highlight = idx
            idx += 1
        ol.highlighted = highlight
        for cid, err in snap.get('errors', {}).items():
            self.notify(err, title=f"{CATEGORIES.get(cid, (cid,))[0]}: loading error", severity="warning", timeout=10)
        self.populate()

    # ------------------------------------------------------------------------------------------------------ table
    def _columns_and_cells(self, r):
        c = self.category
        if c in ITEMS:
            name = r['name'] + (f"  ↳ {r['parent']}" if r.get('parent') else "")
            return ("Name", "Status", "Formats", "Description"), \
                   (name, _status(r['status']), r['formats'], _flat(r['description']))
        if c in FLAGS:
            flag = FLAGS[c]
            return ("Name", flag.capitalize(), "Formats", "Description"), \
                   (r['name'], _check(r[flag]), _short(r['formats'], 30), _flat(r['description']))
        if c == "algorithms":
            return ("Name", "Category", "Description"), (r['name'], r['category'], _flat(r['description']))
        if c == "scenarios":
            return ("Name", "Description"), (r['name'], _flat(r['description']))
        if c == "datasets":
            kind = "fileless" if r['fileless'] else "files"
            return ("Name", "Executables", "Formats", "Labels", "Type"), \
                   (r['name'], str(r['executables']), r['formats'], _short(_counts(r['counts']), 40),
                    kind + (", altered" if r['altered'] else ""))
        if c == "models":
            return ("Name", "Algorithm", "Dataset", "Formats"), \
                   (r['name'], r['algorithm'], r['dataset'], r['formats'])
        if c == "experiments":
            return ("", "Name", "Datasets", "Models"), \
                   (Text("●", style="green") if r['current'] else "", r['name'], str(r['datasets']),
                    str(r['models']))
        if c == "config":
            return ("Section", "Option", "Value"), (r['section'], r['name'], _short(r['value'], 50))
        return ("Name", ), (r['name'], )

    def populate(self):
        t = self.query_one("#table", DataTable)
        rows = self.rows()
        t.clear(columns=True)
        cols = self._columns_and_cells({'name': "", 'parent': None, 'status': "", 'formats': "", 'description': "",
                                        'category': "", 'fileless': False, 'counts': {}, 'executables': 0,
                                        'altered': False, 'algorithm': "", 'dataset': "", 'current': False,
                                        'datasets': 0, 'models': 0, 'section': "", 'value': "",
                                        **{f: False for f in FLAGS.values()}})[0]
        cells = [self._columns_and_cells(r)[1] for r in rows]
        # the Description column takes the width left by the other columns, so that the table spans the whole pane
        stretch = cols.index("Description") if "Description" in cols else None
        widths = [max([cell_len(c)] +
                      [(c2.cell_len if isinstance(c2, Text) else cell_len(str(c2))) for c2 in (row[i] for row in cells)])
                  for i, c in enumerate(cols)]
        if stretch is not None and t.size.width == 0 and not getattr(self, "_refit", False):
            # not laid out yet (first display) ; fill it again once its width is known
            self._refit = True
            self.call_after_refresh(self._refit_columns)
        if stretch is not None and t.size.width > 0:
            pad = 2 * t.cell_padding
            scrollbar = t.styles.scrollbar_size_vertical if len(rows) > t.size.height - 1 else 0
            others = sum(w + pad for i, w in enumerate(widths) if i != stretch)
            widths[stretch] = max(10, t.size.width - scrollbar - others - pad)
            cells = [row[:stretch] + (_short(row[stretch], widths[stretch]), ) + row[stretch + 1:] for row in cells]
        for i, col in enumerate(cols):
            t.add_column(col, key=f"c{i}", width=widths[i] if i == stretch and t.size.width > 0 else None)
        for r, row in zip(rows, cells):
            t.add_row(*row, key=r['name'])
        # restore the selection
        key = self._selected.get(self.category)
        if key is not None:
            try:
                t.move_cursor(row=t.get_row_index(key))
            except Exception:
                pass
        total = len(self.cat.get('rows', []))
        flt = " (enabled only)" if self.enabled_only and (self.category in ITEMS or self.category in FLAGS) else ""
        where = self.cat.get('file') or self.cat.get('folder') or ""
        self.query_one("#status", Static).update(f"{len(rows)}/{total}{flt}  {where}")
        self.show_details()
        self.refresh_bindings()

    def _refit_columns(self):
        self._refit = False
        if self.query_one("#table", DataTable).size.width > 0:
            self.populate()

    @on(OptionList.OptionHighlighted, "#categories")
    def _category(self, event):
        if event.option.id and event.option.id != self.category:
            self.category = event.option.id
            self.populate()

    @on(OptionList.OptionSelected, "#categories")
    def _category_selected(self, event):
        self.query_one("#table").focus()

    @on(Input.Changed, "#filter")
    def _filter_changed(self, event):
        self.populate()

    @on(Input.Submitted, "#filter")
    def _filter_submitted(self, event):
        self.query_one("#table").focus()

    @on(DataTable.RowHighlighted, "#table")
    def _row(self, event):
        if event.row_key is not None:
            self._selected[self.category] = event.row_key.value
        self.show_details()
        self.refresh_bindings()

    # ------------------------------------------------------------------------------------------------------ details
    def show_details(self):
        r, c = self.selected(), self.category
        if r is None:
            err = (self.snap or {}).get('errors', {}).get(c)
            self.query_one("#detail", Static).update(Markdown(f"**Loading error**\n\n```\n{err}\n```" if err else
                                                              "*Nothing selected*"))
            return
        md = [f"## {r['name']}"]
        if c in ITEMS:
            md.append(f"**Status**: {r['status']}  \n**Formats**: {r['formats']}")
            if r.get('parent'):
                md.append(f"*Variant of* `{r['parent']}`")
            if r.get('description'):
                md.append(r['description'])
            if r.get('comment'):
                md.append(f"> **Note**: {r['comment']}")
            if r.get('source'):
                md.append(f"**Source**: {r['source']}")
            refs = r.get('extra', {}).get('references')
            extra = {k: v for k, v in r.get('extra', {}).items() if k != "references"}
            if extra:
                md.append("\n".join(f"- **{k}**: {', '.join(map(str, v)) if isinstance(v, list) else v}"
                                    for k, v in extra.items()))
            if refs:
                md.append("**References**\n\n" + "\n".join(f"{i}. {ref}" for i, ref in enumerate(refs, 1)))
        elif c in DEFINITIONS:
            if r.get('description'):
                md.append(r['description'])
            if c in FLAGS:
                md.append(f"**{FLAGS[c].capitalize()}**: {'yes' if r[FLAGS[c]] else 'no'}  \n"
                          f"**Formats**: {r['formats']}")
            if c == "algorithms" and r.get('category'):
                md.append(f"**Category**: {r['category']}")
        elif c == "datasets":
            md.append(f"**Executables**: {r['executables']}  \n**Formats**: {r['formats']}  \n"
                      f"**Labels**: {_counts(r['counts']) or '-'}  \n"
                      f"**Type**: {'fileless' if r['fileless'] else 'with files'}"
                      f"{' (altered)' if r['altered'] else ''}  \n**Path**: `{r['path']}`")
            if r.get('sources'):
                md.append("**Sources**\n\n" + "\n".join(f"- `{s}`" for s in r['sources']))
        elif c == "models":
            md.append(f"**Algorithm**: {r['algorithm']}  \n**Dataset**: {r['dataset']} ({r['executables']} "
                      f"executables, {r['formats']})  \n**Path**: `{r['path']}`")
            params = r.get('metadata', {}).get('algorithm', {}).get('parameters')
            if params:
                md.append("**Parameters**\n\n```yaml\n" + _yaml(params) + "\n```")
            if len(r.get('performance', [])) > 1:
                md.append("**Performance**\n\n```\n" + "\n".join(r['performance']) + "\n```")
        elif c == "experiments":
            md.append(("**Currently open**  \n" if r['current'] else "") +
                      f"**Datasets**: {r['datasets']}  \n**Models**: {r['models']}  \n**Path**: `{r['path']}`")
            if r.get('conf'):
                md.append("**Configurations**: " + ", ".join(f"`{x}`" for x in r['conf']))
            if r.get('readme', "").strip():
                md.append("---\n\n" + r['readme'])
        elif c == "config":
            md.append(f"**Section**: {r['section']}  \n**Value**: `{r['value']}`  \n"
                      f"**Command**: `packing-box config {r['option']} {r['metavar']}`")
            if r.get('description'):
                md.append(r['description'])
        # YAML definition, read from the file in use (so that it reflects the last edits)
        if c in ITEMS or c in DEFINITIONS:
            key = r.get('parent') or r['key']
            block = None
            try:
                block = yamledit.get_block(self.cat['file'], key)
            except Exception:
                pass
            if block:
                md.append(f"**Definition** (`{yamledit.find_entry(self.cat['file'], key)}`)\n\n```yaml\n{block}\n```")
        self.query_one("#detail", Static).update(Markdown("\n\n".join(md)))

    # ------------------------------------------------------------------------------------------------------ actions
    def check_action(self, action, parameters):
        c, r = self.category, None
        try:
            r = self.selected()
        except Exception:
            pass
        rules = {
            'toggle_all': c in ITEMS or c in FLAGS,
            'edit': (c in ITEMS or c in DEFINITIONS or c in ("config", "experiments")) and r is not None,
            'set_status': c in ITEMS and r is not None,
            'toggle_flag': c in FLAGS and r is not None,
            'setup': c in ITEMS and r is not None,
            'test': c in ITEMS and r is not None,
            'show': c in ("datasets", "models", "experiments") and r is not None,
            'rename': c in ("datasets", "models") and r is not None,
            'purge': c in ("datasets", "models", "experiments") and r is not None,
            'open_experiment': c == "experiments" and r is not None and not r['current'],
            'close_experiment': self.experiment is not None,
        }
        return rules.get(action, True)

    def action_filter(self):
        self.query_one("#filter", Input).focus()

    def action_toggle_all(self):
        self.enabled_only = not self.enabled_only
        self.populate()

    def run_in_terminal(self, *commands, refresh=True):
        """ Suspend the TUI and run the given commands (lists of arguments) in the terminal, stopping at the first
             failure, then wait for Enter so that the output can be read. """
        rc = 0
        with self.suspend():
            print()
            for cmd in commands:
                print(f"\033[1m$ {shlex.join(cmd)}\033[0m", flush=True)
                try:
                    rc = subprocess.call(cmd)
                except FileNotFoundError:
                    print(f"command not found: {cmd[0]}")
                    rc = 127
                if rc != 0:
                    break
            try:
                input(f"\n[exit code {rc}] Press Enter to return to the TUI...")
            except EOFError:
                pass
        if refresh:
            self.action_refresh()
        return rc

    def writable_definition(self):
        """ Path of the definition file (or folder) of the current category, for writing. When an experiment is open
             and the definition still comes from the default workspace, it is first copied into the experiment (as
             'experiment edit' does), so that changes stay within the experiment. """
        src = Path(self.cat['file'])
        xp = self.experiment
        if xp is None or src.is_relative_to(Path(xp)):
            return src
        dst = Path(xp, "conf", src.name)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
            self.notify(f"Copied {src} into the experiment", title="Definition imported")
            self.cat['file'] = str(dst)
        return dst

    def _line_of(self, path, key):
        """ (file, line) of the entry 'key' among the definition files at path. """
        f = yamledit.find_entry(path, key)
        if f is None:
            return (yamledit.definition_files(path) or [Path(path)])[0], 1
        for i, l in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if yamledit._TOP.match(l) and yamledit._TOP.match(l).group(1).strip("'\"") == key:
                return f, i
        return f, 1

    def action_edit(self):
        r, c = self.selected(), self.category
        if r is None:
            return
        if c == "config":
            def _set(value):
                if value is None or value == r['value']:
                    return
                p = subprocess.run(["packing-box", "config", r['option'], value], capture_output=True, text=True,
                                   stdin=subprocess.DEVNULL)
                if p.returncode == 0:
                    self.notify(f"{r['name']} = {value}", title="Configuration updated")
                else:
                    self.notify((p.stderr or p.stdout).strip()[-500:], title="Configuration not updated",
                                severity="error", timeout=10)
                self.action_refresh()
            self.push_screen(InputScreen(f"{r['name']} ({r['metavar']}): {r['description']}", str(r['value'])), _set)
            return
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vim"
        if c == "experiments":
            readme = Path(r['path'], "README.md")
            self.run_in_terminal(shlex.split(editor) + [str(readme)])
            return
        f, line = self._line_of(self.writable_definition(), r.get('parent') or r['key'])
        self.run_in_terminal(shlex.split(editor) + [f"+{line}", str(f)])

    def _set_property(self, prop, value, remove=False):
        r = self.selected()
        key = r['key']
        try:
            path = self.writable_definition()
            if remove:
                yamledit.remove_property(path, key, prop)
            else:
                yamledit.set_property(path, key, prop, value)
        except (KeyError, ValueError, OSError) as e:
            self.notify(str(e), title="Not changed", severity="error")
            return False
        return True

    def action_set_status(self):
        r = self.selected()
        if r is None:
            return
        if r.get('parent'):
            self.notify(f"{r['name']} is a variant: it inherits the status of {r['parent']}", severity="warning")
            return
        def _set(status):
            if status is None:
                return
            if self._set_property("status", None, remove=True) if status == UNSET else \
               self._set_property("status", status):
                self.notify(f"{r['name']}: status {status}", title="Status updated")
                self.action_refresh()
        self.push_screen(ChoiceScreen(f"Status of {r['name']}", STATUSES + [UNSET], r['status']), _set)

    def action_toggle_flag(self):
        r, flag = self.selected(), FLAGS.get(self.category)
        if r is None or flag is None:
            return
        value = not r[flag]
        if self._set_property(flag, value):
            # update in place for a snappy feedback, then reload what pbox computes from the definitions
            r[flag] = value
            self.populate()
            self.notify(f"{r['name']}: {flag} = {str(value).lower()}")
            self.action_refresh()

    def action_setup(self):
        r = self.selected()
        if r is not None:
            self.run_in_terminal(["packing-box", "-v", "setup", self.cat['type'], r['name']])

    def action_test(self):
        r = self.selected()
        if r is not None:
            self.run_in_terminal(["packing-box", "test", self.cat['type'], r['name']])

    def action_show(self):
        r, c = self.selected(), self.category
        if r is None:
            return
        if c == "experiments":
            if not r['current']:
                self.notify("Only the open experiment can be shown ; open it first (o)", severity="warning")
                return
            self.run_in_terminal(["experiment", "show"], refresh=False)
        else:
            self.run_in_terminal([c[:-1], "show", r['name']], refresh=False)

    def action_rename(self):
        r, c = self.selected(), self.category
        if r is None:
            return
        def _rename(name):
            if name and name.strip() and name.strip() != r['name']:
                self._selected[c] = name.strip()
                self.run_in_terminal([c[:-1], "rename", r['name'], name.strip()])
        self.push_screen(InputScreen(f"New name for {c[:-1]} '{r['name']}'", r['name']), _rename)

    def action_purge(self):
        r, c = self.selected(), self.category
        if r is None:
            return
        entity = c[:-1]
        if c == "experiments" and self.experiment is not None:
            self.notify("Close the open experiment (c) before purging an experiment", severity="warning")
            return
        def _purge(ok):
            if ok:
                self.run_in_terminal([entity, "purge", r['name']])
        self.push_screen(ConfirmScreen(f"Purge {entity} '{r['name']}' ?\n\nThis deletes {r['path']} and cannot be "
                                       "undone.", yes="Purge", danger=True), _purge)

    def action_open_experiment(self):
        r = self.selected()
        if r is None or r['current']:
            return
        cmds = ([["experiment", "close"]] if self.experiment else []) + [["experiment", "open", r['name']]]
        self.run_in_terminal(*cmds)

    def action_close_experiment(self):
        if self.experiment is not None:
            self.run_in_terminal(["experiment", "close"])

    def action_command(self):
        def _run(cmd):
            if cmd and cmd.strip():
                try:
                    self.run_in_terminal(shlex.split(cmd))
                except ValueError as e:
                    self.notify(str(e), title="Bad command", severity="error")
        self.push_screen(InputScreen("Command to run in the terminal (e.g. dataset make ..., model train ...)",
                                     placeholder="dataset list"), _run)
