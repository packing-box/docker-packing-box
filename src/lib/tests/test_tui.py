# -*- coding: UTF-8 -*-
""" Tests for the TUI (pboxtools.tui): line-based YAML editing and the Textual application, driven with a static
     snapshot (the snapshot itself needs pbox and a workspace ; it is smoke-tested only when pbox is importable). """
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1].joinpath("src")))

from pboxtools.tui import yamledit  # noqa: E402


DETECTORS = """\
defaults:
  maintained: true
  vote:       false


# a comment introducing the detectors
Alpha:
  description: First detector  # inline comment
  formats:
    - PE
  source: https://example.com/alpha
  status: ok

Beta:
  comment: 'quoted: value'
  formats:
    - ELF
  vote: true

# trailing comment of Beta, introducing Gamma
Gamma:
  description: Third
"""


@pytest.fixture
def defs(tmp_path):
    p = tmp_path / "detectors.yml"
    p.write_text(DETECTORS)
    return p


def test_get_block(defs):
    assert yamledit.get_block(defs, "Beta") == "Beta:\n  comment: 'quoted: value'\n  formats:\n    - ELF\n  vote: true"
    assert yamledit.get_block(defs, "Gamma") == "Gamma:\n  description: Third"
    assert yamledit.get_block(defs, "Missing") is None


def test_set_existing_property_keeps_the_rest(defs):
    yamledit.set_property(defs, "Alpha", "status", "broken")
    text = defs.read_text()
    assert "  status: broken\n" in text and "  status: ok" not in text
    # everything else is untouched, comments included
    assert text.replace("status: broken", "status: ok") == DETECTORS
    assert yaml.safe_load(text)['Alpha']['status'] == "broken"


def test_set_property_keeps_inline_comment(defs):
    yamledit.set_property(defs, "Alpha", "description", "Renamed")
    assert "  description: Renamed  # inline comment\n" in defs.read_text()


def test_insert_property_sorted(defs):
    yamledit.set_property(defs, "Beta", "status", "test")
    lines = yamledit.get_block(defs, "Beta").splitlines()
    assert lines == ["Beta:", "  comment: 'quoted: value'", "  formats:", "    - ELF", "  status: test", "  vote: true"]
    yamledit.set_property(defs, "Gamma", "keep", False)
    assert yamledit.get_block(defs, "Gamma") == "Gamma:\n  description: Third\n  keep: false"
    d = yaml.safe_load(defs.read_text())
    assert d['Beta']['status'] == "test" and d['Gamma']['keep'] is False
    # the comment introducing Gamma stays before it
    assert "# trailing comment of Beta, introducing Gamma\nGamma:" in defs.read_text()


def test_insert_at_end_of_file_without_newline(tmp_path):
    p = tmp_path / "x.yml"
    p.write_text("A:\n  description: a")
    yamledit.set_property(p, "A", "keep", True)
    assert yaml.safe_load(p.read_text()) == {'A': {'description': "a", 'keep': True}}


def test_values_are_quoted_when_needed(defs):
    for v in ("a: b", "yes", "", "#x"):
        yamledit.set_property(defs, "Gamma", "comment", v)
        assert yaml.safe_load(defs.read_text())['Gamma']['comment'] == v


def test_errors(defs):
    with pytest.raises(KeyError):
        yamledit.set_property(defs, "Missing", "status", "ok")
    with pytest.raises(ValueError):
        yamledit.set_property(defs, "Alpha", "formats", "PE")


def test_remove_property(defs):
    assert yamledit.remove_property(defs, "Alpha", "status") == defs
    assert "status" not in yaml.safe_load(defs.read_text())['Alpha']
    assert yamledit.remove_property(defs, "Alpha", "status") is None


def test_definitions_folder(tmp_path):
    d = tmp_path / "features.yml"
    (d / "sub").mkdir(parents=True)
    (d / "defaults.yml").write_text("keep: false\n")
    (d / "a.yml").write_text("f1:\n  description: one\n")
    (d / "sub" / "b.yml").write_text("f2:\n  description: two\n")
    assert yamledit.find_entry(d, "f2") == d / "sub" / "b.yml"
    yamledit.set_property(d, "f2", "keep", True)
    assert yaml.safe_load((d / "sub" / "b.yml").read_text()) == {'f2': {'description': "two", 'keep': True}}


# ---------------------------------------------------------------------------------------------------------------- APP
def _snapshot(tmp_path, experiment=None):
    defs = tmp_path / "detectors.yml"
    defs.write_text(DETECTORS)
    feats = tmp_path / "features.yml"
    feats.write_text("entropy:\n  description: Entropy\n  keep: true\n\nsize:\n  description: Size\n")
    det = lambda n, s, **kw: {'name': n.lower(), 'key': n, 'status': s, 'formats': "PE", 'source': "", 'comment': "",
                             'description': f"{n} detector", 'parent': None, 'enabled': s == "ok", 'extra': {}, **kw}
    return {
        'workspace': str(experiment or tmp_path), 'experiment': experiment, 'experiments': str(tmp_path),
        'home': str(tmp_path), 'errors': {},
        'categories': {
            'detectors': {'file': str(defs), 'type': "detector",
                          'rows': [det("Alpha", "ok"), det("Beta", "broken"), det("Gamma", "not installed")]},
            'features': {'file': str(feats), 'flag': "keep", 'rows': [
                {'name': "entropy", 'key': "entropy", 'formats': "PE", 'keep': True, 'description': "Entropy",
                 'result': {}, 'extra': {}},
                {'name': "size", 'key': "size", 'formats': "PE", 'keep': False, 'description': "Size",
                 'result': {}, 'extra': {}}]},
            'datasets': {'folder': str(tmp_path), 'rows': [
                {'name': "ds1", 'path': str(tmp_path / "ds1"), 'formats': "PE32", 'executables': 3,
                 'counts': {'-': 2, 'upx': 1}, 'sources': [], 'fileless': False, 'altered': False, 'metadata': {}}]},
            'config': {'rows': [{'name': "exec_timeout", 'key': "exec_timeout", 'section': "main", 'value': "20",
                                 'metavar': "SECONDS", 'description': "timeout", 'option': "--exec-timeout"}]},
        },
    }


def _app(snap, category="detectors"):
    from pboxtools.tui.app import PackingBoxApp
    app = PackingBoxApp(category=category, snapshot=snap)
    app.action_refresh = lambda: None   # no pbox subprocess in tests
    return app


def _run(coro):
    return asyncio.run(coro)


def test_app_lists_and_filters(tmp_path):
    from textual.widgets import DataTable, Input
    async def scenario():
        app = _app(_snapshot(tmp_path))
        async with app.run_test(size=(160, 40)) as pilot:
            t = app.query_one("#table", DataTable)
            assert t.row_count == 3
            assert app.selected()['name'] == "alpha"
            await pilot.press("a")                      # enabled only
            assert t.row_count == 1
            await pilot.press("a")
            app.query_one("#filter", Input).value = "beta"
            await pilot.pause()
            assert t.row_count == 1 and app.selected()['name'] == "beta"
            assert app.check_action("set_status", ()) and not app.check_action("toggle_flag", ())
    _run(scenario())


def test_app_switches_category(tmp_path):
    from textual.widgets import DataTable, OptionList
    async def scenario():
        app = _app(_snapshot(tmp_path))
        async with app.run_test(size=(160, 40)) as pilot:
            ol = app.query_one("#categories", OptionList)
            ol.highlighted = [i for i in range(ol.option_count) if ol.get_option_at_index(i).id == "datasets"][0]
            await pilot.pause()
            assert app.category == "datasets" and app.query_one("#table", DataTable).row_count == 1
            assert app.check_action("rename", ()) and app.check_action("purge", ())
            assert not app.check_action("set_status", ())
    _run(scenario())


def test_app_sets_status(tmp_path):
    async def scenario():
        snap = _snapshot(tmp_path)
        app = _app(snap)
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.press("down")                   # beta
            await pilot.press("s")
            await pilot.pause()
            from pboxtools.tui.app import ChoiceScreen
            assert isinstance(app.screen, ChoiceScreen)
            app.screen.dismiss("ok")
            await pilot.pause()
        return snap
    snap = _run(scenario())
    assert yaml.safe_load(Path(snap['categories']['detectors']['file']).read_text())['Beta']['status'] == "ok"


def test_app_toggles_feature(tmp_path):
    async def scenario():
        snap = _snapshot(tmp_path)
        app = _app(snap, "features")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.press("down")                   # size (keep: false)
            await pilot.press("space")
            await pilot.pause()
            assert app.selected()['keep'] is True
        return snap
    snap = _run(scenario())
    assert yaml.safe_load(Path(snap['categories']['features']['file']).read_text())['size']['keep'] is True


def test_app_copies_definition_into_experiment(tmp_path):
    """ Changes made while an experiment is open never touch the default workspace's definitions. """
    xp = tmp_path / "experiments" / "xp1"
    (xp / "conf").mkdir(parents=True)
    async def scenario():
        snap = _snapshot(tmp_path, experiment=str(xp))
        app = _app(snap)
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.press("s")
            await pilot.pause()
            app.screen.dismiss("todo")
            await pilot.pause()
    _run(scenario())
    assert (tmp_path / "detectors.yml").read_text() == DETECTORS
    assert yaml.safe_load((xp / "conf" / "detectors.yml").read_text())['Alpha']['status'] == "todo"


def test_app_unset_status_and_variant(tmp_path):
    from pboxtools.tui.app import UNSET
    async def scenario():
        snap = _snapshot(tmp_path)
        snap['categories']['detectors']['rows'][2]['parent'] = "Alpha"
        app = _app(snap)
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.press("s")
            await pilot.pause()
            app.screen.dismiss(UNSET)
            await pilot.pause()
            await pilot.press("down", "down", "s")      # variant: refused
            await pilot.pause()
            from pboxtools.tui.app import ChoiceScreen
            assert not isinstance(app.screen, ChoiceScreen)
        return snap
    snap = _run(scenario())
    assert "status" not in yaml.safe_load(Path(snap['categories']['detectors']['file']).read_text())['Alpha']


def test_app_terminal_commands(tmp_path, monkeypatch):
    """ Actions that delegate to the CLI tools build the right command lines. """
    calls = []
    async def scenario():
        app = _app(_snapshot(tmp_path))
        monkeypatch.setattr(app, "run_in_terminal", lambda *cmds, **kw: calls.extend(cmds))
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.press("i", "t")
            from textual.widgets import OptionList
            ol = app.query_one("#categories", OptionList)
            ol.highlighted = [i for i in range(ol.option_count) if ol.get_option_at_index(i).id == "datasets"][0]
            await pilot.pause()
            await pilot.press("v", "delete")
            await pilot.pause()
            app.screen.dismiss(True)
            await pilot.pause()
    _run(scenario())
    assert calls == [["packing-box", "-v", "setup", "detector", "alpha"], ["packing-box", "test", "detector", "alpha"],
                     ["dataset", "show", "ds1"], ["dataset", "purge", "ds1"]]


# ----------------------------------------------------------------------------------------------------------- SNAPSHOT
def test_snapshot_smoke(tmp_path):
    pytest.importorskip("pbox")
    out = tmp_path / "snap.json"
    env_src = str(Path(__file__).resolve().parents[1].joinpath("src"))
    p = subprocess.run([sys.executable, "-m", "pboxtools.tui.snapshot", str(out)], capture_output=True, text=True,
                       env={**__import__("os").environ, 'PYTHONPATH': env_src}, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    snap = json.loads(out.read_text())
    assert snap['errors'] == {}
    for c in ("analyzers", "detectors", "packers", "unpackers", "algorithms", "alterations", "features", "scenarios",
              "datasets", "models", "experiments", "config"):
        assert c in snap['categories']
    assert len(snap['categories']['detectors']['rows']) > 0
