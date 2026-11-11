# -*- coding: UTF-8 -*-
""" Tests for the export of trained models as standalone tools (pbox.core.model.export). They need pbox (and its
     dependencies) and use the PE samples of the repository as a small dataset. """
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1].joinpath("src")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

pbox = pytest.importorskip("pbox")
pytest.importorskip("sklearn")

FEATURES = ["entropy", "number_standard_sections", "average_256B_block_entropy", "has_tls_section",
            "highest_per_section_256B_block_entropy", "is_ep_in_w_section", "has_known_packer_section_names",
            "is_import_functions_count<=2"]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """ A dataset (with files) of the repository's PE samples and their features, computed by pbox with the portable
         parser ; models are trained on it by the tests. """
    import pandas as pd
    from pbox import Executable, Features
    root = tmp_path_factory.mktemp("export")
    ds = root.joinpath("datasets", "samples")
    ds.joinpath("files").mkdir(parents=True)
    previous = config['pe_parser']
    config['pe_parser'] = "portable"
    rows = []
    try:
        Features()
        names = [n for n in FEATURES if n in Features.registry.get('PE32', {})]
        for f in sorted(REPO.joinpath("tests", "samples").rglob("*.exe")):
            e = Executable(str(f))
            if e.format not in ("PE32", "PE64", ".NET"):
                continue
            feats = Features(e, feature_names=names)
            shutil.copy(f, ds.joinpath("files", f.name))
            rows.append({'hash': f.name, 'format': e.format, **{n: feats.get(n) for n in names},
                         'label': "-" if "not-packed" in f.parts else f.parent.name})
    finally:
        config['pe_parser'] = previous
    if len(rows) < 4:
        pytest.skip("not enough PE samples")
    df = pd.DataFrame(rows)
    df.to_csv(ds.joinpath("data.csv"), sep=";", index=False)
    return root, ds, df, sorted(names)


def _model(root, name, pipeline, df, names, formats):
    import joblib
    from pbox import Features
    m = root.joinpath("models", name)
    m.mkdir(parents=True)
    X, y = df[names].fillna(-1), (df['label'] != "-").astype(int)
    pipeline.fit(X, y)
    joblib.dump(pipeline, m.joinpath("dump.joblib"))
    m.joinpath("features.json").write_text(json.dumps({n: Features.registry['PE32'][n].description for n in names}))
    m.joinpath("metadata.json").write_text(json.dumps({
        'algorithm': {'name': name, 'description': name, 'preprocessors': [], 'parameters': {}},
        'dataset': {'name': "samples", 'formats': formats, 'executables': len(df),
                    'counts': df['label'].value_counts().to_dict()}}))
    return m


@pytest.mark.parametrize("algo", ["dt", "rf", "svm", "mlp"])
def test_export_and_validate(workspace, algo, tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from pbox.core.model.export import export_tool
    root, ds, df, names = workspace
    clf = {'dt': DecisionTreeClassifier(random_state=0), 'rf': RandomForestClassifier(10, random_state=0),
           'svm': SVC(probability=True, random_state=0), 'mlp': MLPClassifier((8, ), max_iter=200, random_state=0)}
    pipe = Pipeline([("MinMaxScaler" if algo != "svm" else "StandardScaler",
                      MinMaxScaler() if algo != "svm" else StandardScaler()), (algo, clf[algo])])
    m = _model(tmp_path, f"model_{algo}", pipe, df, names, sorted(df['format'].unique().tolist()))
    out, report = export_tool(m, output=tmp_path.joinpath("tool"), dataset=ds, n_samples=20)
    assert report['model']['rows'] == len(df) and report['model']['mismatches'] == 0
    assert report['files']['checked'] == len(df) and report['files']['mismatches'] == 0
    # only the inference modules of this model are embedded
    runtime = sorted(p.stem for p in out.joinpath("src", "model_" + algo, "portable", "runtime").glob("*.py"))
    expected = {'dt': "tree", 'rf': "tree", 'svm': "svm", 'mlp': "mlp"}[algo]
    assert runtime == sorted(["__init__", "preprocessing", expected])
    # the tests of the generated repository pass, with the samples available
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=out,
                       capture_output=True, text=True, env={'SAMPLES': str(ds.joinpath("files")), 'PATH': "/usr/bin"})
    assert r.returncode == 0, r.stdout[-2000:]
    # the tool, as a .pyz file, runs without site-packages (no dependency)
    pyz = tmp_path.joinpath("tool.pyz")
    subprocess.run([sys.executable, "-m", "zipapp", str(out.joinpath("src")), "-m", f"model_{algo}.__main__:run",
                    "-o", str(pyz)], check=True)
    sample = ds.joinpath("files", df['hash'].iloc[0])
    r = subprocess.run([sys.executable, "-I", "-S", str(pyz), "--json", str(sample)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout)['format'] == df['format'].iloc[0]


def test_unsupported_model(workspace, tmp_path):
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline
    from pbox.core.model.export import export_tool, UnsupportedModelError
    root, ds, df, names = workspace
    m = _model(tmp_path, "model_kmeans", Pipeline([("kmeans", KMeans(2, n_init=1, random_state=0))]), df, names,
               ["PE32"])
    with pytest.raises(UnsupportedModelError, match="KMeans"):
        export_tool(m, output=tmp_path.joinpath("tool"), dataset=ds)


def test_unsupported_formats_and_features(workspace, tmp_path):
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from pbox.core.model.export import export_tool, UnsupportedModelError
    root, ds, df, names = workspace
    m = _model(tmp_path, "model_elf", Pipeline([("dt", DecisionTreeClassifier())]), df, names, ["ELF64", "PE32"])
    with pytest.raises(UnsupportedModelError, match="ELF64"):
        export_tool(m, output=tmp_path.joinpath("tool1"), dataset=ds)
    # a feature that is not defined in the workspace (the model was trained with other definitions)
    df2 = df.assign(no_such_feature=1.)
    names2 = sorted(names + ["no_such_feature"])
    m = tmp_path.joinpath("models", "model_bad")
    _model(tmp_path, "model_bad_base", Pipeline([("dt", DecisionTreeClassifier())]), df, names, ["PE32"])
    shutil.copytree(tmp_path.joinpath("models", "model_bad_base"), m)
    import joblib
    pipe = Pipeline([("dt", DecisionTreeClassifier())]).fit(df2[names2], (df2['label'] != "-").astype(int))
    joblib.dump(pipe, m.joinpath("dump.joblib"))
    meta = json.loads(m.joinpath("features.json").read_text())
    m.joinpath("features.json").write_text(json.dumps({**meta, 'no_such_feature': "?"}))
    with pytest.raises(UnsupportedModelError, match="no_such_feature"):
        export_tool(m, output=tmp_path.joinpath("tool2"), dataset=ds)


def test_existing_output_refused(workspace, tmp_path):
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from pbox.core.model.export import export_tool
    root, ds, df, names = workspace
    m = _model(tmp_path, "model_dt2", Pipeline([("dt", DecisionTreeClassifier())]), df, names, ["PE32"])
    out = tmp_path.joinpath("existing")
    out.mkdir()
    out.joinpath("keep.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError):
        export_tool(m, output=out, dataset=ds)
    assert out.joinpath("keep.txt").read_text() == "do not overwrite"
