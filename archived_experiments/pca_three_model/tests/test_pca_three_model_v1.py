"""Synthetic engine tests; native XGBoost/LightGBM integration remains a local check."""
import importlib.util
from pathlib import Path
import types
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from scripts import run_pca_three_model_v1 as m


def context(tmp_path, n=80, p=60):
    x, y = make_classification(n_samples=n, n_features=p, n_informative=5, random_state=6)
    x = pd.DataFrame(x, columns=[f"x{i}" for i in range(p)])
    y = pd.Series(y)
    ids = pd.Series([f"000{i}" for i in range(n)], dtype="string")
    splits, baseline, prediction_rows = [], [], []
    rng = np.random.default_rng(42)
    for r in range(10):
        for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(x, y)):
            splits.append(dict(scheme="cv5", repeat=r, split_seed=r, fold=f, train_indices=tr.tolist(), valid_indices=va.tolist()))
            probs = {name: rng.uniform(.1, .9, len(va)) for name in m.ALIASES}
            probs["ensemble"] = sum(probs.values()) / 3
            for name, prob in probs.items():
                baseline.append(dict(candidate="original_" + name, model=name, repeat=r, split_seed=r, fold=f, components=None, auc=float(roc_auc_score(y.iloc[va], prob))))
            for j, i in enumerate(va):
                prediction_rows.append(dict(scheme="cv5", repeat=r, split_seed=r, fold=f, row_index=int(i), id=str(ids.iloc[i]), target=int(y.iloc[i]), **{m.SOURCE_COLUMNS[name]: float(prob[j]) for name, prob in probs.items()}))
    watched = tmp_path / "watched.txt"
    watched.write_text("unchanged historical source")
    c = dict(context=dict(x=x, y=y, ids=ids), pca_splits=splits, splits=splits,
             config=dict(components=[5, 10, 20, 50]), output=tmp_path / "result", baseline=baseline,
             reproduction=[], hashes={str(watched): m.digest(watched)}, versions={"synthetic": "test"})
    return c, prediction_rows


def surrogate_models(c):
    return {name: LogisticRegression(C=value, max_iter=2000, random_state=42)
            for name, value in zip(m.ALIASES, [.01, .1, 1.])}


def test_full_engine_600_classifier_fits(tmp_path):
    c, _ = context(tmp_path)
    result = m.execute(c, model_factory=surrogate_models)
    assert result["classifier_fits"] == 600 and result["pca_fits"] == 200
    out = c["output"]
    assert len(pd.read_csv(out / "fold_scores.csv")) == 1000
    assert len(pd.read_csv(out / "repeat_scores.csv")) == 200
    assert len(pd.read_csv(out / "model_summary.csv")) == 20
    assert len(pd.read_csv(out / "comparison.csv")) == 16
    assert len(pd.read_csv(out / "validation_probabilities.csv")) == 3200
    assert m.read(out / "decision.json")["automatic_promotion"] is False
    for name, value in m.read(out / "run_manifest.json")["artifact_sha256"].items():
        assert m.digest(out / name) == value
    with pytest.raises(FileExistsError):
        m.execute(c, model_factory=surrogate_models)


def test_training_only_representation(tmp_path):
    c, _ = context(tmp_path)
    s = c["splits"][0]
    x = c["context"]["x"].copy()
    x.iloc[s["valid_indices"]] += 1000
    a, b, scaler, pca = m.representation(x.iloc[s["train_indices"]], x.iloc[s["valid_indices"]], 5)
    assert np.allclose(scaler.mean_, x.iloc[s["train_indices"]].mean())
    assert not np.allclose(scaler.mean_, x.mean())
    assert pca.n_samples_ == len(s["train_indices"])
    assert a.shape[1] == b.shape[1] == 5
    assert not pca.whiten


def test_warning_is_failed_not_selected(tmp_path):
    c, _ = context(tmp_path)
    class Bad(LogisticRegression):
        def fit(self, x, y):
            result = super().fit(x, y)
            warnings.warn("synthetic failure", ConvergenceWarning)
            return result
    def factory(c):
        return {name: Bad(max_iter=1000) for name in m.ALIASES}
    with pytest.raises(RuntimeError, match="ConvergenceWarning"):
        m.execute(c, model_factory=factory)
    assert m.read(c["output"] / "run_manifest.json")["status"] == "failed_or_interrupted"
    assert not (c["output"] / "decision.json").exists()


def test_nonfinite_metadata_recording():
    assert m.safe_metadata({"missing": np.nan, "value": np.int64(4)}) == {"missing": "nan", "value": 4}


def test_partial_results_rejected():
    with pytest.raises((ValueError, KeyError)):
        m.summarize(pd.DataFrame([dict(candidate="pca5_ensemble", repeat=0, fold=0, auc=.8)]), [5, 10, 20, 50])


def test_original_grid_and_probability_guards():
    assert m.validate(dict(validation_config="configs/v.json", output_dir="out", components=[5,10,20,50]))
    with pytest.raises(ValueError):
        m.validate(dict(validation_config="configs/v.json", output_dir="out", components=[50]))
    with pytest.raises(ValueError):
        m.valid_probability(np.array([np.nan]), 1)
    with pytest.raises(ValueError):
        m.valid_probability(np.array([1.1]), 1)


def test_ties_not_wins():
    repeats = pd.DataFrame([dict(candidate=name, repeat=r, mean_fold_auc=.8) for name in ["a", "b"] for r in range(10)])
    _, stat = m.paired(repeats, "b", "a", "test")
    assert stat["ties"] == 10 and stat["wins"] == stat["losses"] == 0


def fixture_preflight(tmp_path, monkeypatch):
    c, predictions = context(tmp_path, 250, 150)
    source = tmp_path / "source"
    source.mkdir()
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    m.write(cfgdir / "validation.json", {"output_dir": "source"})
    m.write(cfgdir / "pca.json", dict(validation_config="configs/validation.json", output_dir="result", components=[5,10,20,50]))
    m.write(source / "split_indices.json", c["splits"])
    pd.DataFrame(predictions).to_csv(source / "validation_probabilities.csv", index=False)
    folds = pd.DataFrame(c["baseline"])
    folds["candidate"] = folds.model.map(m.SOURCE_COLUMNS)
    folds["scheme"] = "cv5"
    folds.to_csv(source / "fold_scores.csv", index=False)
    m.write(source / "settings.json", {})
    m.write(source / "run_manifest.json", dict(status="completed", artifact_sha256={p.name: m.digest(p) for p in source.iterdir()}))
    def verify(root, helper, required):
        manifest = m.read(root / "run_manifest.json")
        assert required <= manifest["artifact_sha256"].keys()
        paths = [root / "run_manifest.json"]
        for name, value in manifest["artifact_sha256"].items():
            path = root / name
            if m.digest(path) != value:
                raise ValueError("changed source artifact")
            paths.append(path)
        return manifest, paths
    stub = types.ModuleType("src.extensions.validation_protocol_v1")
    stub.verify_artifacts = verify
    monkeypatch.setitem(sys.modules, "src.extensions.validation_protocol_v1", stub)
    c["helper"] = object()
    monkeypatch.setattr(m, "historical_context", lambda *args: c)
    return c, source


def test_preflight_rejects_wrong_historical_mean(tmp_path, monkeypatch):
    fixture_preflight(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="ensemble mean failed"):
        m.prepare("configs/pca.json", root=tmp_path)
    assert not (tmp_path / "result").exists()


def test_preflight_rejects_ids(tmp_path, monkeypatch):
    _, source = fixture_preflight(tmp_path, monkeypatch)
    path = source / "validation_probabilities.csv"
    frame = pd.read_csv(path, dtype={"id": "string"}, float_precision="round_trip")
    frame.loc[0, "id"] = "changed"
    frame.to_csv(path, index=False)
    manifest = m.read(source / "run_manifest.json")
    manifest["artifact_sha256"][path.name] = m.digest(path)
    m.write(source / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="identity/labels"):
        m.prepare("configs/pca.json", root=tmp_path)


def test_preflight_rejects_reordered_training_indices(tmp_path, monkeypatch):
    _, source = fixture_preflight(tmp_path, monkeypatch)
    path = source / "split_indices.json"
    splits = m.read(path)
    splits[0]["train_indices"].reverse()
    m.write(path, splits)
    manifest = m.read(source / "run_manifest.json")
    manifest["artifact_sha256"][path.name] = m.digest(path)
    m.write(source / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="ordered cv5"):
        m.prepare("configs/pca.json", root=tmp_path)


def test_hash_change_guard(tmp_path):
    path = tmp_path / "x"
    path.write_text("old")
    expected = {str(path): m.digest(path)}
    path.write_text("new")
    with pytest.raises(RuntimeError):
        m.check_hashes(expected)


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None or importlib.util.find_spec("lightgbm") is None, reason="Native libraries unavailable")
def test_native_classifier_smoke():
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    x, y = make_classification(n_samples=50, n_features=10, n_informative=4, random_state=5)
    a, b, _, _ = m.representation(x[:40], x[40:], 5)
    models = [LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=42),
              XGBClassifier(n_estimators=5, max_depth=2, random_state=42, n_jobs=1),
              LGBMClassifier(n_estimators=5, num_leaves=3, min_child_samples=2, verbosity=-1, n_jobs=1)]
    for model in models:
        model.fit(a, y[:40])
        m.valid_probability(model.predict_proba(b)[:, 1], 10)


def test_factory_preserves_classifier_parameters_without_second_scaler(monkeypatch):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    original = LogisticRegression(penalty="l1", solver="saga", C=1, max_iter=1234, tol=1e-5, random_state=42)
    class Factory:
        def create(self, cfg):
            return Pipeline([("scaler", StandardScaler()), ("classifier", original)])
    module = types.ModuleType("src.modeling.pipeline_factory")
    module.PipelineFactory = Factory
    monkeypatch.setitem(sys.modules, "src.modeling.pipeline_factory", module)
    class Constructor:
        def __init__(self, **kwargs):
            self.parameters = kwargs
    xgb = types.ModuleType("xgboost"); xgb.XGBClassifier = Constructor
    lgb = types.ModuleType("lightgbm"); lgb.LGBMClassifier = Constructor
    monkeypatch.setitem(sys.modules, "xgboost", xgb)
    monkeypatch.setitem(sys.modules, "lightgbm", lgb)
    c = dict(context=dict(linear=object(), candidate=dict(estimator_parameters={"n_estimators":50, "random_state":42})), lightgbm_parameters={"num_leaves":7, "n_estimators":50})
    models = m.build_models(c)
    assert models["l1"].get_params() == original.get_params()
    assert models["l1"] is not original and not isinstance(models["l1"], Pipeline)
    assert models["xgboost"].parameters == c["context"]["candidate"]["estimator_parameters"]
    assert models["lightgbm"].parameters == c["lightgbm_parameters"]


def test_dry_run_no_fits_or_output(tmp_path, monkeypatch):
    c, _ = context(tmp_path)
    monkeypatch.setattr(m, "prepare", lambda *args: c)
    monkeypatch.setattr(m, "build_models", surrogate_models)
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected fit")
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(m.PCA, "fit_transform", forbidden)
    assert m.run("unused", dry_run=True)["fits_performed"] == 0
    assert not c["output"].exists()
