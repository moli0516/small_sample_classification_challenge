"""Synthetic guard/aggregation/lifecycle tests; no Kaggle data or native boosters."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import src.experiments.rf_stabilizer as m


class Helper:
    @staticmethod
    def _sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _write(path, value):
        Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture
def source():
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(250, 150)))
    y = pd.Series(np.tile([0, 1], 125))
    ids = pd.Series([f"{i:04d}" for i in range(250)], dtype="string")
    splits, rows, scores = [], [], []
    for repeat in range(10):
        cv = StratifiedKFold(5, shuffle=True, random_state=repeat)
        for fold, (tr, va) in enumerate(cv.split(x, y)):
            key = dict(repeat=repeat, split_seed=repeat, fold=fold)
            splits.append(dict(key, train_indices=tr.tolist(), valid_indices=va.tolist()))
            p = {name: 1 / (1 + np.exp(-x.iloc[va, i].to_numpy())) for i, name in enumerate(("l1", "xgboost", "lightgbm"))}
            p["l1_xgb_lgbm"] = sum(p.values()) / 3
            scores.append(dict(key, **{n: roc_auc_score(y.iloc[va], a) for n, a in p.items()}))
            for j, idx in enumerate(va):
                rows.append(dict(key, row_index=int(idx), id=ids.iloc[idx], target=int(y.iloc[idx]),
                    **{n: a[j] for n, a in p.items()}))
    frame, scores = pd.DataFrame(rows), pd.DataFrame(scores)
    ctx = dict(x=x, y=y, ids=ids, splits=splits, cache=frame.set_index(m.KEYS),
        expected=scores.set_index(["repeat", "fold"]).xgboost)
    return frame, scores, ctx


def test_probability_and_exact_weight():
    assert np.allclose(m.blend(np.array([.2, .8]), np.array([.9, .1])), [.27, .73])
    for bad in ([np.nan], [-.1], [1.1], [[.5]]):
        with pytest.raises(ValueError):
            m.probability(bad, 1)
    with pytest.raises(ValueError):
        m.blend(np.array([.1]), np.array([.1, .2]))


def test_config_lock():
    cfg = dict(source_study="a", output_dir="b", expected_trial=14, expected_baseline_auc=.9083364180103309)
    m.validate_config(cfg)
    for key, value in (("expected_trial", 3), ("expected_trial", True), ("expected_baseline_auc", np.nan), ("output_dir", "")):
        with pytest.raises(ValueError):
            m.validate_config(dict(cfg, **{key: value}))


def test_oof_reproduces(source):
    frame, scores, ctx = source
    indexed = m.validate_oof(frame, scores, ctx)
    assert len(indexed) == 2500


@pytest.mark.parametrize("mutation", ["id", "target", "duplicate", "nan", "blend", "key", "auc", "cache"])
def test_oof_rejects_tampering(source, mutation):
    frame, scores, ctx = source
    frame, scores = frame.copy(), scores.copy()
    if mutation == "id": frame.loc[0, "id"] = "wrong"
    if mutation == "target": frame.loc[0, "target"] = 1 - frame.loc[0, "target"]
    if mutation == "duplicate": frame.loc[1, m.KEYS] = frame.loc[0, m.KEYS].to_numpy()
    if mutation == "nan": frame.loc[0, "lightgbm"] = np.nan
    if mutation == "blend": frame.loc[0, "l1_xgb_lgbm"] = .123
    if mutation == "key":
        frame["row_index"] = frame["row_index"].astype(float)
        frame.loc[0, "row_index"] = .5
    if mutation == "auc": scores.loc[0, "lightgbm"] += .1
    if mutation == "cache":
        ctx = dict(ctx, cache=ctx["cache"].copy())
        ctx["cache"].iloc[0, ctx["cache"].columns.get_loc("l1")] = .123
    with pytest.raises((ValueError, KeyError)):
        m.validate_oof(frame, scores, ctx)


def result_table(deltas=None):
    rows = []
    for repeat in range(10):
        base = .8 + .001 * repeat
        for fold in range(5):
            key = dict(repeat=repeat, split_seed=repeat, fold=fold)
            rows.append(dict(key, candidate=m.BASELINE, auc=base))
            for depth, leaf in m.GRID:
                alias = f"d{depth}_leaf{leaf}"
                delta = (deltas or {}).get(alias, [0] * 10)[repeat]
                rows.append(dict(key, candidate="rf_" + alias, auc=.7))
                rows.append(dict(key, candidate="blend_" + alias, auc=base + delta))
    return pd.DataFrame(rows)


def test_exact_tie_order_and_no_improvement():
    _, table, decision = m.summarize(result_table())
    assert decision["best_candidate"] == "blend_d3_leaf5"
    assert decision["selected"] is None
    assert not decision["promote"]
    assert table.loc[table.candidate == "blend_d3_leaf5", "ties"].iloc[0] == 10


def test_winner_fails_gate_no_fallback():
    d = {"d3_leaf5": [.04] * 5 + [-.01] * 5, "d3_leaf10": [.001] * 10}
    _, table, decision = m.summarize(result_table(d))
    assert decision["best_candidate"] == "blend_d3_leaf5"
    assert not decision["promote"]
    assert decision["selected"] is None
    assert bool(table.loc[table.candidate == "blend_d3_leaf10", "passes_auc_gate"].iloc[0])


def test_mean_of_fold_auc_and_stability():
    repeats, table, decision = m.summarize(result_table({"d3_leaf5": [.01] * 10}))
    row = table.set_index("candidate").loc["blend_d3_leaf5"]
    assert decision["selected"] == "blend_d3_leaf5"
    assert row.wins == 10
    assert row.stability_observed
    assert np.isclose(row.mean_difference, .01)
    with pytest.raises(ValueError):
        m.summarize(result_table().iloc[:-1])


def test_real_sklearn_rf_api_smoke():
    x = np.arange(80).reshape(40, 2)
    y = np.tile([0, 1], 20)
    model = RandomForestClassifier(n_estimators=8, max_depth=3, min_samples_leaf=5,
        max_features="sqrt", bootstrap=True, random_state=42, n_jobs=1)
    model.fit(x, y)
    assert model.classes_.tolist() == [0, 1]
    m.probability(model.predict_proba(x)[:, 1], 40)


def test_manifest_checks(tmp_path):
    h = Helper()
    (tmp_path / "data.json").write_text("{}")
    h._write(tmp_path / "run_manifest.json", dict(status="completed", artifact_sha256={"data.json": h._sha(tmp_path / "data.json")}))
    m.verify_artifacts(tmp_path, h, {"data.json"})
    (tmp_path / "data.json").write_text("changed")
    with pytest.raises(ValueError): m.verify_artifacts(tmp_path, h, {"data.json"})
    h._write(tmp_path / "run_manifest.json", dict(status="completed", artifact_sha256={"../escape": "x"}))
    with pytest.raises(ValueError): m.verify_artifacts(tmp_path, h, set())


def prepared(tmp_path, source):
    frame, scores, ctx = source
    sentinel = tmp_path / "input.txt"
    sentinel.write_text("unchanged")
    return dict(helper=Helper(), context=ctx, config={}, output=tmp_path / "run", source=tmp_path,
        oof=m.validate_oof(frame, scores, ctx), scores=scores, versions={},
        hashes={str(sentinel): Helper._sha(sentinel)})


def test_mocked_complete_lifecycle(tmp_path, source, monkeypatch):
    c = prepared(tmp_path, source)
    counts = []
    class FakeRF:
        def __init__(self, **kw):
            assert kw["n_estimators"] == 500 and kw["random_state"] == 42
            self.classes_ = np.array([0, 1])
            self.estimators_ = [SimpleNamespace(tree_=SimpleNamespace(max_depth=3))]
        def fit(self, x, y):
            assert len(x) == 200
            self.train_indices = set(x.index)
            counts.append(1)
        def predict_proba(self, x):
            assert len(x) == 50 and self.train_indices.isdisjoint(x.index)
            p = 1 / (1 + np.exp(-x.iloc[:, 3].to_numpy()))
            return np.column_stack([1-p, p])
    monkeypatch.setattr(m, "RandomForestClassifier", FakeRF)
    runner = m.RFStabilizer()
    monkeypatch.setattr(runner, "_prepare", lambda *a: c)
    dry = runner.run("unused", dry_run=True)
    assert dry["new_forest_fits"] == 200 and not c["output"].exists() and not counts
    runner.run("unused")
    record = m.read(c["output"] / "run_manifest.json")
    assert record["status"] == "completed" and len(counts) == 200
    assert record["completed_forest_fits"] == 200
    assert not record["test_opened"] and not record["kaggle_upload_performed"]
    assert len(pd.read_csv(c["output"] / "comparison.csv")) == 9
    assert len(pd.read_csv(c["output"] / "validation_probabilities.csv")) == 10000
    for name, digest in record["artifact_sha256"].items():
        assert Helper._sha(c["output"] / name) == digest
    with pytest.raises(FileExistsError): runner.run("unused")


def test_failure_and_warning_recorded(tmp_path, source, monkeypatch):
    c = prepared(tmp_path, source)
    class WarnRF:
        def __init__(self, **kw): pass
        def fit(self, x, y):
            import warnings
            warnings.warn("review me", UserWarning)
            raise RuntimeError("fit failed")
    runner = m.RFStabilizer()
    monkeypatch.setattr(runner, "_prepare", lambda *a: c)
    monkeypatch.setattr(m, "RandomForestClassifier", WarnRF)
    with pytest.raises(RuntimeError): runner.run("unused")
    record = m.read(c["output"] / "run_manifest.json")
    assert record["status"] == "failed_or_interrupted"
    assert record["completed_forest_fits"] == 0 and record["attempted_forest_fits"] == 1
    assert m.read(c["output"] / "warnings.json")[0]["message"] == "review me"
    assert not (c["output"] / "decision.json").exists()


def test_changed_inputs_rejected(tmp_path, source, monkeypatch):
    c = prepared(tmp_path, source)
    (tmp_path / "input.txt").write_text("changed")
    runner = m.RFStabilizer()
    monkeypatch.setattr(runner, "_prepare", lambda *a: c)
    with pytest.raises(RuntimeError): runner.run("unused", dry_run=True)
    assert not c["output"].exists()


def test_invalid_auc_or_unpaired_folds_rejected():
    frame = result_table()
    frame.loc[0, "auc"] = np.nan
    with pytest.raises(ValueError): m.summarize(frame)
    frame = result_table()
    frame.loc[frame.candidate == "blend_d3_leaf5", "fold"] += 20
    with pytest.raises(ValueError): m.summarize(frame)
