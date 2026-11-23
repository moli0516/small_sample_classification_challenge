"""Complete synthetic artifacts; no competition data needed."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_pca_ensemble_v1 as m


def seal(root):
    m.write(root / "run_manifest.json", dict(status="completed", artifact_sha256={
        p.name: m.digest(p) for p in root.iterdir() if p.name != "run_manifest.json"}))


def fixture_project(tmp_path):
    n = 20
    e, p = tmp_path / "ensemble", tmp_path / "pca"
    e.mkdir(); p.mkdir()
    rng = np.random.default_rng(72)
    splits, rows = [], []
    for r in range(10):
        for f in range(5):
            va = list(range(f * 4, (f + 1) * 4))
            tr = [i for i in range(n) if i not in va]
            splits.append(dict(scheme="cv5", repeat=r, split_seed=r, fold=f, train_indices=tr, valid_indices=va))
            for i in va:
                probs = rng.uniform(.05, .95, 3)
                rows.append(dict(scheme="cv5", repeat=r, fold=f, split_seed=r, row_index=i,
                                 id=f"00{i}", target=i % 2, l1_c1=probs[0], xgboost=probs[1],
                                 lightgbm=probs[2], l1_xgb_lgbm=float(probs.mean())))
    frame = pd.DataFrame(rows)
    frame.to_csv(e / "validation_probabilities.csv", index=False)
    ff, rr, ss = m.score(frame, [m.BASE])
    ff["scheme"] = "cv5"
    ff.to_csv(e / "fold_scores.csv", index=False)
    m.write(e / "split_indices.json", splits)
    m.write(e / "settings.json", {"input_and_code_sha256": {"/original/data/raw/train.csv": "same-original-training-sha"}})
    seal(e)
    watched = m.source(e, {"settings.json"})
    records, fold_tables, repeat_tables, summaries = [], [], [], []
    for name, branch in (("pca_50_C0.01", "pca"), ("no_pca_C0.001", "control")):
        part = frame[["repeat", "fold", "split_seed", "row_index", "id", "target"]].copy()
        part["candidate"] = name
        part["probability"] = rng.uniform(.1, .9, len(part))
        records.append(part)
        f, r, s = m.score(part.rename(columns={"probability": name}), [name])
        s["branch"] = branch
        fold_tables.append(f); repeat_tables.append(r); summaries.append(s)
    pd.concat(records).to_csv(p / "validation_probabilities.csv", index=False)
    pd.concat(fold_tables).to_csv(p / "fold_scores.csv", index=False)
    pd.concat(repeat_tables).to_csv(p / "repeat_scores.csv", index=False)
    pd.concat(summaries).to_csv(p / "model_summary.csv", index=False)
    m.write(p / "split_indices.json", splits)
    m.write(p / "decision.json", dict(selected_pca="pca_50_C0.01", selected_control="no_pca_C0.001"))
    m.write(p / "settings.json", dict(config=dict(source_run="ensemble", train_path="data/raw/train.csv"),
            input_and_code_sha256={**watched, "/original/data/raw/train.csv": "same-original-training-sha"}))
    seal(p)
    cfg = dict(ensemble_run="ensemble", pca_run="pca", output_dir="result",
               pca_candidate="pca_50_C0.01", l2_candidate="no_pca_C0.001", alphas=[.05, .1, .25],
               expected_rows=n, expected_reference_auc=float(ss.mean_auc.iloc[0]))
    m.write(tmp_path / "config.json", cfg)
    return cfg


def test_preflight_does_not_write_or_open_raw_data(tmp_path, monkeypatch):
    fixture_project(tmp_path)
    original = pd.read_csv
    opened = []
    def guarded(path, *args, **kwargs):
        assert Path(path).name not in {"train.csv", "test.csv", "sample_submission.csv"}
        opened.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(m.pd, "read_csv", guarded)
    result = m.run("config.json", dry_run=True, root=tmp_path)
    assert result["fits_performed"] == 0 and result["blends"] == 6
    assert not (tmp_path / "result").exists() and opened


def test_complete_run(tmp_path):
    fixture_project(tmp_path)
    result = m.run("config.json", root=tmp_path)
    assert result["status"] == "completed" and result["fits_performed"] == 0
    out = tmp_path / "result"
    assert len(pd.read_csv(out / "fold_scores.csv")) == 450
    assert len(pd.read_csv(out / "repeat_scores.csv")) == 90
    assert len(pd.read_csv(out / "model_summary.csv")) == 9
    assert len(pd.read_csv(out / "comparison.csv")) == 9
    assert len(pd.read_csv(out / "paired_repeat_differences.csv")) == 90
    decision = m.read(out / "decision.json")
    assert decision["automatic_promotion"] is False
    assert decision["wins"] + decision["losses"] + decision["ties"] == 10
    manifest = m.read(out / "run_manifest.json")
    for name, value in manifest["artifact_sha256"].items():
        assert m.digest(out / name) == value
    with pytest.raises(ValueError, match="Fresh output"):
        m.run("config.json", root=tmp_path)


def test_shuffle_is_aligned_by_keys(tmp_path):
    fixture_project(tmp_path)
    p = tmp_path / "pca/validation_probabilities.csv"
    pd.read_csv(p, dtype={"id": "string"}, float_precision="round_trip").sample(frac=1, random_state=9).to_csv(p, index=False)
    seal(p.parent)
    assert m.run("config.json", dry_run=True, root=tmp_path)["status"] == "preflight_passed"


def test_tampered_artifact_rejected(tmp_path):
    fixture_project(tmp_path)
    with (tmp_path / "pca/decision.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="source artifact"):
        m.run("config.json", dry_run=True, root=tmp_path)


def test_ordered_training_indices_must_match(tmp_path):
    fixture_project(tmp_path)
    p = tmp_path / "pca/split_indices.json"
    splits = m.read(p)
    splits[0]["train_indices"].reverse()
    m.write(p, splits); seal(p.parent)
    with pytest.raises(ValueError, match="Ordered training"):
        m.run("config.json", dry_run=True, root=tmp_path)


def test_id_mismatch_rejected(tmp_path):
    fixture_project(tmp_path)
    p = tmp_path / "pca/validation_probabilities.csv"
    df = pd.read_csv(p, dtype={"id": "string"}, float_precision="round_trip")
    df.loc[df.candidate == "pca_50_C0.01", "id"] += "changed"
    df.to_csv(p, index=False); seal(p.parent)
    with pytest.raises(ValueError, match="alignment mismatch"):
        m.run("config.json", dry_run=True, root=tmp_path)


def test_duplicate_prediction_rejected(tmp_path):
    fixture_project(tmp_path)
    p = tmp_path / "pca/validation_probabilities.csv"
    df = pd.read_csv(p, dtype={"id": "string"}, float_precision="round_trip")
    pd.concat([df, df.iloc[:1]]).to_csv(p, index=False); seal(p.parent)
    with pytest.raises(ValueError, match="duplicate prediction"):
        m.run("config.json", dry_run=True, root=tmp_path)


def test_source_score_mismatch_rejected(tmp_path):
    fixture_project(tmp_path)
    p = tmp_path / "pca/fold_scores.csv"
    df = pd.read_csv(p, float_precision="round_trip")
    df.loc[0, "auc"] += .001
    df.to_csv(p, index=False); seal(p.parent)
    with pytest.raises(ValueError, match="reproduction failed"):
        m.run("config.json", dry_run=True, root=tmp_path)


def test_baseline_expected_auc_guard(tmp_path):
    cfg = fixture_project(tmp_path)
    cfg["expected_reference_auc"] += .001
    m.write(tmp_path / "config.json", cfg)
    with pytest.raises(ValueError, match="baseline mean"):
        m.run("config.json", dry_run=True, root=tmp_path)


def test_equal_weight_formula(tmp_path):
    cfg = fixture_project(tmp_path)
    _, _, frame, _, _, _ = m.prepare("config.json", root=tmp_path)
    blended, *_ = m.evaluate(frame, cfg)
    assert np.allclose(blended["pca_l2_alpha0.25"], .75 * frame[m.BASE] + .25 * frame.pca_l2)


def test_ties_are_not_wins():
    repeats = pd.DataFrame([dict(candidate=name, repeat=r, mean_fold_auc=.8) for name in ["a", "b"] for r in range(10)])
    _, stat = m.paired(repeats, "b", "a", "test")
    assert stat["ties"] == 10 and stat["wins"] == stat["losses"] == 0


def test_frozen_alpha_grid(tmp_path):
    cfg = fixture_project(tmp_path)
    cfg["alphas"] = [.1, .2]
    m.write(tmp_path / "config.json", cfg)
    with pytest.raises(ValueError, match="V1 fixes"):
        m.run("config.json", dry_run=True, root=tmp_path)
