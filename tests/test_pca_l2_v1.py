"""Synthetic tests only: no competition data or model training at import time."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from scripts import run_pca_l2_v1 as m


def fixture_project(tmp_path):
    x, y = make_classification(n_samples=40, n_features=8, n_informative=3, random_state=2)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(8)])
    frame.insert(0, "id", [f"row{i}" for i in range(40)])
    frame["target"] = y
    train = tmp_path / "data/raw/train.csv"
    train.parent.mkdir(parents=True)
    frame.to_csv(train, index=False)
    (train.parent / "test.csv").write_text("DO NOT OPEN THIS FILE")
    source = tmp_path / "source"
    source.mkdir()
    records = []
    for r in range(10):
        for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(x, y)):
            records.append(dict(scheme="cv5", repeat=r, fold=f, split_seed=r,
                                train_indices=tr.tolist(), valid_indices=va.tolist()))
    m.write(source / "split_indices.json", records)
    m.write(source / "settings.json", {"input_and_code_sha256": {str(train): m.digest(train)}})
    m.write(source / "run_manifest.json", dict(status="completed", artifact_sha256={
        p.name: m.digest(p) for p in source.iterdir()}))
    cfg = dict(train_path="data/raw/train.csv", target_column="target", id_column="id",
               expected_rows=40, expected_features=8, source_run="source", output_dir="result",
               c_values=[.1, 1.0], components=[2], max_iter=1000, tol=.0001, model_seed=42)
    m.write(tmp_path / "config.json", cfg)
    return cfg, frame


def test_preflight_no_fit_or_output(tmp_path, monkeypatch):
    cfg, _ = fixture_project(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("No fit during dry run")
    monkeypatch.setattr(m.Pipeline, "fit", forbidden)
    original = pd.read_csv
    def train_only(path, *args, **kwargs):
        assert Path(path).name == "train.csv"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(m.pd, "read_csv", train_only)
    result = m.run("config.json", root=tmp_path, dry_run=True)
    assert result["planned_pipeline_fits"] == 200
    assert not (tmp_path / "result").exists()


def test_complete_synthetic_run(tmp_path):
    fixture_project(tmp_path)
    result = m.run("config.json", root=tmp_path)
    assert result["status"] == "completed"
    out = tmp_path / "result"
    manifest = m.read(out / "run_manifest.json")
    assert manifest["completed_pipeline_fits"] == 200
    assert not manifest["test_opened"]
    assert len(pd.read_csv(out / "fold_scores.csv")) == 200
    assert len(pd.read_csv(out / "validation_probabilities.csv")) == 1600
    assert len(pd.read_csv(out / "repeat_scores.csv")) == 40
    decision = m.read(out / "decision.json")
    assert decision["automatic_promotion"] is False
    assert decision["wins"] + decision["losses"] + decision["ties"] == 10
    for name, expected in manifest["artifact_sha256"].items():
        assert m.digest(out / name) == expected
    with pytest.raises(FileExistsError):
        m.run("config.json", root=tmp_path, dry_run=True)


def test_pipeline_fits_training_only(tmp_path):
    cfg, _ = fixture_project(tmp_path)
    _, _, x, y, _, splits, _ = m.prepare("config.json", tmp_path)
    split = splits[0]
    tr, va = split["train_indices"], split["valid_indices"]
    x.iloc[va] += 1000
    model = m.build_model(m.candidates(cfg)[2], cfg)
    model.fit(x.iloc[tr], y.iloc[tr])
    assert np.allclose(model.named_steps["scaler"].mean_, x.iloc[tr].mean())
    assert not np.allclose(model.named_steps["scaler"].mean_, x.mean())
    assert model.named_steps["pca"].n_samples_ == len(tr)


def test_reject_tampered_source(tmp_path):
    fixture_project(tmp_path)
    with (tmp_path / "source/split_indices.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="source artifact"):
        m.prepare("config.json", tmp_path)


def test_reject_changed_training(tmp_path):
    fixture_project(tmp_path)
    with (tmp_path / "data/raw/train.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA256"):
        m.prepare("config.json", tmp_path)


def test_reject_overlap(tmp_path):
    cfg, frame = fixture_project(tmp_path)
    records = m.read(tmp_path / "source/split_indices.json")
    records[0]["valid_indices"][0] = records[0]["train_indices"][0]
    with pytest.raises(ValueError, match="partition"):
        m.validate_splits(records, frame.target, cfg)


def test_paired_direction_and_ties():
    table = pd.DataFrame([dict(candidate=name, repeat=r, mean_fold_auc=.7 + (r % 2) * .01)
                          for name in ["a", "b"] for r in range(10)])
    _, stat = m.compare(table, "b", "a")
    assert stat["ties"] == 10
    table.loc[table.candidate == "b", "mean_fold_auc"] += .02
    _, stat = m.compare(table, "b", "a")
    assert stat["wins"] == 10 and np.isclose(stat["mean_difference"], .02)


def test_default_design_and_bad_components():
    root = Path(__file__).resolve().parents[1]
    cfg = m.validate_config(m.read(root / "configs/pca_l2_v1.json"))
    assert len(m.candidates(cfg)) == 25
    assert len(m.candidates(cfg)) * 50 == 1250
    cfg["components"] = [5, 5]
    with pytest.raises(ValueError):
        m.validate_config(cfg)
