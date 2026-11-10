"""Trial25 integrity, actual sklearn persistence and failure-safe export tests."""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from src.modeling.elasticnet_trial_model import ElasticnetTrialModel
from src.modeling.model_registry import RegisteredModel
from src.submission.elasticnet_trial_snapshot import ElasticnetTrialSnapshot
from src.submission.elasticnet_trial_submission import ElasticnetTrialSubmission


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, allow_nan=False))


def test_helper(root):
    # On the real repository use its actual strict CSV helper; contract fallback in sandbox.
    try:
        from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission
        return FrozenXGBoostSubmission()
    except ModuleNotFoundError as exc:
        if exc.name != "src.submission.frozen_xgboost_submission":
            raise
    def read_csv(path, id_column):
        with Path(path).open(newline="") as stream:
            header = next(csv.reader(stream))
        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV header")
        return pd.read_csv(path, dtype={id_column: "string"})
    def inputs(test, template, features, id_col, target_col, expected):
        t, target = read_csv(test, id_col), read_csv(template, id_col)
        if set(t.columns) != {id_col, *features} or list(target.columns) != [id_col, target_col]:
            raise ValueError("Schema mismatch")
        if len(t) != expected or len(target) != expected:
            raise ValueError("Row count mismatch")
        for f in (t, target):
            if f[id_col].isna().any() or not f[id_col].is_unique or f[id_col].str.strip().eq("").any():
                raise ValueError("Invalid IDs")
        if set(t[id_col]) != set(target[id_col]):
            raise ValueError("ID sets differ")
        x = t[features]
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite features")
        return x, t[id_col].copy(), target
    def align(ids, scores, template, id_col, target_col):
        frame = template[[id_col]].copy()
        frame[target_col] = frame[id_col].map(pd.Series(scores, index=ids.to_numpy()))
        if not np.isfinite(frame[target_col]).all():
            raise ValueError("Unmatched IDs")
        return frame
    return SimpleNamespace(_path=lambda p: (root / p).resolve(), _sha=digest,
                           _write=write, _test_inputs=inputs, _align=align, _read_csv=read_csv)


test_helper.__test__ = False


@pytest.fixture
def sample(tmp_path, monkeypatch):
    root = tmp_path
    helper = test_helper(root)
    run, dest = root / "study", root / "export"
    run.mkdir()
    x, y = make_classification(n_samples=40, n_features=8, n_informative=4, random_state=5)
    x, y = pd.DataFrame(x, columns=[f"f{i}" for i in range(8)]), pd.Series(y)
    groups, rows, offset = [], [], 0
    for repeat in range(10):
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=repeat).split(x, y)):
            groups.append(dict(repeat=repeat, fold=fold, split_seed=repeat,
                               train_indices=tr.tolist(), valid_indices=va.tolist(),
                               start=offset, stop=offset + len(va)))
            rows.extend(dict(repeat=repeat, fold=fold, split_seed=repeat, row_index=int(i), id=str(i), target=int(y.iloc[i])) for i in va)
            offset += len(va)
    frame = pd.DataFrame(rows)
    frame["id"] = frame["id"].astype("string")
    train = root / "train.csv"
    x.assign(target=y).to_csv(train, index=False)
    code = root / "frozen_source.py"
    code.write_text("frozen")
    config = root / "config.json"
    cfg = dict(suite_config="suite.json", output_dir=str(run), design={"n_trials": 30})
    write(config, cfg)
    hashes = {str(p): digest(p) for p in (train, code, config)}
    test = root / "test.csv"
    t = x.iloc[:7].copy() + 15
    t.insert(0, "id", [f"00{i}" for i in range(7)])
    t.to_csv(test, index=False)
    template = root / "sample_submission.csv"
    pd.DataFrame(dict(id=t.id.iloc[::-1].to_list(), target=0)).to_csv(template, index=False)
    ctx = dict(x=x, y=y, candidate=dict(default_test_path=str(test), default_template_path=str(template),
               expected_test_rows=7, estimator_parameters={"seed": 42}, k=20),
               base=SimpleNamespace(id_column="id", target_column="target"),
               linear=SimpleNamespace(to_dict=lambda: {"regularization_c": 1.0}))
    c = dict(context=ctx, groups=groups, frame=frame, baseline=np.full(offset, .5),
             catboost_control=np.full(offset, .5), elasticnet_hashes=hashes,
             optuna_config=cfg, lightgbm_parameters={"n_estimators": 50}, versions={})
    stats = dict(reference="reference", candidate="candidate", mean_difference=.5,
                 difference_sd=0., wins=10, losses=0, ties=0)
    catstats = dict(stats, reference="catboost_frozen")
    params = ElasticnetTrialSnapshot.expected_parameters.copy()
    best = dict(number=25, state="COMPLETE", parameters=params, objective_auc=1., standalone_auc=1.,
                repeat_sd=0., reference_comparison=stats, catboost_comparison=catstats, control_reproduced=None)
    decision = dict(selected_trial=25, selected_mean_auc=1., parameters=params, screening_passed=True,
                    screening_both_controls=True, reference_comparison=stats,
                    catboost_control_comparison=catstats, control_reproduced=True, automatic_promotion=False)
    settings = dict(config=cfg, input_and_code_sha256=hashes, versions={"scikit-learn": "synthetic"},
                    solver="saga", max_iter=20000, tol=1e-4, model_seed=42)
    records = [dict(number=i, state="COMPLETE", value=1. if i == 25 else .5,
                    parameters=params if i == 25 else dict(C=.1, l1_ratio=.75, alpha=.05),
                    user_attrs={"control_reproduced": True} if i == 0 else {}) for i in range(30)]
    for name, value in [("settings", settings), ("decision", decision), ("best_trial", best),
                        ("trials", records), ("split_indices", groups)]:
        write(run / (name + ".json"), value)
    pd.DataFrame([dict(number=r["number"], state=r["state"], objective_auc=r["value"], **r["parameters"]) for r in records]).to_csv(run / "trials.csv", index=False)
    frame.to_csv(run / "cohort.csv", index=False)
    for number in (0, 25):
        d = run / "trials" / f"trial_{number:04d}"
        d.mkdir(parents=True)
        result = best if number == 25 else dict(number=0, state="COMPLETE", parameters=records[0]["parameters"], objective_auc=.5, control_reproduced=True)
        write(d / "result.json", result)
        write(d / "parameters.json", records[number]["parameters"])
        raw = .1 + .8 * frame.target.to_numpy() if number == 25 else np.full(offset, .5)
        alpha = records[number]["parameters"]["alpha"]
        np.savez_compressed(d / "predictions.npz", elasticnet=raw, ensemble=(1-alpha)*c["baseline"] + alpha*raw)
    manifest = dict(status="completed", planned_trials=30, completed_trials=30, planned_fits=1500,
                    completed_fits=1500, test_opened=False, kaggle_upload_performed=False)
    def rehash():
        manifest["artifact_sha256"] = {str(p.relative_to(run)): digest(p) for p in run.rglob("*") if p.is_file() and p.name != "run_manifest.json"}
        write(run / "run_manifest.json", manifest)
    rehash()
    monkeypatch.setattr(ElasticnetTrialSnapshot, "expected_auc", 1.)
    monkeypatch.setattr(ElasticnetTrialSnapshot, "control_auc", .5)
    snapshot = ElasticnetTrialSnapshot(helper=helper, prepare_study=lambda *a, **k: c,
                                     version_reader=lambda p: "synthetic")
    # Four actual sklearn procedures; tree branches substituted, not native-tree integration.
    factory = lambda alias, c, p: ElasticnetTrialModel({k: p[k] for k in ("C", "l1_ratio")})
    exporter = ElasticnetTrialSubmission(snapshot=snapshot, model_factory=factory)
    return SimpleNamespace(root=root, run=run, dest=dest, config=config, context=c, helper=helper,
                           snapshot=snapshot, exporter=exporter, rehash=rehash, test=test, template=template)


def invoke(s, **kwargs):
    return s.exporter.predict(run=str(s.run), config=str(s.config), output=str(s.dest), **kwargs)


def test_dry_run_no_fit_or_output(sample, monkeypatch):
    monkeypatch.setattr(ElasticnetTrialModel, "fit", lambda *a: pytest.fail("dry-run fitted a model"))
    result = invoke(sample, dry_run=True)
    assert result["fits_performed"] == 0 and result["planned_estimator_fits"] == 5
    assert result["test_opened"] is True and not sample.dest.exists()


def test_actual_logistic_save_reload_and_submission(sample):
    result = invoke(sample)
    assert result["status"] == "ready_for_manual_submission" and result["fits_performed"] == 5
    manifest = sample.snapshot.read(sample.dest / "prediction_manifest.json")
    assert manifest["saved_and_reloaded_models_verified"] is True
    assert manifest["kaggle_upload_performed"] is False and manifest["automatic_promotion"] is False
    exported = sample.helper._read_csv(sample.dest / "submission.csv", "id")
    target = sample.helper._read_csv(sample.template, "id")
    assert exported.id.equals(target.id) and exported.id.iloc[0] == "006"
    components = pd.read_csv(sample.dest / "component_predictions.csv", dtype={"id": "string"})
    raw = (1-sample.snapshot.expected_parameters["alpha"]) * components.original_three_model + sample.snapshot.expected_parameters["alpha"] * components.elasticnet
    expected = pd.Series(raw.to_numpy(), index=components.id)
    assert np.allclose(exported.target, exported.id.map(expected), rtol=0, atol=1e-12)
    model = RegisteredModel.load(sample.dest / "models/elasticnet.joblib")
    assert np.allclose(model.model.named_steps["scaler"].mean_, sample.context["context"]["x"].mean())
    for path, value in manifest["artifact_sha256"].items():
        assert digest(sample.dest / path) == value


def test_existing_output_rejected(sample):
    sample.dest.mkdir()
    with pytest.raises(FileExistsError):
        invoke(sample, dry_run=True)


def test_artifact_tampering_rejected(sample):
    (sample.run / "best_trial.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        invoke(sample, dry_run=True)
    assert not sample.dest.exists()


def test_rehashed_selection_change_rejected(sample):
    p = sample.run / "decision.json"
    decision = sample.snapshot.read(p)
    decision["selected_trial"] = 24
    write(p, decision)
    sample.rehash()
    with pytest.raises(ValueError, match="Trial25"):
        invoke(sample, dry_run=True)


def test_rehashed_prediction_formula_change_rejected(sample):
    path = sample.run / "trials/trial_0025/predictions.npz"
    with np.load(path) as z:
        raw = z["elasticnet"].copy()
    np.savez_compressed(path, elasticnet=raw, ensemble=raw)
    sample.rehash()
    with pytest.raises(ValueError, match="formula"):
        invoke(sample, dry_run=True)


def test_environment_change_rejected(sample):
    sample.snapshot.version_reader = lambda p: "different"
    with pytest.raises(ValueError, match="version changed"):
        invoke(sample, dry_run=True)


def test_source_mutation_rejected(sample):
    (sample.root / "frozen_source.py").write_text("mutated")
    with pytest.raises(ValueError, match="changed"):
        invoke(sample, dry_run=True)


@pytest.mark.parametrize("problem", ["duplicate_ids", "missing_id", "wrong_schema", "nonfinite"])
def test_invalid_test_data_rejected(sample, problem):
    test = pd.read_csv(sample.test, dtype={"id": "string"})
    if problem == "duplicate_ids":
        test.loc[0, "id"] = test.loc[1, "id"]
    elif problem == "missing_id":
        test.loc[0, "id"] = "unknown"
    elif problem == "wrong_schema":
        test = test.drop(columns="f0")
    else:
        test.loc[0, "f0"] = np.inf
    test.to_csv(sample.test, index=False)
    with pytest.raises(ValueError):
        invoke(sample, dry_run=True)
    assert not sample.dest.exists()


def test_warning_aborts_without_submission(sample, monkeypatch):
    import warnings
    old = ElasticnetTrialModel.fit
    def warn_fit(self, x, y):
        model = old(self, x, y)
        warnings.warn("synthetic convergence issue", ConvergenceWarning)
        return model
    monkeypatch.setattr(ElasticnetTrialModel, "fit", warn_fit)
    with pytest.raises(RuntimeError, match="warnings"):
        invoke(sample)
    assert not (sample.dest / "submission.csv").exists()
    assert sample.snapshot.read(sample.dest / "prediction_manifest.json")["status"] == "failed_or_interrupted"


def test_reload_mismatch_aborts_without_submission(sample, monkeypatch):
    monkeypatch.setattr(RegisteredModel, "load", staticmethod(lambda p: SimpleNamespace(positive_probability=lambda x: np.ones(len(x)))))
    with pytest.raises(ValueError, match="Reloaded"):
        invoke(sample)
    assert not (sample.dest / "submission.csv").exists()


def test_test_input_change_during_fit_aborts(sample, monkeypatch):
    old = ElasticnetTrialModel.fit
    def mutate(self, x, y):
        fitted = old(self, x, y)
        sample.test.write_text(sample.test.read_text() + "\n")
        return fitted
    monkeypatch.setattr(ElasticnetTrialModel, "fit", mutate)
    with pytest.raises(ValueError, match="changed"):
        invoke(sample)
    assert not (sample.dest / "submission.csv").exists()


def test_model_rejects_feature_order_change(sample):
    x, y = sample.context["context"]["x"], sample.context["context"]["y"]
    model = ElasticnetTrialModel(dict(C=.2, l1_ratio=.95)).fit(x, y)
    with pytest.raises(ValueError):
        model.positive_probability(x[x.columns[::-1]])


def test_failed_study_rejected(sample):
    p = sample.run / "run_manifest.json"
    manifest = sample.snapshot.read(p)
    manifest["status"] = "failed_or_interrupted"
    write(p, manifest)
    with pytest.raises(ValueError, match="completed"):
        invoke(sample, dry_run=True)


def test_path_traversal_rejected(sample):
    p = sample.run / "run_manifest.json"
    manifest = sample.snapshot.read(p)
    manifest["artifact_sha256"]["../train.csv"] = digest(sample.root / "train.csv")
    write(p, manifest)
    with pytest.raises(ValueError, match="Unsafe"):
        invoke(sample, dry_run=True)


def test_rehashed_score_inconsistency_rejected(sample):
    path = sample.run / "trials/trial_0025/predictions.npz"
    raw = np.full(len(sample.context["frame"]), .5)
    np.savez_compressed(path, elasticnet=raw, ensemble=raw)
    sample.rehash()
    with pytest.raises(ValueError, match="recomputed ensemble"):
        invoke(sample, dry_run=True)


def test_rehashed_false_screening_rejected(sample):
    for name in ("best_trial.json", "trials/trial_0025/result.json"):
        path = sample.run / name
        record = sample.snapshot.read(path)
        record["reference_comparison"]["wins"] = 9
        write(path, record)
    sample.rehash()
    with pytest.raises(ValueError, match="wins"):
        invoke(sample, dry_run=True)


def test_cli_is_thin_and_passes_flags(monkeypatch, capsys):
    from scripts import predict_elasticnet_trial_25 as cli
    calls = []
    monkeypatch.setattr(cli, "ElasticnetTrialSubmission", lambda: SimpleNamespace(predict=lambda **kw: calls.append(kw) or {"status": "checked"}))
    assert cli.main(["--dry-run", "--output", "fresh_output"]) == 0
    assert calls[0]["dry_run"] is True and calls[0]["output"] == "fresh_output"
    assert json.loads(capsys.readouterr().out)["status"] == "checked"
