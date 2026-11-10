"""Bounded RF addition to a frozen three-model OOF baseline; CV only."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

TOL = 1e-12
KEYS = ["repeat", "split_seed", "fold", "row_index"]
GRID = [(3, 5), (3, 10), (5, 5), (5, 10)]
BASELINE = "trial14_three_model"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def probability(p, length):
    p = np.asarray(p, dtype=float)
    if p.shape != (length,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid positive-class probability array.")
    return p


def blend(base, rf):
    base = probability(base, len(base))
    rf = probability(rf, len(base))
    return .9 * base + .1 * rf


def validate_config(cfg):
    fields = {"source_study", "expected_trial", "expected_baseline_auc", "output_dir"}
    if not isinstance(cfg, dict) or set(cfg) != fields:
        raise ValueError("Unexpected RF configuration fields.")
    if type(cfg["expected_trial"]) is not int or cfg["expected_trial"] != 14:
        raise ValueError("V1 is locked to the previously selected Trial14.")
    value = cfg["expected_baseline_auc"]
    if type(value) not in (int, float) or not np.isfinite(value) or abs(value - .9083364180103309) > TOL:
        raise ValueError("V1 baseline AUC differs from the recorded Trial14 result.")
    if any(not isinstance(cfg[k], str) or not cfg[k].strip() for k in ("source_study", "output_dir")):
        raise ValueError("Paths must be nonempty strings.")
    return cfg


def verify_artifacts(root, helper, required):
    root = root.resolve()
    manifest_path = root / "run_manifest.json"
    manifest = read(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("Source run is not completed.")
    checks = manifest.get("artifact_sha256", {})
    if not required <= checks.keys():
        raise ValueError("Required source artifact hashes are missing.")
    paths = [manifest_path]
    for name, digest in checks.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or helper._sha(path) != digest:
            raise ValueError(f"Source artifact changed or escaped source directory: {name}")
        paths.append(path)
    return manifest, paths


def validate_oof(frame, scores, c):
    """Verify exact split keys, labels, IDs, probabilities, blends and fold AUCs."""
    names = ["l1", "xgboost", "lightgbm", "l1_xgb_lgbm"]
    required = set(KEYS + ["id", "target"] + names)
    if not required <= set(frame) or len(frame) != 10 * len(c["x"]):
        raise ValueError("Missing OOF columns or wrong OOF row count.")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate OOF keys.")
    for key in KEYS:
        v = frame[key].to_numpy(dtype=float)
        if not np.isfinite(v).all() or not np.equal(v, np.floor(v)).all():
            raise ValueError("OOF keys must be finite integers.")
    indexed = frame.set_index(KEYS)
    if not indexed.index.is_unique:
        raise ValueError("Duplicate OOF index keys.")
    expected_keys = []
    if len(scores) != 50 or scores.duplicated(["repeat", "fold"]).any():
        raise ValueError("Invalid source fold-score keys.")
    scores = scores.set_index(["repeat", "fold"])
    if not scores.index.is_unique:
        raise ValueError("Duplicate source fold-score index keys.")
    expected_folds = {(s["repeat"], s["fold"]) for s in c["splits"]}
    if set(scores.index) != expected_folds:
        raise ValueError("Source fold assignments changed.")
    for s in c["splits"]:
        va = s["valid_indices"]
        keys = [(s["repeat"], s["split_seed"], s["fold"], int(i)) for i in va]
        expected_keys.extend(keys)
        f = indexed.loc[keys]
        y = c["y"].iloc[va].to_numpy()
        if f.id.astype(str).tolist() != c["ids"].iloc[va].astype(str).tolist() or not np.array_equal(f.target, y):
            raise ValueError("Source OOF IDs or targets differ from training data.")
        p = {n: probability(f[n].to_numpy(), len(va)) for n in names}
        if not np.allclose(p["l1_xgb_lgbm"], (p["l1"] + p["xgboost"] + p["lightgbm"]) / 3, rtol=0, atol=TOL):
            raise ValueError("Source is not the equal-weight three-model baseline.")
        cached = c["cache"].loc[keys]
        for n in ("l1", "xgboost"):
            if not np.allclose(p[n], cached[n].to_numpy(), rtol=0, atol=TOL):
                raise ValueError("Source differs from verified L1/XGBoost cache.")
        for n in names:
            if not np.isclose(roc_auc_score(y, p[n]), scores.loc[(s["repeat"], s["fold"]), n], rtol=0, atol=TOL):
                raise ValueError(f"Source probability/AUC mismatch: {n}")
        if not np.isclose(scores.loc[(s["repeat"], s["fold"]), "xgboost"], c["expected"].loc[(s["repeat"], s["fold"])], rtol=0, atol=TOL):
            raise ValueError("Frozen XGBoost control mismatch.")
    if set(indexed.index) != set(expected_keys):
        raise ValueError("Unexpected source OOF assignments.")
    return indexed


def summarize(folds):
    required = {"repeat", "split_seed", "fold", "candidate", "auc"}
    if not required <= set(folds) or folds.duplicated(["candidate", "repeat", "fold"]).any():
        raise ValueError("Invalid result keys.")
    names = [BASELINE] + [f"{kind}_d{d}_leaf{leaf}" for d, leaf in GRID for kind in ("rf", "blend")]
    if set(folds.candidate) != set(names) or len(folds) != 450:
        raise ValueError("Incomplete results: all 9 procedures must have 50 folds.")
    values = folds.auc.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid fold AUC values.")
    baseline_keys = set(folds.loc[folds.candidate == BASELINE, ["repeat", "split_seed", "fold"]].itertuples(index=False, name=None))
    for _, f in folds.groupby("candidate"):
        if set(f[["repeat", "split_seed", "fold"]].itertuples(index=False, name=None)) != baseline_keys:
            raise ValueError("Candidate fold keys differ from baseline.")
        if len(f) != 50 or f.groupby("repeat").size().tolist() != [5] * 10 or f.groupby("repeat").split_seed.nunique().max() != 1:
            raise ValueError("Each candidate needs ten complete five-fold repeats.")
    repeats = folds.groupby(["candidate", "repeat", "split_seed"], as_index=False).auc.mean()
    base = repeats.loc[repeats.candidate == BASELINE].set_index(["repeat", "split_seed"]).auc
    rows = []
    for name in names:
        r = repeats.loc[repeats.candidate == name].set_index(["repeat", "split_seed"]).auc
        if set(r.index) != set(base.index):
            raise ValueError("Paired repeat keys differ.")
        r = r.loc[base.index]
        delta = r - base
        sd, base_sd = float(r.std(ddof=1)), float(base.std(ddof=1))
        mean_delta = float(delta.mean())
        wins = int((delta > TOL).sum())
        rows.append(dict(candidate=name, mean_auc=float(r.mean()), repeat_sd=sd,
            minimum_repeat_auc=float(r.min()), maximum_repeat_auc=float(r.max()),
            mean_difference=mean_delta, difference_sd=float(delta.std(ddof=1)),
            minimum_difference=float(delta.min()), maximum_difference=float(delta.max()),
            wins=wins, losses=int((delta < -TOL).sum()), ties=int((abs(delta) <= TOL).sum()),
            passes_auc_gate=bool(name.startswith("blend_") and mean_delta > TOL and wins >= 6),
            stability_observed=bool(name.startswith("blend_") and sd <= base_sd + TOL and r.min() >= base.min() - TOL)))
    table = pd.DataFrame(rows)
    candidates = table.loc[table.candidate.str.startswith("blend_")].copy()
    # Exact mean ties retain the earlier predeclared GRID order. No fallback.
    best = candidates.loc[candidates.mean_auc.idxmax()]
    promote = bool(best.passes_auc_gate)
    decision = dict(baseline=BASELINE, best_candidate=str(best.candidate),
        selected=str(best.candidate) if promote else None, promote=promote,
        stability_observed=bool(best.stability_observed),
        rule="Highest blend mean AUC; exact ties use GRID order. Winner must improve mean by >1e-12 and win >=6/10 repeats. No fallback.",
        stability_definition="Descriptive only: repeat SD no higher AND minimum repeat AUC no lower than baseline (tolerance 1e-12). Not an inferential guarantee.",
        interpretation="Adaptive development on reused splits; neither independent evaluation nor proof of private-score improvement.",
        next_action="Manual review only. No prediction, refit or upload implemented.")
    return repeats, table, decision


class RFStabilizer:
    def _prepare(self, config_path, output=None):
        from src.experiments.lgbm_optuna_experiment import LGBMOptunaExperiment
        from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission
        h = FrozenXGBoostSubmission()
        path = h._path(config_path)
        cfg = validate_config(read(path))
        out = h._path(output or cfg["output_dir"])
        if out.exists():
            raise FileExistsError("Choose a new output directory; no overwrite or resume.")
        study = h._path(cfg["source_study"])
        manifest, paths = verify_artifacts(study, h, {"settings.json", "best_trial.json", "decision.json"})
        settings, best, decision = (read(study / n) for n in ("settings.json", "best_trial.json", "decision.json"))
        if (manifest.get("control_trial_reproduced") is not True or not decision.get("promote")
            or decision.get("selected_trial") != 14 or best.get("number") != 14
            or best.get("state") != "COMPLETE" or best.get("reference_reproduced") is not True):
            raise ValueError("Need the promoted, control-reproduced Trial14 source.")
        if abs(float(best["objective_auc"]) - cfg["expected_baseline_auc"]) > TOL:
            raise ValueError("Selected source AUC differs from the recorded baseline.")
        original = LGBMOptunaExperiment()._prepare(settings["config_path"], str(out))
        if (original["hashes"] != settings["input_and_code_sha256"]
            or original["versions"] != settings["versions"] or original["config"] != settings["config"]):
            raise ValueError("Historical inputs/code/config/environment changed. Do not bypass checks.")
        c = original["context"]
        if len(c["splits"]) != 50 or len(c["x"]) != 250 or c["x"].shape[1] != 150:
            raise ValueError("V1 requires the original 250-row/150-feature/50-fold development problem.")
        trial = h._path(best["cv_run"])
        if trial != (study / "trials/trial_0014/cv").resolve():
            raise ValueError("Unexpected Trial14 path; moving historical absolute paths requires a separate migration, not a bypass.")
        if h._sha(trial / "run_manifest.json") != best["registry_manifest_sha256"]:
            raise ValueError("Trial manifest differs from selected source record.")
        tm, tp = verify_artifacts(trial, h, {"settings.json", "fold_scores.csv", "validation_probabilities.csv", "comparison.csv"})
        if tm.get("completed_folds") != 50 or tm.get("control_auc_reproduced") is not True:
            raise ValueError("Trial14 is not a complete control-reproduced CV run.")
        trial_settings = read(trial / "settings.json")
        model_cfg = trial_settings["config"]
        expected_cfg = json.loads(json.dumps(original["base"]))
        expected_cfg["models"]["lightgbm"]["parameters"] = best["parameters"]
        expected_cfg["output_dir"] = str(trial)
        if model_cfg != expected_cfg:
            raise ValueError("Trial14 resolved model specification differs from stored selected parameters.")
        p = pd.read_csv(trial / "validation_probabilities.csv", dtype={"id": "string"}, float_precision="round_trip")
        scores = pd.read_csv(trial / "fold_scores.csv", float_precision="round_trip")
        indexed = validate_oof(p, scores, c)
        auc = scores.groupby(["repeat", "split_seed"]).l1_xgb_lgbm.mean().mean()
        if not np.isclose(auc, cfg["expected_baseline_auc"], rtol=0, atol=TOL):
            raise ValueError("Baseline aggregate did not reproduce.")
        paths += tp + [path, Path(__file__), h._path("scripts/run_rf_stabilizer.py")]
        hashes = dict(original["hashes"])
        hashes.update({str(p.resolve()): h._sha(p) for p in paths})
        return dict(helper=h, context=c, config=cfg, output=out, source=trial, oof=indexed,
                    scores=scores, hashes=hashes, versions=original["versions"])

    @staticmethod
    def _unchanged(c):
        for path, digest in c["hashes"].items():
            if c["helper"]._sha(path) != digest:
                raise RuntimeError(f"Input/code/artifact changed: {path}")

    def run(self, config_path, output=None, dry_run=False):
        c = self._prepare(config_path, output)
        self._unchanged(c)
        if dry_run:
            return dict(status="preflight_passed", new_forest_fits=200, trees_per_forest=500,
                baseline=BASELINE, rf_weight=.1, cached_models=["l1", "xgboost", "lightgbm"],
                source_trial=14, test_opened=False, output=str(c["output"]))
        h, out, ctx = c["helper"], c["output"], c["context"]
        out.mkdir(parents=True, exist_ok=False)
        manifest = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
            planned_forest_fits=200, attempted_forest_fits=0, completed_forest_fits=0,
            test_opened=False, kaggle_upload_performed=False, source_trial=14)
        rows, predictions, diagnostics, warning_records = [], [], [], []
        h._write(out / "run_manifest.json", manifest)
        try:
            h._write(out / "settings.json", dict(config=c["config"], versions=c["versions"],
                input_and_code_sha256=c["hashes"], source_trial_cv=str(c["source"]),
                grid=[dict(max_depth=d, min_samples_leaf=l) for d, l in GRID],
                fixed_rf=dict(n_estimators=500, max_features="sqrt", bootstrap=True, random_state=42, n_jobs=1),
                baseline_weight=.9, rf_weight=.1, prediction_export_implemented=False))
            for s in ctx["splits"]:
                key = {k: s[k] for k in ("repeat", "split_seed", "fold")}
                score = c["scores"].set_index(["repeat", "fold"]).loc[(s["repeat"], s["fold"]), "l1_xgb_lgbm"]
                rows.append(dict(key, candidate=BASELINE, auc=float(score)))
            for depth, leaf in GRID:
                alias = f"d{depth}_leaf{leaf}"
                for s in ctx["splits"]:
                    self._unchanged(c)
                    tr, va = s["train_indices"], s["valid_indices"]
                    key = {k: s[k] for k in ("repeat", "split_seed", "fold")}
                    source_keys = [(s["repeat"], s["split_seed"], s["fold"], int(i)) for i in va]
                    base = c["oof"].loc[source_keys, "l1_xgb_lgbm"].to_numpy()
                    model = RandomForestClassifier(n_estimators=500, max_depth=depth,
                        min_samples_leaf=leaf, max_features="sqrt", bootstrap=True,
                        random_state=42, n_jobs=1, class_weight=None)
                    caught = []
                    manifest["attempted_forest_fits"] += 1
                    h._write(out / "run_manifest.json", manifest)
                    try:
                        with warnings.catch_warnings(record=True) as caught, (out / "fit.log").open("a", encoding="utf-8") as log:
                            warnings.simplefilter("always")
                            with redirect_stdout(log), redirect_stderr(log):
                                model.fit(ctx["x"].iloc[tr], ctx["y"].iloc[tr])
                                if not np.array_equal(model.classes_, [0, 1]):
                                    raise ValueError("Unexpected RF class order.")
                                rf = probability(model.predict_proba(ctx["x"].iloc[va])[:, 1], len(va))
                    finally:
                        warning_records.extend(dict(key, configuration=alias, category=w.category.__name__, message=str(w.message)) for w in caught)
                        h._write(out / "warnings.json", warning_records)
                    if caught:
                        raise RuntimeError("Python warnings require review; no scored candidate accepted.")
                    mixed = blend(base, rf)
                    for name, p in (("rf_" + alias, rf), ("blend_" + alias, mixed)):
                        rows.append(dict(key, candidate=name, auc=float(roc_auc_score(ctx["y"].iloc[va], p))))
                    diagnostics.append(dict(key, configuration=alias, rf_unique_probabilities=int(np.unique(rf).size),
                        rf_probability_sd=float(np.std(rf)), baseline_rf_rank_correlation=(
                            None if np.unique(rf).size < 2 or np.unique(base).size < 2 else
                            float(pd.Series(base).rank().corr(pd.Series(rf).rank()))),
                        mean_tree_depth=float(np.mean([t.tree_.max_depth for t in model.estimators_]))))
                    for j, i in enumerate(va):
                        predictions.append(dict(key, configuration=alias, row_index=int(i),
                            id=str(ctx["ids"].iloc[i]), target=int(ctx["y"].iloc[i]),
                            baseline=float(base[j]), rf=float(rf[j]), blend=float(mixed[j])))
                    manifest["completed_forest_fits"] += 1
                    pd.DataFrame(rows).to_csv(out / "fold_scores.csv", index=False)
                    pd.DataFrame(predictions).to_csv(out / "validation_probabilities.csv", index=False)
                    h._write(out / "diagnostics.json", diagnostics)
                    h._write(out / "run_manifest.json", manifest)
            self._unchanged(c)
            repeats, table, decision = summarize(pd.DataFrame(rows))
            repeats.to_csv(out / "repeat_scores.csv", index=False)
            table.to_csv(out / "comparison.csv", index=False)
            h._write(out / "decision.json", decision)
            manifest.update(status="completed", baseline_oof_verified=True, promote=decision["promote"])
            manifest["artifact_sha256"] = {p.name: h._sha(p) for p in out.iterdir()
                if p.is_file() and p.name != "run_manifest.json"}
            return decision
        except BaseException as exc:
            manifest.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
            h._write(out / "error.json", dict(error=manifest["error"], traceback=traceback.format_exc()))
            raise
        finally:
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            h._write(out / "run_manifest.json", manifest)
