"""Automatic bounded max_features search; verified RF V1 control, CV only."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.experiments import rf_stabilizer as v1

TOL = 1e-12
KEYS = ["repeat", "split_seed", "fold", "row_index"]
FEATURES = [("sqrt", "sqrt"), ("fraction020", .2), ("fraction050", .5)]
BASELINE = "trial14_three_model"
RF_FIXED = dict(n_estimators=500, max_depth=3, min_samples_leaf=5,
                bootstrap=True, random_state=42, n_jobs=1, class_weight=None)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(cfg):
    expected = {"v1_config", "reference_v1_run", "output_dir", "minimum_mean_gain"}
    if not isinstance(cfg, dict) or set(cfg) != expected:
        raise ValueError(f"Configuration requires exactly {sorted(expected)}")
    for key in expected - {"minimum_mean_gain"}:
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError("Paths must be nonempty strings.")
    gain = cfg["minimum_mean_gain"]
    if type(gain) not in (int, float) or not np.isfinite(gain) or not 0 <= gain <= .01:
        raise ValueError("minimum_mean_gain must be finite in [0,.01]; declare before execution.")
    return cfg


def fingerprint(paths):
    return {str(p): (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths}


def check_metadata(expected):
    if fingerprint(expected) != expected:
        raise RuntimeError("A source file changed during fitting. Preserve this failed run.")


def full_check(c):
    for path, digest in c["hashes"].items():
        if c["helper"]._sha(path) != digest:
            raise RuntimeError(f"Input/code/artifact changed: {path}")


def load_reference(c, root):
    """Validate V1 outputs; never trust the winner row alone."""
    h, ctx = c["helper"], c["context"]
    root = h._path(root)
    manifest, paths = v1.verify_artifacts(root, h, {
        "settings.json", "decision.json", "comparison.csv", "fold_scores.csv", "validation_probabilities.csv"})
    if (manifest.get("completed_forest_fits") != 200 or manifest.get("baseline_oof_verified") is not True
        or manifest.get("test_opened") is not False):
        raise ValueError("Need the original complete 200-fit RF V1 study.")
    settings = read(root / "settings.json")
    fixed = dict(n_estimators=500, max_features="sqrt", bootstrap=True, random_state=42, n_jobs=1)
    grid = [dict(max_depth=d, min_samples_leaf=l) for d, l in [(3,5),(3,10),(5,5),(5,10)]]
    if (settings["input_and_code_sha256"] != c["hashes"] or settings["versions"] != c["versions"]
        or settings["config"] != c["config"] or settings.get("fixed_rf") != fixed
        or settings.get("grid") != grid or settings.get("baseline_weight") != .9 or settings.get("rf_weight") != .1):
        raise ValueError("V1 provenance/settings/environment differ. Do not edit historical hashes.")
    decision = read(root / "decision.json")
    if decision.get("selected") != "blend_d3_leaf5" or decision.get("promote") is not True:
        raise ValueError("This follow-up is locked to the reported V1 depth3/leaf5 winner.")
    p = pd.read_csv(root / "validation_probabilities.csv", dtype={"id": "string"}, float_precision="round_trip")
    needed = set(KEYS + ["configuration", "id", "target", "baseline", "rf", "blend"])
    if not needed <= set(p) or len(p) != 10000 or p.duplicated(["configuration"] + KEYS).any():
        raise ValueError("Invalid V1 OOF shape/keys.")
    expected_aliases = {"d3_leaf5", "d3_leaf10", "d5_leaf5", "d5_leaf10"}
    if set(p.configuration) != expected_aliases or not (p.groupby("configuration").size() == 2500).all():
        raise ValueError("Incomplete V1 configuration predictions.")
    for key in KEYS:
        values = p[key].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not (values == np.floor(values)).all():
            raise ValueError("V1 OOF keys must be finite integers.")
    cached = p.loc[p.configuration == "d3_leaf5"].set_index(KEYS)
    if not cached.index.is_unique:
        raise ValueError("Duplicate selected V1 OOF keys.")
    folds = pd.read_csv(root / "fold_scores.csv", float_precision="round_trip")
    if (len(folds) != 450 or folds.duplicated(["candidate", "repeat", "fold"]).any()
        or set(folds.candidate) != {BASELINE, *[f"{kind}_{a}" for a in expected_aliases for kind in ("rf", "blend")]}):
        raise ValueError("Invalid V1 fold-score table.")
    indexed_scores = folds.set_index(["candidate", "repeat", "fold"])
    if not indexed_scores.index.is_unique:
        raise ValueError("Duplicate V1 fold-score keys.")
    expected_keys, rows = [], []
    for s in ctx["splits"]:
        va = s["valid_indices"]
        keys = [(s["repeat"], s["split_seed"], s["fold"], int(i)) for i in va]
        expected_keys.extend(keys)
        f = cached.loc[keys]
        if (f.id.astype(str).tolist() != ctx["ids"].iloc[va].astype(str).tolist()
            or not np.array_equal(f.target, ctx["y"].iloc[va])):
            raise ValueError("V1 OOF IDs/labels changed.")
        base, rf, mixed = [v1.probability(f[n].to_numpy(), len(va)) for n in ("baseline", "rf", "blend")]
        if not np.allclose(base, c["oof"].loc[keys, "l1_xgb_lgbm"], rtol=0, atol=TOL):
            raise ValueError("V1 baseline differs from verified Trial14 OOF.")
        if not np.allclose(mixed, v1.blend(base, rf), rtol=0, atol=TOL):
            raise ValueError("V1 blend weights/probabilities changed.")
        key = {n: s[n] for n in ("repeat", "split_seed", "fold")}
        for old, new, prob in ((BASELINE, BASELINE, base), ("rf_d3_leaf5", "rf_sqrt", rf), ("blend_d3_leaf5", "blend_sqrt", mixed)):
            record = indexed_scores.loc[(old, s["repeat"], s["fold"])]
            auc = float(roc_auc_score(ctx["y"].iloc[va], prob))
            if record.split_seed != s["split_seed"] or not np.isclose(record.auc, auc, rtol=0, atol=TOL):
                raise ValueError("V1 OOF/fold AUC mismatch.")
            rows.append(dict(key, candidate=new, auc=auc))
    if set(cached.index) != set(expected_keys):
        raise ValueError("V1 validation assignments changed.")
    table = pd.read_csv(root / "comparison.csv", float_precision="round_trip")
    row = table.loc[table.candidate == "blend_d3_leaf5"]
    mean = pd.DataFrame(rows).query("candidate == 'blend_sqrt'").groupby(["repeat", "split_seed"]).auc.mean().mean()
    if len(row) != 1 or not np.isclose(row.iloc[0].mean_auc, mean, rtol=0, atol=TOL):
        raise ValueError("V1 control mean AUC did not reproduce.")
    predictions = cached.reset_index().copy()
    predictions["configuration"] = "sqrt"
    return dict(paths=paths, control_rows=rows, predictions=predictions)


def summarize(folds, minimum_gain):
    names = [BASELINE] + [f"{kind}_{tag}" for tag, _ in FEATURES for kind in ("rf", "blend")]
    if (set(folds.candidate) != set(names) or len(folds) != 350
        or folds.duplicated(["candidate", "repeat", "fold"]).any()):
        raise ValueError("Need all seven procedures on all fifty folds.")
    values = folds.auc.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid AUC values.")
    keys = ["repeat", "split_seed", "fold"]
    baseline_keys = set(folds.loc[folds.candidate == BASELINE, keys].itertuples(index=False, name=None))
    for _, f in folds.groupby("candidate"):
        if (set(f[keys].itertuples(index=False, name=None)) != baseline_keys
            or f.groupby("repeat").size().tolist() != [5] * 10
            or f.groupby("repeat").split_seed.nunique().max() != 1):
            raise ValueError("Incomplete or unpaired folds.")
    repeats = folds.groupby(["candidate", "repeat", "split_seed"], as_index=False).auc.mean()
    def series(name):
        return repeats.loc[repeats.candidate == name].set_index(["repeat", "split_seed"]).auc.sort_index()
    base, control = series(BASELINE), series("blend_sqrt")
    rows = []
    for name in names:
        r = series(name)
        d, dc = r - base, r - control
        rows.append(dict(candidate=name, mean_auc=float(r.mean()), repeat_sd=float(r.std(ddof=1)),
            minimum_repeat_auc=float(r.min()), maximum_repeat_auc=float(r.max()),
            difference_vs_baseline=float(d.mean()), difference_sd_vs_baseline=float(d.std(ddof=1)),
            wins_vs_baseline=int((d > TOL).sum()), losses_vs_baseline=int((d < -TOL).sum()), ties_vs_baseline=int((abs(d) <= TOL).sum()),
            difference_vs_v1=float(dc.mean()), difference_sd_vs_v1=float(dc.std(ddof=1)),
            wins_vs_v1=int((dc > TOL).sum()), losses_vs_v1=int((dc < -TOL).sum()), ties_vs_v1=int((abs(dc) <= TOL).sum()),
            stability_observed=bool(name.startswith("blend_") and r.std(ddof=1) <= base.std(ddof=1) + TOL and r.min() >= base.min() - TOL)))
    table = pd.DataFrame(rows)
    blends = table.loc[table.candidate.str.startswith("blend_")]
    winner = blends.loc[blends.mean_auc.idxmax()]  # exact ties: FEATURES order
    new = winner.candidate != "blend_sqrt"
    accepted = bool(new and winner.difference_vs_baseline > TOL and winner.wins_vs_baseline >= 6
        and winner.difference_vs_v1 > minimum_gain + TOL and winner.wins_vs_v1 >= 6)
    decision = dict(baseline=BASELINE, rf_v1_control="blend_sqrt", best_candidate=str(winner.candidate),
        selected_new_candidate=str(winner.candidate) if accepted else None, promote_new_configuration=accepted,
        minimum_mean_gain_vs_v1=minimum_gain, stability_observed=bool(winner.stability_observed),
        selection_rule="Highest blend mean AUC, exact ties use declared max_features order. No fallback.",
        promotion_rule="Winner must be new; positive mean and >=6 wins vs Trial14; gain > minimum_mean_gain and >=6 wins vs V1. Pragmatic gate, not significance test.",
        interpretation="Adaptive development on reused splits; not independent evaluation or proof of private performance.",
        next_action="Manual review only. No final refit, competition test reading, or upload implemented.")
    return repeats, table, decision


class RFAutoSearch:
    def _prepare(self, config_path, output=None):
        from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission
        h = FrozenXGBoostSubmission()
        path = h._path(config_path)
        cfg = validate(read(path))
        out = h._path(output or cfg["output_dir"])
        if out.exists():
            raise FileExistsError("Choose a NEW output directory; no overwrite or resume.")
        source = v1.RFStabilizer()._prepare(cfg["v1_config"], str(out))
        reference = load_reference(source, cfg["reference_v1_run"])
        paths = reference["paths"] + [path, Path(__file__), h._path("scripts/run_rf_auto.py")]
        hashes = dict(source["hashes"])
        hashes.update({str(p.resolve()): h._sha(p) for p in paths})
        return dict(helper=h, context=source["context"], config=cfg, output=out, versions=source["versions"],
            hashes=hashes, reference=reference, source=source)

    def run(self, config_path, output=None, dry_run=False):
        c = self._prepare(config_path, output)
        full_check(c)
        stamps = fingerprint(c["hashes"])
        if dry_run:
            return dict(status="preflight_passed", max_features=[v for _, v in FEATURES],
                new_forest_fits=100, trees_per_forest=500, reused_control="V1 depth3/leaf5 sqrt (50 folds)",
                rf_weight=.1, minimum_mean_gain_vs_v1=c["config"]["minimum_mean_gain"],
                test_opened=False, output=str(c["output"]))
        h, out, ctx = c["helper"], c["output"], c["context"]
        out.mkdir(parents=True, exist_ok=False)
        manifest = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
            planned_forest_fits=100, attempted_forest_fits=0, completed_forest_fits=0,
            reused_control_folds=50, test_opened=False, kaggle_upload_performed=False)
        rows = list(c["reference"]["control_rows"])
        frames = [c["reference"]["predictions"]]
        warning_rows, diagnostics = [], []
        started = time.monotonic()
        h._write(out / "run_manifest.json", manifest)
        try:
            h._write(out / "settings.json", dict(config=c["config"], versions=c["versions"],
                input_and_code_sha256=c["hashes"], max_features=[v for _, v in FEATURES],
                fixed_rf=RF_FIXED, rf_weight=.1, baseline_weight=.9,
                hash_policy="Full SHA256 at preflight/start/end; size+mtime before and after every new fit.",
                control_reuse="V1 selected depth3/leaf5 sqrt OOF, verified by full provenance and fold AUC checks."))
            h._write(out / "warnings.json", [])
            for tag, feature in FEATURES[1:]:
                new_rows = []
                for s in ctx["splits"]:
                    check_metadata(stamps)
                    tr, va = s["train_indices"], s["valid_indices"]
                    key = {n: s[n] for n in ("repeat", "split_seed", "fold")}
                    keys = [(s["repeat"], s["split_seed"], s["fold"], int(i)) for i in va]
                    base = c["source"]["oof"].loc[keys, "l1_xgb_lgbm"].to_numpy()
                    model = RandomForestClassifier(**RF_FIXED, max_features=feature)
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
                                rf = v1.probability(model.predict_proba(ctx["x"].iloc[va])[:, 1], len(va))
                    finally:
                        warning_rows.extend(dict(key, configuration=tag, category=w.category.__name__, message=str(w.message)) for w in caught)
                        h._write(out / "warnings.json", warning_rows)
                    if caught:
                        raise RuntimeError("Python warnings require review; entire search stopped.")
                    check_metadata(stamps)
                    mixed = v1.blend(base, rf)
                    for name, p in (("rf_" + tag, rf), ("blend_" + tag, mixed)):
                        rows.append(dict(key, candidate=name, auc=float(roc_auc_score(ctx["y"].iloc[va], p))))
                    for j, i in enumerate(va):
                        new_rows.append(dict(key, configuration=tag, row_index=int(i), id=str(ctx["ids"].iloc[i]),
                            target=int(ctx["y"].iloc[i]), baseline=float(base[j]), rf=float(rf[j]), blend=float(mixed[j])))
                    correlation = None if np.unique(base).size < 2 or np.unique(rf).size < 2 else float(pd.Series(base).rank().corr(pd.Series(rf).rank()))
                    diagnostics.append(dict(key, configuration=tag, unique_rf_probabilities=int(np.unique(rf).size),
                        baseline_rf_rank_correlation=correlation))
                    manifest["completed_forest_fits"] += 1
                    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
                    h._write(out / "run_manifest.json", manifest)
                    done = manifest["completed_forest_fits"]
                    print(f"[{done}/100] max_features={feature} repeat={s['repeat']} fold={s['fold']} elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)
                frames.append(pd.DataFrame(new_rows))
                new_rows = []
                pd.DataFrame(rows).to_csv(out / "fold_scores.csv", index=False)
                pd.concat(frames, ignore_index=True).to_csv(out / "validation_probabilities.csv", index=False)
                h._write(out / "diagnostics.json", diagnostics)
            full_check(c)
            repeats, table, decision = summarize(pd.DataFrame(rows), c["config"]["minimum_mean_gain"])
            repeats.to_csv(out / "repeat_scores.csv", index=False)
            table.to_csv(out / "comparison.csv", index=False)
            h._write(out / "decision.json", decision)
            manifest.update(status="completed", source_oof_verified=True,
                promote_new_configuration=decision["promote_new_configuration"])
            manifest["artifact_sha256"] = {p.name: h._sha(p) for p in out.iterdir() if p.is_file() and p.name != "run_manifest.json"}
            return decision
        except BaseException as exc:
            manifest.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
            # Persist available partial results for diagnosis, not candidate selection.
            pd.DataFrame(rows).to_csv(out / "partial_fold_scores.csv", index=False)
            if "new_rows" in locals() and new_rows:
                partial = frames + [pd.DataFrame(new_rows)]
            else:
                partial = frames
            pd.concat(partial, ignore_index=True).to_csv(out / "partial_validation_probabilities.csv", index=False)
            h._write(out / "diagnostics.json", diagnostics)
            h._write(out / "error.json", dict(error=manifest["error"], traceback=traceback.format_exc()))
            raise
        finally:
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            h._write(out / "run_manifest.json", manifest)
