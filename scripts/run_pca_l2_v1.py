"""Bounded PCA/L2 development comparison; never loads competition test data."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from inspect import signature
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve(root, value):
    return (root / Path(value).expanduser()).resolve()


def validate_config(cfg):
    required = {"train_path", "target_column", "id_column", "expected_rows", "expected_features",
                "source_run", "output_dir", "c_values", "components", "max_iter", "tol", "model_seed"}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError("Unexpected config fields")
    for key in ("train_path", "target_column", "id_column", "source_run", "output_dir"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError("Nonempty paths/column names required")
    if cfg["id_column"] == cfg["target_column"]:
        raise ValueError("ID and target must differ")
    for key in ("expected_rows", "expected_features", "max_iter"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError("Positive integer required")
    if type(cfg["model_seed"]) is not int or type(cfg["tol"]) not in (int, float) or not 0 < cfg["tol"] < 1:
        raise ValueError("Invalid estimator seed/tolerance")
    cs, ks = cfg["c_values"], cfg["components"]
    if not cs or any(type(c) not in (float, int) or not np.isfinite(c) or c <= 0 for c in cs):
        raise ValueError("Positive finite C values required")
    if len(set(cs)) != len(cs) or cs != sorted(cs):
        raise ValueError("Unique ascending C values required")
    if not ks or any(type(k) is not int or k <= 0 for k in ks) or len(set(ks)) != len(ks) or ks != sorted(ks):
        raise ValueError("Unique ascending component counts required")
    return cfg


def load_train(path, cfg):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or any(not s.strip() for s in header) or len(set(header)) != len(header):
        raise ValueError("Blank/duplicate CSV header")
    frame = pd.read_csv(path, dtype={cfg["id_column"]: "string"})
    if not {cfg["id_column"], cfg["target_column"]} <= set(frame):
        raise ValueError("Explicit ID/target missing; edit config, never infer from test.csv")
    ids, y = frame[cfg["id_column"]], frame[cfg["target_column"]]
    x = frame.drop(columns=[cfg["id_column"], cfg["target_column"]])
    if x.shape != (cfg["expected_rows"], cfg["expected_features"]):
        raise ValueError("Unexpected training shape")
    if ids.isna().any() or not ids.is_unique or ids.str.strip().eq("").any():
        raise ValueError("Invalid row IDs")
    if any(not pd.api.types.is_numeric_dtype(x[c]) or pd.api.types.is_bool_dtype(x[c]) for c in x):
        raise ValueError("Numeric predictors required")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite predictors")
    if pd.api.types.is_bool_dtype(y) or y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Numeric binary target required")
    return x, y.astype(int), ids


def validate_splits(records, y, cfg):
    if len(records) != 50:
        raise ValueError("Need all 50 saved cv5 splits")
    keys = [(s["repeat"], s["fold"]) for s in records]
    if len(set(keys)) != 50 or set(keys) != {(r, f) for r in range(10) for f in range(5)}:
        raise ValueError("Incomplete/duplicate split keys")
    n, universe = len(y), set(range(len(y)))
    for repeat in range(10):
        seen = []
        for s in [s for s in records if s["repeat"] == repeat]:
            tr, va = s["train_indices"], s["valid_indices"]
            if s["scheme"] != "cv5" or s["split_seed"] != repeat:
                raise ValueError("Unexpected split scheme/seed")
            if any(type(i) is not int for i in tr + va):
                raise ValueError("Integer indices required")
            if len(tr) != len(set(tr)) or len(va) != len(set(va)) or set(tr) & set(va) or set(tr) | set(va) != universe:
                raise ValueError("Invalid split partition")
            if len(va) != n // 5 or len(tr) != n - n // 5:
                raise ValueError("Unexpected five-fold sizes")
            if y.iloc[tr].nunique() != 2 or y.iloc[va].nunique() != 2:
                raise ValueError("Both classes required in every partition")
            if max(cfg["components"]) > min(len(tr) - 1, cfg["expected_features"]):
                raise ValueError("Component count exceeds centered training rank bound")
            seen.extend(va)
        if sorted(seen) != list(range(n)):
            raise ValueError("Each row must be validated once per repeat")
    return sorted(records, key=lambda s: (s["repeat"], s["fold"]))


def prepare(config_path, root=ROOT, output=None):
    path = resolve(root, config_path)
    cfg = validate_config(read(path))
    train = resolve(root, cfg["train_path"])
    source = resolve(root, cfg["source_run"])
    dest = resolve(root, output or cfg["output_dir"])
    if dest.exists():
        raise FileExistsError("Use a fresh output directory; no overwrite/resume")
    manifest_path = source / "run_manifest.json"
    manifest = read(manifest_path)
    checks = manifest.get("artifact_sha256", {})
    if manifest.get("status") != "completed" or not {"settings.json", "split_indices.json"} <= checks.keys():
        raise ValueError("Completed validation_protocol_v1 source with checksums required")
    watched = [path, train, manifest_path, Path(__file__).resolve()]
    for name, expected in checks.items():
        p = (source / name).resolve()
        if not p.is_relative_to(source) or not p.is_file() or digest(p) != expected:
            raise ValueError("Changed/unsafe source artifact: " + name)
        watched.append(p)
    settings = read(source / "settings.json")
    historical = settings.get("input_and_code_sha256", {})
    train_suffix = Path(cfg["train_path"]).as_posix()
    matches = [v for k, v in historical.items()
               if Path(k).as_posix() == train.as_posix() or Path(k).as_posix().endswith("/" + train_suffix)]
    if not matches or len(set(matches)) != 1 or matches[0] != digest(train):
        raise ValueError("Training CSV does not match the source study SHA256")
    x, y, ids = load_train(train, cfg)
    splits = validate_splits([s for s in read(source / "split_indices.json") if s.get("scheme") == "cv5"], y, cfg)
    return cfg, dest, x, y, ids, splits, {str(p): digest(p) for p in watched}


def candidates(cfg):
    rows = []
    for k in [None] + cfg["components"]:
        for c in cfg["c_values"]:
            rows.append(dict(candidate=f"{'no_pca' if k is None else 'pca_' + str(k)}_C{c:g}",
                             branch="control" if k is None else "pca", components=k, C=c))
    return rows


def build_model(candidate, cfg):
    parameters = dict(C=candidate["C"], solver="lbfgs", max_iter=cfg["max_iter"],
                      tol=cfg["tol"], random_state=cfg["model_seed"])
    penalty = signature(LogisticRegression).parameters.get("penalty")
    if penalty is None or penalty.default == "deprecated":
        parameters["l1_ratio"] = 0.0
    else:
        parameters["penalty"] = "l2"
    steps = [("scaler", StandardScaler())]
    if candidate["components"] is not None:
        steps.append(("pca", PCA(n_components=candidate["components"], svd_solver="full", whiten=False)))
    steps.append(("classifier", LogisticRegression(**parameters)))
    return Pipeline(steps)


def compare(repeats, b, a):
    left = repeats[repeats.candidate == a].set_index("repeat").mean_fold_auc.sort_index()
    right = repeats[repeats.candidate == b].set_index("repeat").mean_fold_auc.sort_index()
    if list(left.index) != list(range(10)) or not left.index.equals(right.index):
        raise ValueError("Unpaired repeat scores")
    d = right - left
    detail = pd.DataFrame(dict(repeat=d.index, reference=a, candidate=b, difference=d.to_numpy()))
    summary = dict(reference=a, candidate=b, mean_difference=float(d.mean()), difference_sd=float(d.std(ddof=1)),
                   wins=int((d > TOL).sum()), losses=int((d < -TOL).sum()), ties=int((abs(d) <= TOL).sum()))
    return detail, summary


def summarize(folds, specs):
    expected = {(s["candidate"], r, f) for s in specs for r in range(10) for f in range(5)}
    keys = list(folds[["candidate", "repeat", "fold"]].itertuples(index=False, name=None))
    if len(keys) != len(expected) or set(keys) != expected or not np.isfinite(folds.auc).all():
        raise ValueError("Missing/duplicate/invalid fold scores")
    repeats = folds.groupby(["candidate", "repeat"], as_index=False).auc.mean().rename(columns={"auc": "mean_fold_auc"})
    summary = repeats.groupby("candidate", sort=False).mean_fold_auc.agg(mean_auc="mean", repeat_sd="std").reset_index()
    summary = pd.DataFrame(specs).merge(summary, on="candidate", validate="one_to_one")
    winners = {}
    for branch in ("control", "pca"):
        part = summary[summary.branch == branch]
        winners[branch] = part.loc[part.mean_auc.idxmax(), "candidate"]
    details, comparisons = [], []
    for s in specs:
        if s["branch"] != "pca":
            continue
        same_c = next(a["candidate"] for a in specs if a["branch"] == "control" and a["C"] == s["C"])
        d, stat = compare(repeats, s["candidate"], same_c)
        d["comparison"] = "same_C"; stat["comparison"] = "same_C"
        details.append(d); comparisons.append(stat)
    d, stat = compare(repeats, winners["pca"], winners["control"])
    d["comparison"] = "selected_branches"; stat["comparison"] = "selected_branches"
    details.append(d); comparisons.append(stat)
    decision = dict(selected_control=winners["control"], selected_pca=winners["pca"],
                    selection_rule="Highest mean per branch; exact ties use config order (control: smaller C; PCA: smaller k then C). No fallback.",
                    screening_passed=bool(stat["mean_difference"] > TOL and stat["wins"] >= 6),
                    automatic_promotion=False, **stat,
                    interpretation="Adaptive development comparison, not nested CV or an independent generalisation estimate. Screening is descriptive, not a significance test. No incumbent ensemble changed.")
    return repeats, summary, pd.concat(details, ignore_index=True), pd.DataFrame(comparisons), decision


def run(config_path, output=None, dry_run=False, root=ROOT):
    cfg, dest, x, y, ids, splits, hashes = prepare(config_path, root, output)
    specs = candidates(cfg)
    plan = dict(status="preflight_passed", candidates=len(specs), splits=50,
                planned_pipeline_fits=len(specs) * 50, test_opened=False, output=str(dest))
    if dry_run:
        return plan
    dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(plan, status="running", completed_pipeline_fits=0,
                    started_utc=datetime.now(timezone.utc).isoformat(), kaggle_upload_performed=False)
    rows, predictions, diagnostics, warning_rows = [], [], [], []
    write(dest / "run_manifest.json", manifest)
    try:
        write(dest / "settings.json", dict(config=cfg, candidates=specs, input_and_code_sha256=hashes,
              versions={p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl")},
              preprocessing="Training-fitted StandardScaler then optional full-SVD PCA, whiten=False; L2 lbfgs for both branches"))
        write(dest / "split_indices.json", splits)
        with threadpool_limits(limits=1):
            for s in splits:
                tr, va = s["train_indices"], s["valid_indices"]
                for spec in specs:
                    model = build_model(spec, cfg)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model.fit(x.iloc[tr], y.iloc[tr])
                    key = dict(candidate=spec["candidate"], repeat=s["repeat"], split_seed=s["split_seed"], fold=s["fold"])
                    warning_rows.extend(dict(key, category=w.category.__name__, message=str(w.message)) for w in caught)
                    write(dest / "warnings.json", warning_rows)
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        raise RuntimeError("ConvergenceWarning: review failed run; do not select partial results")
                    p = model.predict_proba(x.iloc[va])[:, 1]
                    if p.shape != (len(va),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                        raise ValueError("Invalid probabilities")
                    rows.append(dict(key, auc=float(roc_auc_score(y.iloc[va], p)), train_rows=len(tr), valid_rows=len(va)))
                    predictions.extend(dict(key, row_index=int(i), id=str(ids.iloc[i]), target=int(y.iloc[i]), probability=float(q)) for i, q in zip(va, p))
                    if "pca" in model.named_steps:
                        pca = model.named_steps["pca"]
                        diagnostics.append(dict(key, explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
                                                explained_variance_sum=float(pca.explained_variance_ratio_.sum())))
                    manifest["completed_pipeline_fits"] += 1
                pd.DataFrame(rows).to_csv(dest / "fold_scores.csv", index=False)
                write(dest / "run_manifest.json", manifest)
                print(f"[{manifest['completed_pipeline_fits']}/{plan['planned_pipeline_fits']}] repeat={s['repeat']} fold={s['fold']}", flush=True)
        for path, expected in hashes.items():
            if digest(path) != expected:
                raise RuntimeError("Input/source/code changed during run: " + path)
        repeats, summary, paired, comparisons, decision = summarize(pd.DataFrame(rows), specs)
        for name, table in (("repeat_scores", repeats), ("model_summary", summary),
                            ("paired_repeat_differences", paired), ("comparison", comparisons)):
            table.to_csv(dest / (name + ".csv"), index=False)
        pd.DataFrame(predictions).to_csv(dest / "validation_probabilities.csv", index=False)
        write(dest / "pca_diagnostics.json", diagnostics)
        write(dest / "decision.json", decision)
        manifest["status"] = "completed"
        manifest["artifact_sha256"] = {p.name: digest(p) for p in dest.iterdir() if p.is_file() and p.name != "run_manifest.json"}
        return dict(status="completed", output=str(dest), automatic_promotion=False)
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        pd.DataFrame(rows).to_csv(dest / "partial_fold_scores.csv", index=False)
        write(dest / "warnings.json", warning_rows)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(dest / "run_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pca_l2_v1.json")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.dry_run), indent=2))


if __name__ == "__main__":
    main()
