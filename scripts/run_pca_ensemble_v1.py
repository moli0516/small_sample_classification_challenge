"""Read-only-source PCA/L2 ensemble sensitivity experiment. Zero model fits."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-12
KEYS = ["repeat", "fold", "row_index"]
BASE = "l1_xgb_lgbm"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def table(path):
    return pd.read_csv(path, dtype={"id": "string"}, float_precision="round_trip")


def source(root, required):
    root = root.resolve()
    manifest_path = root / "run_manifest.json"
    manifest = read(manifest_path)
    hashes = manifest.get("artifact_sha256", {})
    if manifest.get("status") != "completed" or not required <= hashes.keys():
        raise ValueError("Completed source with required artifact hashes needed: " + str(root))
    watched = {str(manifest_path): digest(manifest_path)}
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError("Changed or unsafe source artifact: " + name)
        watched[str(path)] = expected
    return watched


def config(path):
    c = read(path)
    expected = {"ensemble_run", "pca_run", "output_dir", "pca_candidate", "l2_candidate", "alphas", "expected_rows", "expected_reference_auc"}
    if set(c) != expected:
        raise ValueError("Unexpected configuration fields")
    if any(not isinstance(c[k], str) or not c[k].strip() for k in ("ensemble_run", "pca_run", "output_dir", "pca_candidate", "l2_candidate")):
        raise ValueError("Nonempty paths/candidate names required")
    if c["alphas"] != [.05, .10, .25] or any(type(a) not in (float, int) for a in c["alphas"]):
        raise ValueError("V1 fixes alpha to 0.05/0.10/0.25; change experiment version for another design")
    if type(c["expected_rows"]) is not int or c["expected_rows"] <= 0 or c["expected_rows"] % 5:
        raise ValueError("Positive row count divisible by five required")
    if not isinstance(c["expected_reference_auc"], (int, float)) or not np.isfinite(c["expected_reference_auc"]) or not 0 <= c["expected_reference_auc"] <= 1:
        raise ValueError("Invalid reference AUC")
    return c


def saved_splits(path, n):
    splits = [s for s in read(path) if s.get("scheme") == "cv5"]
    keys = [(s["repeat"], s["fold"]) for s in splits]
    if len(keys) != 50 or set(keys) != {(r, f) for r in range(10) for f in range(5)}:
        raise ValueError("Exactly 50 distinct cv5 splits required")
    for r in range(10):
        seen = []
        for s in [s for s in splits if s["repeat"] == r]:
            tr, va = s["train_indices"], s["valid_indices"]
            if s["split_seed"] != r or any(type(i) is not int for i in tr + va):
                raise ValueError("Bad seed/index types")
            if len(tr) != n - n // 5 or len(va) != n // 5 or len(set(tr)) != len(tr) or len(set(va)) != len(va):
                raise ValueError("Bad partition sizes/duplicates")
            if set(tr) & set(va) or set(tr) | set(va) != set(range(n)):
                raise ValueError("Split overlap or missing rows")
            seen.extend(va)
        if sorted(seen) != list(range(n)):
            raise ValueError("Validation coverage is not once per repeat")
    return sorted(splits, key=lambda s: (s["repeat"], s["fold"]))


def probability(values):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError("Invalid probability")


def aligned(frame, splits, n, columns):
    required = set(KEYS + ["split_seed", "id", "target"] + columns)
    if not required <= set(frame) or frame.duplicated(KEYS).any() or len(frame) != 10 * n:
        raise ValueError("Missing columns, duplicate prediction keys, or incomplete predictions")
    if frame.id.isna().any() or frame.id.str.strip().eq("").any() or frame.target.isna().any() or set(frame.target.unique()) != {0, 1}:
        raise ValueError("Invalid ID/target")
    if any(not pd.api.types.is_integer_dtype(frame[c]) for c in KEYS + ["split_seed", "target"]):
        raise ValueError("Integer row/split/target fields required")
    byrow = frame.groupby("row_index")
    if set(byrow.groups) != set(range(n)) or (byrow[["id", "target"]].nunique() != 1).any().any():
        raise ValueError("Row ID/label inconsistent across repeats")
    if frame.drop_duplicates("row_index").id.nunique() != n:
        raise ValueError("IDs not unique across rows")
    for s in splits:
        part = frame[(frame.repeat == s["repeat"]) & (frame.fold == s["fold"])]
        if set(part.row_index) != set(s["valid_indices"]) or not (part.split_seed == s["split_seed"]).all() or part.target.nunique() != 2:
            raise ValueError("Prediction rows do not match saved split")
    for col in columns:
        probability(frame[col])
    return frame.sort_values(KEYS).reset_index(drop=True)


def score(frame, columns):
    rows = []
    for (r, f), group in frame.groupby(["repeat", "fold"], sort=True):
        for name in columns:
            rows.append(dict(candidate=name, repeat=int(r), fold=int(f), auc=float(roc_auc_score(group.target, group[name]))))
    folds = pd.DataFrame(rows)
    repeats = folds.groupby(["candidate", "repeat"], as_index=False).auc.mean().rename(columns={"auc": "mean_fold_auc"})
    summary = repeats.groupby("candidate").mean_fold_auc.agg(mean_auc="mean", repeat_sd="std").reset_index()
    return folds, repeats, summary


def reproduce(actual, saved, names, field="auc"):
    expected = saved[saved.candidate.isin(names)].copy()
    keys = ["candidate", "repeat", "fold"] if field == "auc" else ["candidate", "repeat"]
    if expected.duplicated(keys).any():
        raise ValueError("Duplicate source score keys")
    merged = actual[actual.candidate.isin(names)].merge(expected[keys + [field]], on=keys, how="outer", validate="one_to_one", suffixes=("", "_source"), indicator=True)
    if not (merged._merge == "both").all() or not np.allclose(merged[field], merged[field + "_source"], rtol=0, atol=TOL):
        raise ValueError("Source score reproduction failed: " + field)
    merged["difference"] = merged[field] - merged[field + "_source"]
    return merged.drop(columns="_merge")


def prepare(path, output=None, root=ROOT):
    path = (root / path).resolve()
    c = config(path)
    e, p = (root / c["ensemble_run"]).resolve(), (root / c["pca_run"]).resolve()
    dest = (root / (output or c["output_dir"])).resolve()
    if dest.exists() or any(dest.is_relative_to(src) or src.is_relative_to(dest) for src in (e, p)):
        raise ValueError("Fresh output outside source trees required")
    eh = source(e, {"settings.json", "split_indices.json", "validation_probabilities.csv", "fold_scores.csv"})
    ph = source(p, {"settings.json", "split_indices.json", "validation_probabilities.csv", "fold_scores.csv", "repeat_scores.csv", "model_summary.csv", "decision.json"})
    es, ps, decision = read(e / "settings.json"), read(p / "settings.json"), read(p / "decision.json")
    if (Path(ps["config"]["source_run"]).is_absolute() and Path(ps["config"]["source_run"]).resolve() != e) or (not Path(ps["config"]["source_run"]).is_absolute() and (root / ps["config"]["source_run"]).resolve() != e):
        raise ValueError("PCA source must be this exact ensemble validation study")
    train_suffix = Path(ps["config"]["train_path"]).as_posix()
    def train_hash(settings):
        found = [v for k, v in settings["input_and_code_sha256"].items() if Path(k).as_posix() == train_suffix or Path(k).as_posix().endswith("/" + train_suffix)]
        if not found or len(set(found)) != 1:
            raise ValueError("Unique training hash missing")
        return found[0]
    if train_hash(es) != train_hash(ps):
        raise ValueError("Training data hashes differ")
    for name, value in eh.items():
        if ps["input_and_code_sha256"].get(name) != value:
            raise ValueError("PCA did not record the same verified ensemble source")
    splits = saved_splits(e / "split_indices.json", c["expected_rows"])
    if splits != saved_splits(p / "split_indices.json", c["expected_rows"]):
        raise ValueError("Ordered training/validation splits differ")
    model_summary = table(p / "model_summary.csv")
    if model_summary.candidate.duplicated().any() or not np.isfinite(model_summary[["mean_auc", "repeat_sd"]].to_numpy()).all():
        raise ValueError("Invalid source model summary")
    selected = [c["pca_candidate"], c["l2_candidate"]]
    for branch, name, key in (("pca", selected[0], "selected_pca"), ("control", selected[1], "selected_control")):
        part = model_summary[model_summary.branch == branch]
        if part.empty or part.loc[part.mean_auc.idxmax(), "candidate"] != name or decision[key] != name:
            raise ValueError("Frozen candidate is not the source-selected branch winner")
    raw = table(e / "validation_probabilities.csv")
    ef = aligned(raw[raw.scheme == "cv5"].copy(), splits, c["expected_rows"], [BASE, "l1_c1", "xgboost", "lightgbm"])
    expected_base = (ef.l1_c1 + ef.xgboost + ef.lightgbm) / 3
    if not np.allclose(ef[BASE], expected_base, rtol=0, atol=TOL):
        raise ValueError("Original ensemble does not reproduce component averaging")
    base_folds, base_repeats, base_summary = score(ef, [BASE])
    reproduction = reproduce(base_folds, table(e / "fold_scores.csv").query("scheme == 'cv5'"), [BASE])
    if not np.isclose(base_summary.mean_auc.iloc[0], c["expected_reference_auc"], rtol=0, atol=TOL):
        raise ValueError("Frozen baseline mean AUC did not reproduce")
    rawp = table(p / "validation_probabilities.csv")
    combined = ef[KEYS + ["split_seed", "id", "target", BASE]].copy()
    pca_reproductions = []
    for name, alias in zip(selected, ("pca_l2", "no_pca_l2")):
        pf = aligned(rawp[rawp.candidate == name].copy(), splits, c["expected_rows"], ["probability"])
        if not pf[KEYS + ["split_seed", "id", "target"]].equals(ef[KEYS + ["split_seed", "id", "target"]]):
            raise ValueError("Prediction ID/label/split alignment mismatch")
        combined[alias] = pf.probability.to_numpy()
        ff, rr, ss = score(pf.rename(columns={"probability": name}), [name])
        pca_reproductions.append(reproduce(ff, table(p / "fold_scores.csv"), [name]))
        reproduce(rr, table(p / "repeat_scores.csv"), [name], field="mean_fold_auc")
        source_summary = model_summary[model_summary.candidate == name].iloc[0]
        if not np.allclose(ss[["mean_auc", "repeat_sd"]].iloc[0], source_summary[["mean_auc", "repeat_sd"]].astype(float), rtol=0, atol=TOL):
            raise ValueError("Selected candidate summary did not reproduce")
    hashes = dict(eh, **ph)
    hashes.update({str(path): digest(path), str(Path(__file__).resolve()): digest(__file__)})
    return c, dest, combined, splits, hashes, pd.concat([reproduction] + pca_reproductions, ignore_index=True)


def paired(repeats, b, a, kind, alpha=None):
    first = repeats[repeats.candidate == a].set_index("repeat").mean_fold_auc.sort_index()
    second = repeats[repeats.candidate == b].set_index("repeat").mean_fold_auc.sort_index()
    if list(first.index) != list(range(10)) or not first.index.equals(second.index):
        raise ValueError("Unpaired scores")
    d = second - first
    detail = pd.DataFrame(dict(repeat=d.index, reference=a, candidate=b, difference=d.to_numpy(), comparison=kind))
    stat = dict(reference=a, candidate=b, comparison=kind, alpha=alpha, mean_difference=float(d.mean()), difference_sd=float(d.std(ddof=1)), wins=int((d > TOL).sum()), losses=int((d < -TOL).sum()), ties=int((abs(d) <= TOL).sum()))
    return detail, stat


def evaluate(frame, c):
    frame = frame.copy()
    names = [BASE, "pca_l2", "no_pca_l2"]
    blend_names = []
    for branch in ("pca_l2", "no_pca_l2"):
        for a in c["alphas"]:
            name = f"{branch}_alpha{a:g}"
            frame[name] = (1 - a) * frame[BASE] + a * frame[branch]
            names.append(name); blend_names.append(name)
    folds, repeats, summary = score(frame, names)
    details, stats = [], []
    for branch in ("pca_l2", "no_pca_l2"):
        for a in c["alphas"]:
            detail, stat = paired(repeats, f"{branch}_alpha{a:g}", BASE, "blend_vs_baseline", a)
            details.append(detail); stats.append(stat)
    for a in c["alphas"]:
        detail, stat = paired(repeats, f"pca_l2_alpha{a:g}", f"no_pca_l2_alpha{a:g}", "pca_vs_l2_same_alpha", a)
        details.append(detail); stats.append(stat)
    indexed = summary.set_index("candidate")
    winner = max(blend_names, key=lambda name: indexed.loc[name, "mean_auc"])
    best = next(s for s in stats if s["candidate"] == winner and s["comparison"] == "blend_vs_baseline")
    decision = dict(selected_candidate=winner, selected_mean_auc=float(indexed.loc[winner, "mean_auc"]),
                    selection_rule="Highest mean across all six blends; exact ties: PCA first, then ascending alpha, then no-PCA. No fallback.",
                    screening_passed=bool(best["mean_difference"] > TOL and best["wins"] >= 6),
                    automatic_promotion=False, fits_performed=0, test_opened=False, **best,
                    interpretation="Adaptive development comparison after prior tuning; not nested CV, independent evaluation, or a significance test. Screening does not authorize a submission or prove PCA-specific benefit.")
    return frame, folds, repeats, summary, pd.concat(details, ignore_index=True), pd.DataFrame(stats), decision


def run(path, output=None, dry_run=False, root=ROOT):
    c, dest, frame, splits, hashes, reproduction = prepare(path, output, root)
    plan = dict(status="preflight_passed", output=str(dest), blends=6, folds=50, fits_performed=0,
                test_opened=False, baseline_and_components_reproduced=True, kaggle_upload_performed=False)
    if dry_run:
        return plan
    dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(plan, status="running", started_utc=datetime.now(timezone.utc).isoformat())
    write(dest / "run_manifest.json", manifest)
    try:
        write(dest / "settings.json", dict(config=c, input_and_code_sha256=hashes,
              versions={p: version(p) for p in ("numpy", "pandas", "scikit-learn")}))
        write(dest / "split_indices.json", splits)
        frame, folds, repeats, summary, details, comparison, decision = evaluate(frame, c)
        for name, df in (("validation_probabilities", frame), ("fold_scores", folds), ("repeat_scores", repeats),
                         ("model_summary", summary), ("paired_repeat_differences", details), ("comparison", comparison),
                         ("source_reproduction", reproduction)):
            df.to_csv(dest / (name + ".csv"), index=False)
        write(dest / "decision.json", decision)
        for p, expected in hashes.items():
            if digest(p) != expected:
                raise RuntimeError("Source/input/code changed during run: " + p)
        manifest["status"] = "completed"
        manifest["artifact_sha256"] = {p.name: digest(p) for p in dest.iterdir() if p.is_file() and p.name != "run_manifest.json"}
        return dict(status="completed", output=str(dest), automatic_promotion=False, fits_performed=0)
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(dest / "run_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pca_ensemble_v1.json")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.dry_run), indent=2))


if __name__ == "__main__":
    main()
