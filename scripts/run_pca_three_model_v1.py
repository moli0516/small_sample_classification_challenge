"""PCA representations for all three frozen classifiers; no fourth model."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-12
ALIASES = ("l1", "xgboost", "lightgbm")
SOURCE_COLUMNS = {"l1": "l1_c1", "xgboost": "xgboost", "lightgbm": "lightgbm", "ensemble": "l1_xgb_lgbm"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate(c):
    if set(c) != {"validation_config", "components", "output_dir"}:
        raise ValueError("Unexpected config fields")
    if c["components"] != [5, 10, 20, 50] or any(type(k) is not int for k in c["components"]):
        raise ValueError("V1 fixes k=5/10/20/50; a changed grid requires another version")
    if any(not isinstance(c[n], str) or not c[n].strip() for n in ("validation_config", "output_dir")):
        raise ValueError("Nonempty paths required")
    return c


def historical_context(config_path, output):
    from src.extensions.validation_protocol_v1 import ValidationProtocolV1
    return ValidationProtocolV1()._prepare(config_path, str(output))


def check_hashes(hashes):
    for name, expected in hashes.items():
        if digest(name) != expected:
            raise RuntimeError("Source/code/input changed: " + name)


def prepare(path, output=None, root=ROOT):
    path = (root / path).resolve()
    cfg = validate(read(path))
    out = (root / (output or cfg["output_dir"])).resolve()
    if out.exists():
        raise FileExistsError("Fresh output directory required; no overwrite/resume")
    source_cfg = (root / cfg["validation_config"]).resolve()
    source = (root / read(source_cfg)["output_dir"]).resolve()
    if source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError("Output must be outside historical source")
    c = historical_context(str(source_cfg), out)
    ctx = c["context"]
    if len(ctx["x"]) != 250 or ctx["x"].shape[1] != 150:
        raise ValueError("Expected original 250x150 problem")
    from src.extensions.validation_protocol_v1 import verify_artifacts
    _, paths = verify_artifacts(source, c["helper"], {"settings.json", "split_indices.json", "fold_scores.csv", "validation_probabilities.csv"})
    saved = sorted([s for s in read(source / "split_indices.json") if s["scheme"] == "cv5"], key=lambda s: (s["repeat"], s["fold"]))
    expected = sorted([s for s in c["splits"] if s["scheme"] == "cv5"], key=lambda s: (s["repeat"], s["fold"]))
    if saved != expected or len(saved) != 50:
        raise ValueError("Historical ordered cv5 assignments changed")
    raw = pd.read_csv(source / "validation_probabilities.csv", dtype={"id": "string"}, float_precision="round_trip")
    raw = raw[raw.scheme == "cv5"].copy()
    keys = ["repeat", "fold", "row_index"]
    if len(raw) != 2500 or raw.duplicated(keys).any():
        raise ValueError("Incomplete/duplicate historical predictions")
    folds = pd.read_csv(source / "fold_scores.csv", float_precision="round_trip")
    folds = folds[folds.scheme == "cv5"].copy()
    baseline, reproduction = [], []
    for s in saved:
        part = raw[(raw.repeat == s["repeat"]) & (raw.fold == s["fold"])].set_index("row_index")
        va = s["valid_indices"]
        if set(part.index) != set(va) or not (part.split_seed == s["split_seed"]).all():
            raise ValueError("Historical probability split mismatch")
        part = part.loc[va]
        if not np.array_equal(part.id.astype(str).to_numpy(), ctx["ids"].iloc[va].astype(str).to_numpy()) or not np.array_equal(part.target.to_numpy(), ctx["y"].iloc[va].to_numpy()):
            raise ValueError("Historical row identity/labels do not match current training data")
        probabilities = {name: part[col].to_numpy(dtype=float) for name, col in SOURCE_COLUMNS.items()}
        for p in probabilities.values():
            valid_probability(p, len(va))
        if not np.allclose(probabilities["ensemble"], sum(probabilities[n] for n in ALIASES) / 3, rtol=0, atol=TOL):
            raise ValueError("Historical component average changed")
        for alias, p in probabilities.items():
            auc = float(roc_auc_score(ctx["y"].iloc[va], p))
            expected_auc, reference_source = historical_reference_auc(folds, s, alias, c)
            if not np.isclose(auc, expected_auc, rtol=0, atol=TOL):
                raise ValueError(f"Historical fold AUC failed reproduction: model={alias} repeat={s['repeat']} fold={s['fold']} actual={auc:.17g} expected={expected_auc:.17g} source={reference_source}")
            key = dict(repeat=s["repeat"], split_seed=s["split_seed"], fold=s["fold"], model=alias)
            baseline.append(dict(key, candidate="original_" + alias, components=None, auc=auc))
            reproduction.append(dict(key, saved_auc=expected_auc, reproduced_auc=auc, reference_source=reference_source))
    mean = np.mean([r["auc"] for r in baseline if r["model"] == "ensemble"])
    if not np.isclose(mean, .9083364180103309, rtol=0, atol=TOL):
        raise ValueError("Frozen ensemble mean failed reproduction")
    if any(max(cfg["components"]) > min(len(s["train_indices"]) - 1, ctx["x"].shape[1]) for s in saved):
        raise ValueError("Invalid component count")
    hashes = dict(c["hashes"])
    for p in paths + [path, Path(__file__).resolve()]:
        hashes[str(p.resolve())] = digest(p)
    c.update(config=cfg, output=out, pca_splits=saved, baseline=baseline, reproduction=reproduction, hashes=hashes)
    return c


def historical_reference_auc(folds, split, alias, context):
    repeat, fold = split["repeat"], split["fold"]
    rows = folds[(folds.repeat == repeat) & (folds.fold == fold)
                 & (folds.candidate == SOURCE_COLUMNS[alias])]
    if len(rows) > 1:
        raise ValueError("Duplicate historical score")
    if len(rows) == 1:
        value, source = float(rows.auc.iloc[0]), "validation_protocol_cv5"
    else:
        if alias not in ("xgboost", "lightgbm"):
            raise ValueError("Required validation score missing: " + alias)
        if context["reference_matches"].get((repeat, fold)) is not True:
            raise ValueError("Trial14 ordered training/validation indices differ")
        ref = context["reference"]
        if not ref.index.is_unique or (repeat, fold) not in ref.index or alias not in ref.columns:
            raise ValueError("Missing/duplicate verified Trial14 score")
        value, source = float(ref.loc[(repeat, fold), alias]), "verified_trial14_matching_ordered_split"
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid historical reference AUC")
    return value, source


def safe_metadata(value):
    if isinstance(value, dict):
        return {str(k): safe_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_metadata(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def build_models(c):
    from src.modeling.pipeline_factory import PipelineFactory
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    ctx = c["context"]
    # Preserve the original L1 classifier parameters; do NOT scale PCA scores again.
    l1 = clone(PipelineFactory().create(ctx["linear"]).named_steps["classifier"])
    return {"l1": l1,
            "xgboost": XGBClassifier(**ctx["candidate"]["estimator_parameters"]),
            "lightgbm": LGBMClassifier(**c["lightgbm_parameters"])}


def representation(x_train, x_valid, k):
    scaler = StandardScaler()
    train = scaler.fit_transform(x_train)
    valid = scaler.transform(x_valid)
    pca = PCA(n_components=k, svd_solver="full", whiten=False)
    return pca.fit_transform(train), pca.transform(valid), scaler, pca


def valid_probability(p, n):
    a = np.asarray(p, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError("Invalid probability shape/range")


def paired(repeats, b, a, kind):
    def get(name):
        return repeats[repeats.candidate == name].set_index("repeat").mean_fold_auc.sort_index()
    first, second = get(a), get(b)
    if list(first.index) != list(range(10)) or not first.index.equals(second.index):
        raise ValueError("Incomplete/unpaired repeat scores")
    d = second - first
    return (pd.DataFrame(dict(repeat=d.index, reference=a, candidate=b, comparison=kind, difference=d.to_numpy())),
            dict(reference=a, candidate=b, comparison=kind, mean_difference=float(d.mean()), difference_sd=float(d.std(ddof=1)),
                 wins=int((d > TOL).sum()), losses=int((d < -TOL).sum()), ties=int((abs(d) <= TOL).sum())))


def summarize(folds, ks):
    names = ["original_" + a for a in (*ALIASES, "ensemble")] + [f"pca{k}_{a}" for k in ks for a in (*ALIASES, "ensemble")]
    expected = {(n, r, f) for n in names for r in range(10) for f in range(5)}
    keys = list(folds[["candidate", "repeat", "fold"]].itertuples(index=False, name=None))
    if len(keys) != len(expected) or set(keys) != expected or not np.isfinite(folds.auc).all():
        raise ValueError("Incomplete/duplicate fold results; no selection on partial output")
    repeats = folds.groupby(["candidate", "repeat"], as_index=False).auc.mean().rename(columns={"auc": "mean_fold_auc"})
    summary = repeats.groupby("candidate").mean_fold_auc.agg(mean_auc="mean", repeat_sd="std").reset_index()
    details, comparisons = [], []
    for k in ks:
        for alias in (*ALIASES, "ensemble"):
            d, s = paired(repeats, f"pca{k}_{alias}", "original_" + alias, "ensemble_vs_original" if alias == "ensemble" else "component_vs_original")
            details.append(d); comparisons.append(dict(s, components=k))
    indexed = summary.set_index("candidate")
    selected_k = max(ks, key=lambda k: indexed.loc[f"pca{k}_ensemble", "mean_auc"])
    selected = next(s for s in comparisons if s["candidate"] == f"pca{selected_k}_ensemble")
    decision = dict(selected_components=selected_k, selected_candidate=f"pca{selected_k}_ensemble",
                    selected_mean_auc=float(indexed.loc[f"pca{selected_k}_ensemble", "mean_auc"]),
                    selection_rule="Highest ensemble mean across common k; exact ties use smaller k; no fallback.",
                    screening_passed=bool(selected["mean_difference"] > TOL and selected["wins"] >= 6),
                    automatic_promotion=False, **selected,
                    interpretation="Fixed-parameter representation comparison on reused development data. Not nested CV or independent evaluation. XGBoost PCA branch replaces original top-20 procedure; not an isolated PCA-only ablation. No retuning/weight search/submission.")
    return repeats, summary, pd.concat(details, ignore_index=True), pd.DataFrame(comparisons), decision


def execute(c, model_factory=build_models):
    out, ctx = c["output"], c["context"]
    ks = c["config"]["components"]
    splits = c["pca_splits"]
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", planned_classifier_fits=len(splits) * len(ks) * 3,
                    planned_pca_fits=len(splits) * len(ks), completed_classifier_fits=0, completed_pca_fits=0,
                    test_opened=False, kaggle_upload_performed=False, started_utc=datetime.now(timezone.utc).isoformat())
    rows, predictions, diagnostics, warnrows = list(c["baseline"]), [], [], []
    write(out / "run_manifest.json", manifest)
    try:
        model_parameters = {name: model.get_params(deep=False) for name, model in model_factory(c).items()}
        write(out / "settings.json", dict(config=c["config"], input_and_code_sha256=c["hashes"],
              versions=c["versions"], classifier_parameters=safe_metadata(model_parameters),
              procedure="Training-only StandardScaler -> full-SVD PCA (no whitening) -> fixed classifier. One shared deterministic representation per split/k. No top-20 in PCA XGBoost, no additional scaling of PCA scores; equal 1/3 weights."))
        write(out / "split_indices.json", splits)
        pd.DataFrame(c["reproduction"]).to_csv(out / "source_reproduction.csv", index=False)
        with threadpool_limits(limits=1):
            for s in splits:
                tr, va = s["train_indices"], s["valid_indices"]
                key = {n: s[n] for n in ("repeat", "split_seed", "fold")}
                for k in ks:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        ztr, zva, scaler, pca = representation(ctx["x"].iloc[tr], ctx["x"].iloc[va], k)
                    warnrows.extend(dict(key, components=k, model="representation", category=w.category.__name__, message=str(w.message)) for w in caught)
                    manifest["completed_pca_fits"] += 1
                    probs = {}
                    for name, model in model_factory(c).items():
                        caught = []
                        try:
                            with warnings.catch_warnings(record=True) as caught, (out / "fit.log").open("a", encoding="utf-8") as log:
                                warnings.simplefilter("always")
                                with redirect_stdout(log), redirect_stderr(log):
                                    model.fit(ztr, ctx["y"].iloc[tr])
                                    classes = np.asarray(model.classes_)
                                    if set(classes) != {0, 1}:
                                        raise ValueError("Unexpected classifier classes")
                                    p = model.predict_proba(zva)[:, int(np.flatnonzero(classes == 1)[0])]
                        finally:
                            warnrows.extend(dict(key, components=k, model=name, category=w.category.__name__, message=str(w.message)) for w in caught)
                            write(out / "warnings.json", warnrows)
                        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                            raise RuntimeError("ConvergenceWarning: failed run; do not select partial output")
                        valid_probability(p, len(va))
                        probs[name] = np.asarray(p, dtype=float)
                        manifest["completed_classifier_fits"] += 1
                    if set(probs) != set(ALIASES):
                        raise ValueError("Exactly three fixed classifiers required")
                    probs["ensemble"] = sum(probs[n] for n in ALIASES) / 3
                    for alias, p in probs.items():
                        rows.append(dict(key, model=alias, components=k, candidate=f"pca{k}_{alias}", auc=float(roc_auc_score(ctx["y"].iloc[va], p))))
                    for j, i in enumerate(va):
                        predictions.append(dict(key, components=k, row_index=int(i), id=str(ctx["ids"].iloc[i]), target=int(ctx["y"].iloc[i]), **{n: float(p[j]) for n, p in probs.items()}))
                    diagnostics.append(dict(key, components=k, scaler_mean=scaler.mean_.tolist(),
                         explained_variance_ratio=pca.explained_variance_ratio_.tolist(), explained_variance_sum=float(pca.explained_variance_ratio_.sum())))
                    write(out / "run_manifest.json", manifest)
                pd.DataFrame(rows).to_csv(out / "fold_scores.csv", index=False)
                print(f"[{manifest['completed_classifier_fits']}/{manifest['planned_classifier_fits']}] repeat={s['repeat']} fold={s['fold']}", flush=True)
        check_hashes(c["hashes"])
        repeats, summary, paired_rows, comparisons, decision = summarize(pd.DataFrame(rows), ks)
        for name, df in (("repeat_scores", repeats), ("model_summary", summary), ("paired_repeat_differences", paired_rows), ("comparison", comparisons)):
            df.to_csv(out / (name + ".csv"), index=False)
        pd.DataFrame(predictions).to_csv(out / "validation_probabilities.csv", index=False)
        write(out / "pca_diagnostics.json", diagnostics)
        write(out / "warnings.json", warnrows)
        write(out / "decision.json", decision)
        check_hashes(c["hashes"])
        manifest["status"] = "completed"
        manifest["artifact_sha256"] = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "run_manifest.json"}
        return dict(status="completed", output=str(out), automatic_promotion=False, classifier_fits=manifest["completed_classifier_fits"], pca_fits=manifest["completed_pca_fits"])
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        pd.DataFrame(rows).to_csv(out / "partial_fold_scores.csv", index=False)
        write(out / "warnings.json", warnrows)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(out / "run_manifest.json", manifest)


def run(path, output=None, dry_run=False):
    c = prepare(path, output)
    check_hashes(c["hashes"])
    if dry_run:
        models = build_models(c)
        if set(models) != set(ALIASES):
            raise ValueError("Exactly three models required")
        return dict(status="preflight_passed", classifiers=list(models), common_components=c["config"]["components"],
                    planned_classifier_fits=600, planned_pca_fits=200, historical_baseline_reproduced=True,
                    fits_performed=0, test_opened=False, output=str(c["output"]))
    return execute(c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pca_three_model_v1.json")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.dry_run), indent=2))


if __name__ == "__main__":
    main()
