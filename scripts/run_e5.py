#!/usr/bin/env python3
"""Run the bounded E5 RBF SVM development experiment.

Pipeline:
    StandardScaler -> SVC(kernel="rbf", probability=False)

Grid:
    C: 0.1, 1.0, 10.0
    gamma: "scale", 0.01

Validation:
    Load saved five-fold assignments for split seeds 0–9.
    Never regenerate splits.
    Compute fold AUC from decision_function scores.
    Average fold AUC within each repetition.

Inputs:
    --train:
        250 labelled rows, id, target, 150 numeric predictors.

    --splits:
        JSON or CSV containing saved validation assignments.

    --references:
        E4 paired repeat-score CSV containing:
        repeat, split_seed, mean_auc, n_folds, C, e3_leader_auc.

No test-data prediction, submission generation, or upload is performed.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TIE_TOL = 1e-12

FIXED = {
    "kernel": "rbf",
    "probability": False,
    "class_weight": None,
    "tol": 1e-3,
    "max_iter": 10000,
    "shrinking": True,
    "cache_size": 200,
    "random_state": 42,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def save_json(path, obj):
    text = json.dumps(
        obj,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    Path(path).write_text(text + "\n", encoding="utf-8")


def integer_values(values, name):
    values = pd.to_numeric(values, errors="raise")
    array = np.asarray(values, dtype=float)

    require(
        np.isfinite(array).all()
        and (array == np.floor(array)).all(),
        f"{name} must contain finite integers.",
    )

    return array.astype(int)


def load_data(path, id_col="id", target_col="target"):
    frame = pd.read_csv(path)

    require(
        id_col != target_col,
        "ID and target column names must differ.",
    )
    require(
        id_col in frame.columns and target_col in frame.columns,
        "Missing ID or target column.",
    )
    require(
        len(frame) == 250,
        f"Expected 250 labelled rows; got {len(frame)}.",
    )
    require(
        frame[id_col].notna().all() and frame[id_col].is_unique,
        "IDs must be non-missing and unique.",
    )

    feature_columns = [
        column
        for column in frame.columns
        if column not in (id_col, target_col)
    ]

    require(
        len(feature_columns) == 150,
        f"Expected 150 predictors; got {len(feature_columns)}.",
    )
    require(
        all(
            pd.api.types.is_numeric_dtype(frame[column])
            for column in feature_columns
        ),
        "All predictors must be numeric. "
        "No automatic encoding or imputation is performed.",
    )

    X = frame[feature_columns].to_numpy(dtype=float)

    require(
        np.isfinite(X).all(),
        "Predictors contain missing or infinite values.",
    )

    y = integer_values(frame[target_col], target_col)

    require(
        set(y) == {0, 1},
        "Target must contain both numeric labels 0 and 1.",
    )

    return frame, X, y, feature_columns


def normalise_json_records(raw, n_rows, feature_columns=None):
    """Support both project metadata wrappers and plain fold-record lists."""
    if isinstance(raw, dict):
        require(
            raw.get("index_type") == "zero_based_row_position",
            "Expected index_type='zero_based_row_position'.",
        )
        require(
            raw.get("n_rows") == n_rows,
            "Saved split n_rows differs from the training data.",
        )
        require(
            raw.get("n_splits") == 5,
            "Saved split n_splits must be 5.",
        )
        require(
            raw.get("split_seeds") == list(range(10)),
            "Saved split_seeds must be 0–9 in order.",
        )
        require(
            "splits" in raw,
            "Split JSON object must contain a 'splits' list.",
        )

        if feature_columns is not None and "feature_names" in raw:
            saved_features = [str(value) for value in raw["feature_names"]]
            current_features = [str(value) for value in feature_columns]

            require(
                saved_features == current_features,
                "Saved split feature_names differ from the current "
                "predictor names or order.",
            )

        raw = raw["splits"]

    require(
        isinstance(raw, list),
        "Split JSON must contain a list of fold records.",
    )
    require(
        len(raw) == 50,
        f"Expected 50 JSON fold records; got {len(raw)}.",
    )

    records = []

    for original in raw:
        require(
            isinstance(original, dict),
            "Each split record must be a JSON object.",
        )

        record = dict(original)

        if "valid_indices" in record:
            if "val_indices" in record:
                require(
                    record["val_indices"] == record["valid_indices"],
                    "Conflicting val_indices and valid_indices.",
                )

            record["val_indices"] = record["valid_indices"]

        required_fields = {
            "repeat",
            "split_seed",
            "fold",
            "train_indices",
            "val_indices",
        }

        require(
            required_fields.issubset(record),
            "Invalid split JSON record. Required fields: "
            "repeat, split_seed, fold, train_indices, "
            "and valid_indices or val_indices.",
        )

        records.append(record)

    return records


def load_splits(path, y, feature_columns=None):
    """Return verified validation assignments using zero-based row positions."""
    path = Path(path)
    n_rows = len(y)
    columns = ["repeat", "split_seed", "fold", "row_index"]

    if path.suffix.lower() == ".csv":
        assignments = pd.read_csv(path)

        require(
            set(columns).issubset(assignments.columns),
            f"Split CSV needs columns: {columns}.",
        )

        assignments = assignments[columns].copy()

    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))

        records = normalise_json_records(
            raw,
            n_rows,
            feature_columns=feature_columns,
        )

        rows = []
        fold_keys = set()

        for record in records:
            repeat = int(
                integer_values([record["repeat"]], "repeat")[0]
            )
            split_seed = int(
                integer_values([record["split_seed"]], "split_seed")[0]
            )
            fold = int(
                integer_values([record["fold"]], "fold")[0]
            )

            key = (repeat, fold)

            require(
                key not in fold_keys,
                f"Duplicate JSON fold record: repeat={repeat}, fold={fold}.",
            )
            fold_keys.add(key)

            train_indices = integer_values(
                record["train_indices"],
                "train_indices",
            )
            valid_indices = integer_values(
                record["val_indices"],
                "validation indices",
            )

            require(
                len(np.unique(train_indices)) == len(train_indices),
                f"Duplicate training indices in repeat={repeat}, fold={fold}.",
            )
            require(
                len(np.unique(valid_indices)) == len(valid_indices),
                f"Duplicate validation indices in repeat={repeat}, fold={fold}.",
            )
            require(
                not set(train_indices) & set(valid_indices),
                f"Train/validation overlap in repeat={repeat}, fold={fold}.",
            )
            require(
                set(train_indices) | set(valid_indices)
                == set(range(n_rows)),
                "Each JSON fold must partition every training row. "
                f"Invalid repeat={repeat}, fold={fold}.",
            )

            rows.extend(
                {
                    "repeat": repeat,
                    "split_seed": split_seed,
                    "fold": fold,
                    "row_index": int(index),
                }
                for index in valid_indices
            )

        assignments = pd.DataFrame(rows, columns=columns)

    else:
        raise ValueError(
            "Saved splits must be a .csv or .json file."
        )

    for column in columns:
        assignments[column] = integer_values(
            assignments[column],
            column,
        )

    require(
        len(assignments) == n_rows * 10,
        f"Expected {n_rows * 10} validation assignments; "
        f"got {len(assignments)}.",
    )
    require(
        set(assignments["repeat"]) == set(range(10)),
        "Expected repeats 0–9.",
    )
    require(
        assignments["row_index"].between(0, n_rows - 1).all(),
        "row_index is out of bounds.",
    )
    require(
        not assignments.duplicated(["repeat", "row_index"]).any(),
        "Each row must appear exactly once as validation "
        "within each repetition.",
    )

    all_indices = np.arange(n_rows)

    for repeat, group in assignments.groupby("repeat"):
        require(
            set(group["split_seed"]) == {repeat},
            "Expected repeat=split_seed with seeds 0–9.",
        )
        require(
            set(group["fold"]) == set(range(5)),
            f"Expected folds 0–4 for repeat={repeat}.",
        )
        require(
            set(group["row_index"]) == set(all_indices),
            f"Incomplete validation coverage for repeat={repeat}.",
        )

        fold_sizes = []
        class_counts = []

        for fold, block in group.groupby("fold"):
            valid_indices = block["row_index"].to_numpy()
            train_indices = np.setdiff1d(
                all_indices,
                valid_indices,
            )

            require(
                set(y[valid_indices]) == {0, 1}
                and set(y[train_indices]) == {0, 1},
                "Both classes must exist in training and validation. "
                f"Invalid repeat={repeat}, fold={fold}.",
            )

            fold_sizes.append(len(valid_indices))
            class_counts.append(
                np.bincount(y[valid_indices], minlength=2)
            )

        require(
            max(fold_sizes) - min(fold_sizes) <= 1,
            f"Unequal five-fold sizes for repeat={repeat}.",
        )
        require(
            (
                np.ptp(np.asarray(class_counts), axis=0) <= 1
            ).all(),
            "Saved assignments do not match stratified "
            f"five-fold class counts for repeat={repeat}.",
        )

    return (
        assignments.sort_values(
            ["repeat", "fold", "row_index"]
        )
        .reset_index(drop=True)
    )


def load_references(path):
    frame = pd.read_csv(path)

    required_columns = [
        "repeat",
        "split_seed",
        "mean_auc",
        "n_folds",
        "C",
        "e3_leader_auc",
    ]

    require(
        set(required_columns).issubset(frame.columns),
        f"E4 reference CSV needs columns: {required_columns}.",
    )

    for column in required_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )
        require(
            np.isfinite(frame[column]).all(),
            f"Non-finite reference values in {column}.",
        )

    for column in ["repeat", "split_seed", "n_folds"]:
        frame[column] = integer_values(frame[column], column)

    require(
        (frame["n_folds"] == 5).all(),
        "Reference scores must use five folds.",
    )
    require(
        (frame["repeat"] == frame["split_seed"]).all(),
        "Reference repeat/split_seed mismatch.",
    )
    require(
        frame["mean_auc"].between(0, 1).all()
        and frame["e3_leader_auc"].between(0, 1).all(),
        "Reference AUC values must be within [0, 1].",
    )

    tables = []

    for c_value, output_name in [
        (1.0, "e4_l1_c_1_auc"),
        (0.1, "e4_l1_c_0_1_auc"),
    ]:
        mask = np.isclose(
            frame["C"],
            c_value,
            atol=1e-12,
            rtol=0,
        )

        selected = frame.loc[
            mask,
            [
                "repeat",
                "split_seed",
                "mean_auc",
                "e3_leader_auc",
            ],
        ].copy()

        require(
            len(selected) == 10
            and set(selected["repeat"]) == set(range(10))
            and not selected["repeat"].duplicated().any(),
            f"Reference C={c_value} needs exactly one score "
            "for each repeat 0–9.",
        )

        selected = (
            selected.sort_values("repeat")
            .reset_index(drop=True)
            .rename(columns={"mean_auc": output_name})
        )
        tables.append(selected)

    require(
        np.allclose(
            tables[0]["e3_leader_auc"],
            tables[1]["e3_leader_auc"],
            atol=1e-12,
            rtol=0,
        ),
        "E3 reference scores disagree across E4 candidates.",
    )

    return tables[0].merge(
        tables[1][
            ["repeat", "split_seed", "e4_l1_c_0_1_auc"]
        ],
        on=["repeat", "split_seed"],
        validate="one_to_one",
    )


def candidates():
    for gamma, gamma_tag in [
        ("scale", "scale"),
        (0.01, "0_01"),
    ]:
        for c_value, c_tag in [
            (0.1, "0_1"),
            (1.0, "1"),
            (10.0, "10"),
        ]:
            experiment_id = (
                f"e5_rbf_svm_c_{c_tag}_gamma_{gamma_tag}"
            )
            yield experiment_id, c_value, gamma


def make_pipeline(c_value, gamma):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    C=c_value,
                    gamma=gamma,
                    **FIXED,
                ),
            ),
        ]
    )


def fit_fold(
    X,
    y,
    ids,
    train_indices,
    valid_indices,
    c_value,
    gamma,
    metadata,
):
    pipeline = make_pipeline(c_value, gamma)

    record = {
        **metadata,
        "C": c_value,
        "gamma": gamma,
        "n_train": len(train_indices),
        "n_validation": len(valid_indices),
        "auc": None,
        "valid": False,
        "error": "",
        "convergence_warning": False,
        "fit_status": None,
        "n_iter": None,
        "hit_iteration_limit": False,
    }

    predictions = []
    captured_warnings = []
    started = time.perf_counter()

    try:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")

            pipeline.fit(
                X[train_indices],
                y[train_indices],
            )

            record["fit_seconds"] = (
                time.perf_counter() - started
            )

            svc = pipeline.named_steps["svc"]

            require(
                np.array_equal(svc.classes_, [0, 1]),
                "Unexpected class order; cannot verify "
                "positive-class score orientation.",
            )

            scores = np.asarray(
                pipeline.decision_function(X[valid_indices])
            )

            require(
                scores.shape == (len(valid_indices),)
                and np.isfinite(scores).all(),
                "Decision scores must be finite, one-dimensional, "
                "and aligned to validation rows.",
            )

            fit_status = int(svc.fit_status_)
            raw_n_iter = getattr(svc, "n_iter_", None)

            n_iter = (
                int(np.max(raw_n_iter))
                if raw_n_iter is not None
                else None
            )

            effective_gamma = getattr(svc, "_gamma", None)

            record.update(
                {
                    "auc": float(
                        roc_auc_score(
                            y[valid_indices],
                            scores,
                        )
                    ),
                    "fit_status": fit_status,
                    "n_iter": n_iter,
                    "hit_iteration_limit": (
                        n_iter is not None
                        and n_iter >= FIXED["max_iter"]
                    ),
                    "n_support_0": int(svc.n_support_[0]),
                    "n_support_1": int(svc.n_support_[1]),
                    "n_support_total": int(
                        svc.n_support_.sum()
                    ),
                    "support_fraction": float(
                        svc.n_support_.sum()
                        / len(train_indices)
                    ),
                    "unique_scores": int(
                        np.unique(scores).size
                    ),
                    "score_min": float(scores.min()),
                    "score_max": float(scores.max()),
                    "score_std": float(
                        scores.std(ddof=0)
                    ),
                    "effective_gamma_private": (
                        float(effective_gamma)
                        if effective_gamma is not None
                        else None
                    ),
                }
            )

            predictions = [
                {
                    **metadata,
                    "row_index": int(index),
                    "id": str(ids[index]),
                    "target": int(y[index]),
                    "decision_score": float(score),
                }
                for index, score in zip(
                    valid_indices,
                    scores,
                )
            ]

    except Exception as exc:
        record["error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        record["traceback"] = traceback.format_exc()

    record.setdefault(
        "fit_seconds",
        time.perf_counter() - started,
    )
    record["total_seconds"] = (
        time.perf_counter() - started
    )

    record["convergence_warning"] = any(
        issubclass(item.category, ConvergenceWarning)
        for item in captured_warnings
    )

    record["warnings_json"] = json.dumps(
        [
            {
                "category": item.category.__name__,
                "message": str(item.message),
            }
            for item in captured_warnings
        ],
        ensure_ascii=False,
    )

    record["valid"] = bool(
        not record["error"]
        and record["fit_status"] == 0
        and not record["convergence_warning"]
        and not record["hit_iteration_limit"]
    )

    for prediction in predictions:
        prediction["fold_valid"] = record["valid"]

    return record, predictions


def summarize(fold_records, references, output):
    summaries = []
    repeat_rows = []

    for experiment_id, c_value, gamma in candidates():
        selected = fold_records.loc[
            fold_records["experiment_id"] == experiment_id
        ]

        if selected.empty:
            continue

        eligible = (
            len(selected) == 50
            and bool(selected["valid"].all())
        )

        summary = {
            "experiment_id": experiment_id,
            "C": c_value,
            "gamma": gamma,
            "eligible": eligible,
            "completed_folds": len(selected),
            "valid_folds": int(
                selected["valid"].sum()
            ),
            "failed_folds": int(
                (
                    selected["error"].fillna("") != ""
                ).sum()
            ),
            "warning_folds": int(
                (
                    selected["warnings_json"] != "[]"
                ).sum()
            ),
            "convergence_warning_folds": int(
                selected["convergence_warning"].sum()
            ),
            "iteration_limit_folds": int(
                selected["hit_iteration_limit"].sum()
            ),
            "max_n_iter": selected["n_iter"].max(),
            "total_fit_seconds": selected[
                "fit_seconds"
            ].sum(),
        }

        if eligible:
            repeats = (
                selected.groupby(
                    ["repeat", "split_seed"],
                    as_index=False,
                )
                .agg(
                    mean_auc=("auc", "mean"),
                    n_folds=("fold", "count"),
                )
            )

            require(
                len(repeats) == 10
                and (repeats["n_folds"] == 5).all(),
                "Incomplete results for an eligible candidate.",
            )

            repeats = repeats.merge(
                references,
                on=["repeat", "split_seed"],
                validate="one_to_one",
            )

            repeats["experiment_id"] = experiment_id
            repeats["C"] = c_value
            repeats["gamma"] = gamma

            summary.update(
                {
                    "mean_auc": repeats["mean_auc"].mean(),
                    "std_auc": repeats["mean_auc"].std(
                        ddof=1
                    ),
                    "min_auc": repeats["mean_auc"].min(),
                    "max_auc": repeats["mean_auc"].max(),
                    "mean_support_fraction": selected[
                        "support_fraction"
                    ].mean(),
                }
            )

            reference_columns = [
                ("e4_leader", "e4_l1_c_1_auc"),
                ("e4_sparse", "e4_l1_c_0_1_auc"),
                ("e3_leader", "e3_leader_auc"),
            ]

            for reference_name, column in reference_columns:
                difference = (
                    repeats["mean_auc"]
                    - repeats[column]
                )

                repeats[
                    f"difference_vs_{reference_name}"
                ] = difference

                summary.update(
                    {
                        f"mean_difference_vs_{reference_name}":
                            difference.mean(),
                        f"sd_difference_vs_{reference_name}":
                            difference.std(ddof=1),
                        f"min_difference_vs_{reference_name}":
                            difference.min(),
                        f"max_difference_vs_{reference_name}":
                            difference.max(),
                        f"wins_vs_{reference_name}": int(
                            (difference > TIE_TOL).sum()
                        ),
                        f"losses_vs_{reference_name}": int(
                            (difference < -TIE_TOL).sum()
                        ),
                        f"ties_vs_{reference_name}": int(
                            (
                                difference.abs()
                                <= TIE_TOL
                            ).sum()
                        ),
                    }
                )

            repeat_rows.extend(
                repeats.to_dict("records")
            )

        summaries.append(summary)

    comparison = pd.DataFrame(summaries)

    comparison.to_csv(
        output / "e5_comparison.csv",
        index=False,
    )

    if repeat_rows:
        repeat_table = pd.DataFrame(repeat_rows)
    else:
        repeat_table = pd.DataFrame(
            columns=[
                "experiment_id",
                "repeat",
                "split_seed",
                "mean_auc",
                "n_folds",
            ]
        )

    repeat_table.to_csv(
        output / "e5_paired_repeat_scores.csv",
        index=False,
    )

    return comparison


def select_leader(comparison, references):
    eligible = comparison.loc[
        comparison["eligible"]
    ].copy()

    if eligible.empty:
        return {
            "status": "no_eligible_candidate",
            "e5_leader": None,
            "promote_over_e4": False,
            "reason": (
                "No candidate has 50 valid converged folds."
            ),
        }

    best_mean = eligible["mean_auc"].max()

    tied = eligible.loc[
        (
            eligible["mean_auc"] - best_mean
        ).abs() <= TIE_TOL
    ].copy()

    tied["gamma_priority"] = (
        tied["gamma"] != "scale"
    ).astype(int)

    winner = tied.sort_values(
        ["C", "gamma_priority"]
    ).iloc[0]

    mean_difference = float(
        winner["mean_auc"]
        - references["e4_l1_c_1_auc"].mean()
    )

    return {
        "status": "completed",
        "e5_leader": str(winner["experiment_id"]),
        "mean_auc": float(winner["mean_auc"]),
        "mean_difference_vs_e4_leader": mean_difference,
        "promote_over_e4": bool(
            mean_difference > TIE_TOL
        ),
        "interpretation": (
            "Development selection only; "
            "no independent final evaluation."
        ),
    }


def git_revision():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        return {
            "revision": (
                revision.stdout.strip()
                if revision.returncode == 0
                else None
            ),
            "dirty": (
                bool(status.stdout.strip())
                if status.returncode == 0
                else None
            ),
        }

    except (OSError, subprocess.SubprocessError):
        return {
            "revision": None,
            "dirty": None,
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--train",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--splits",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--references",
        required=True,
        type=Path,
        help=(
            "E4 paired repeat-score CSV, "
            "not the four-row comparison summary."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New, non-existing output directory.",
    )
    parser.add_argument(
        "--id-column",
        default="id",
    )
    parser.add_argument(
        "--target-column",
        default="target",
    )
    parser.add_argument(
        "--confirm-reference-alignment",
        action="store_true",
        help=(
            "Acknowledge independently verifying E3/E4 "
            "data, row order, split and metric alignment."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate inputs and save provenance "
            "without fitting models."
        ),
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    require(
        args.confirm_reference_alignment,
        "Verify E3/E4 data, row order, splits and metric, "
        "then pass --confirm-reference-alignment. "
        "Score tables alone cannot prove alignment.",
    )
    require(
        not args.output.exists(),
        "Output directory already exists. "
        "Choose a new output directory.",
    )

    frame, X, y, feature_columns = load_data(
        args.train,
        id_col=args.id_column,
        target_col=args.target_column,
    )

    splits = load_splits(
        args.splits,
        y,
        feature_columns=feature_columns,
    )

    references = load_references(
        args.references
    )

    output = args.output
    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    splits.to_csv(
        output / "verified_validation_assignments.csv",
        index=False,
    )
    references.to_csv(
        output / "reference_repeat_scores.csv",
        index=False,
    )

    manifest = {
        "started_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "validated",
        "dry_run": args.dry_run,
        "n_rows": len(frame),
        "n_features": len(feature_columns),
        "feature_order": feature_columns,
        "target_column": args.target_column,
        "id_column": args.id_column,
        "score_method": "decision_function",
        "metric": "mean_fold_auc_per_repeat",
        "split_source": (
            "loaded_saved_indices_not_regenerated"
        ),
        "index_base": 0,
        "reference_alignment": (
            "user_attested_not_proven_by_score_csv"
        ),
        "input_paths": {
            "train": str(args.train.resolve()),
            "splits": str(args.splits.resolve()),
            "references": str(
                args.references.resolve()
            ),
        },
        "input_sha256": {
            "train": sha256(args.train),
            "splits": sha256(args.splits),
            "references": sha256(args.references),
        },
        "canonical_splits_sha256": sha256(
            output
            / "verified_validation_assignments.csv"
        ),
        "script_sha256": sha256(__file__),
        "git": git_revision(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "fixed_svc_settings": FIXED,
        "tie_tolerance": TIE_TOL,
        "effective_gamma_note": (
            "effective_gamma_private reads SVC._gamma "
            "when available. This is a private, "
            "version-sensitive diagnostic only."
        ),
        "planned_candidates": 6,
        "planned_cv_fits": 300,
    }

    save_json(
        output / "run_manifest.json",
        manifest,
    )

    if args.dry_run:
        print(
            "Input validation passed.\n"
            f"Rows: {len(frame)}\n"
            f"Predictors: {len(feature_columns)}\n"
            f"Validation assignments: {len(splits)}\n"
            "Saved folds: 50\n"
            "Reference repeats: 10\n"
            "No model fits performed.\n"
            "Use a NEW output directory for the full run."
        )
        return 0

    all_fold_records = []

    ids = frame[
        args.id_column
    ].astype(str).to_numpy()

    all_indices = np.arange(len(frame))

    manifest["status"] = "running"

    save_json(
        output / "run_manifest.json",
        manifest,
    )

    try:
        for experiment_id, c_value, gamma in candidates():
            candidate_output = output / experiment_id
            candidate_output.mkdir()

            configuration = {
                "experiment_id": experiment_id,
                "model_type": "svc",
                "C": c_value,
                "gamma": gamma,
                "score_method": "decision_function",
                "feature_selection_k": None,
                "all_predictors": True,
                "scaler": (
                    StandardScaler().get_params()
                ),
                "svc": (
                    make_pipeline(c_value, gamma)
                    .named_steps["svc"]
                    .get_params()
                ),
                "train_sha256": (
                    manifest["input_sha256"]["train"]
                ),
                "canonical_splits_sha256": (
                    manifest[
                        "canonical_splits_sha256"
                    ]
                ),
            }

            save_json(
                candidate_output / "config.json",
                configuration,
            )

            candidate_folds = []
            candidate_predictions = []

            grouped_splits = splits.groupby(
                ["repeat", "fold"],
                sort=True,
            )

            for (repeat, fold), block in grouped_splits:
                valid_indices = block[
                    "row_index"
                ].to_numpy()

                train_indices = np.setdiff1d(
                    all_indices,
                    valid_indices,
                )

                metadata = {
                    "experiment_id": experiment_id,
                    "repeat": int(repeat),
                    "split_seed": int(
                        block["split_seed"].iloc[0]
                    ),
                    "fold": int(fold),
                }

                record, predictions = fit_fold(
                    X,
                    y,
                    ids,
                    train_indices,
                    valid_indices,
                    c_value,
                    gamma,
                    metadata,
                )

                all_fold_records.append(record)
                candidate_folds.append(record)
                candidate_predictions.extend(
                    predictions
                )

                pd.DataFrame(
                    candidate_folds
                ).to_csv(
                    candidate_output
                    / "fold_diagnostics.csv",
                    index=False,
                )

                prediction_columns = [
                    "experiment_id",
                    "repeat",
                    "split_seed",
                    "fold",
                    "row_index",
                    "id",
                    "target",
                    "decision_score",
                    "fold_valid",
                ]

                pd.DataFrame(
                    candidate_predictions,
                    columns=prediction_columns,
                ).to_csv(
                    candidate_output
                    / "validation_decision_scores.csv",
                    index=False,
                )

                pd.DataFrame(
                    all_fold_records
                ).to_csv(
                    output
                    / "e5_fold_diagnostics.csv",
                    index=False,
                )

                auc_text = (
                    "NA"
                    if record["auc"] is None
                    else f"{record['auc']:.6f}"
                )

                print(
                    f"[{len(all_fold_records):03d}/300] "
                    f"{experiment_id} "
                    f"repeat={repeat} "
                    f"fold={fold} "
                    f"AUC={auc_text} "
                    f"valid={record['valid']}",
                    flush=True,
                )

                if record["error"]:
                    print(
                        f"  Error: {record['error']}",
                        file=sys.stderr,
                        flush=True,
                    )

            summarize(
                pd.DataFrame(all_fold_records),
                references,
                output,
            )

        comparison = summarize(
            pd.DataFrame(all_fold_records),
            references,
            output,
        )

        decision = select_leader(
            comparison,
            references,
        )

        save_json(
            output / "e5_decision.json",
            decision,
        )

        manifest.update(
            {
                "status": "completed",
                "completed_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "completed_fits": len(
                    all_fold_records
                ),
                "eligible_candidates": int(
                    comparison["eligible"].sum()
                ),
            }
        )

        save_json(
            output / "run_manifest.json",
            manifest,
        )

        print("\nE5 decision:")
        print(
            json.dumps(
                decision,
                indent=2,
                ensure_ascii=False,
            )
        )

        if comparison["eligible"].all():
            return 0

        print(
            "\nThe full comparison finished, but at least "
            "one candidate is ineligible. Review diagnostics.",
            file=sys.stderr,
        )
        return 2

    except BaseException as exc:
        manifest.update(
            {
                "status": (
                    "interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed"
                ),
                "stopped_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "completed_fits": len(
                    all_fold_records
                ),
                "stop_reason": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )

        save_json(
            output / "run_manifest.json",
            manifest,
        )
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print(
            "Interrupted. Preserve partial artifacts; "
            "do not rank incomplete results.",
            file=sys.stderr,
        )
        sys.exit(130)

    except Exception as exc:
        print(
            f"E5 stopped: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)