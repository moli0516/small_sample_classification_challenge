"""Store, summarise, and export cross-validation results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Hold completed fold-level results for one experiment.

    Parameters
    ----------
    fold_scores : pd.DataFrame
        One row per fold, containing the required columns below.

    Notes
    -----
    Repetition and fold identifiers may be zero-based or one-based,
    but must be nonnegative integers and used consistently.

    This class checks the result table's internal consistency.
    ValidationRunner must additionally verify that all requested
    repetitions and folds were actually executed.
    """

    fold_scores: pd.DataFrame

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "repeat",
        "split_seed",
        "fold",
        "train_rows",
        "valid_rows",
        "valid_auc",
    )

    def __post_init__(self) -> None:
        """Validate and copy the supplied result table."""
        if not isinstance(self.fold_scores, pd.DataFrame):
            raise TypeError("fold_scores must be a pandas DataFrame.")

        if self.fold_scores.columns.duplicated().any():
            raise ValueError("fold_scores contains duplicate column names.")

        missing = set(self.REQUIRED_COLUMNS) - set(self.fold_scores.columns)

        if missing:
            raise ValueError(
                f"Missing fold-score columns: {sorted(missing)}."
            )

        table = self.fold_scores.loc[:, list(self.REQUIRED_COLUMNS)].copy()

        if table.empty:
            raise ValueError("fold_scores must contain at least one row.")

        if table.isna().any().any():
            raise ValueError("fold_scores must not contain missing values.")

        integer_columns = {
            "repeat": 0,
            "split_seed": 0,
            "fold": 0,
            "train_rows": 1,
            "valid_rows": 1,
        }

        for column, minimum in integer_columns.items():
            if (
                not pd.api.types.is_integer_dtype(table[column])
                or pd.api.types.is_bool_dtype(table[column])
            ):
                raise TypeError(f"{column} must contain integers.")

            if table[column].lt(minimum).any():
                raise ValueError(
                    f"{column} values must be at least {minimum}."
                )

        auc = table["valid_auc"]

        if (
            not pd.api.types.is_numeric_dtype(auc)
            or pd.api.types.is_bool_dtype(auc)
        ):
            raise TypeError("valid_auc must contain numeric scores.")

        if (
            not np.isfinite(auc.to_numpy(dtype=float)).all()
            or not auc.between(0.0, 1.0).all()
        ):
            raise ValueError("valid_auc must contain finite values in [0, 1].")

        if table.duplicated(subset=["repeat", "fold"]).any():
            raise ValueError("Each repeat/fold pair must be unique.")

        seeds_per_repeat = table.groupby("repeat")["split_seed"].nunique()

        if not seeds_per_repeat.eq(1).all():
            raise ValueError("Each repetition must use exactly one split seed.")

        repeat_metadata = table[
            ["repeat", "split_seed"]
        ].drop_duplicates()

        if repeat_metadata["split_seed"].duplicated().any():
            raise ValueError(
                "Different repetitions must use different split seeds."
            )

        fold_sets = table.groupby("repeat")["fold"].apply(
            lambda values: tuple(sorted(values.tolist()))
        )

        if fold_sets.nunique() != 1:
            raise ValueError(
                "All repetitions must contain the same fold identifiers."
            )

        if len(fold_sets.iloc[0]) < 2:
            raise ValueError("Each repetition must contain at least two folds.")

        self.fold_scores = table.sort_values(
            ["repeat", "fold"]
        ).reset_index(drop=True)

    def repeat_scores(self) -> pd.DataFrame:
        """Calculate the unweighted mean fold AUC for each repetition."""
        return (
            self.fold_scores.groupby(
                ["repeat", "split_seed"],
                as_index=False,
                sort=True,
            )
            .agg(
                mean_auc=("valid_auc", "mean"),
                n_folds=("fold", "count"),
            )
        )

    def summary(self) -> dict[str, int | float | None | str]:
        """Summarise mean fold AUCs across repetitions.

        Returns
        -------
        dict
            JSON-compatible summary. std_auc is None when only one
            repetition is available.

        Notes
        -----
        std_auc describes variation across split assignments.
        It is not a confidence interval or standard error.
        """
        scores = self.repeat_scores()["mean_auc"]
        n_repeats = len(scores)

        return {
            "metric": "roc_auc",
            "aggregation": "mean_fold_auc_per_repeat",
            "n_repeats": n_repeats,
            "n_splits": int(
                self.fold_scores.groupby("repeat").size().iloc[0]
            ),
            "n_model_fits": len(self.fold_scores),
            "mean_auc": float(scores.mean()),
            "std_auc": (
                float(scores.std(ddof=1))
                if n_repeats > 1
                else None
            ),
            "min_auc": float(scores.min()),
            "max_auc": float(scores.max()),
        }

    def save(self, output_dir: str | Path) -> None:
        """Save score tables and summary, replacing existing result files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.fold_scores.to_csv(
            output_dir / "fold_scores.csv",
            index=False,
        )

        self.repeat_scores().to_csv(
            output_dir / "repeat_scores.csv",
            index=False,
        )

        summary_path = output_dir / "summary.json"

        summary_path.write_text(
            json.dumps(
                self.summary(),
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )