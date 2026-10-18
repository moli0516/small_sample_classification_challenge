"""Evaluate classification pipelines using repeated stratified folds."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config.experiment_config import ExperimentConfig
from src.data.dataset import Dataset
from src.validation.validation_result import ValidationResult


class ValidationRunner:
    """Evaluate an unfitted pipeline on predefined cross-validation splits.

    Notes
    -----
    Each configured seed defines one repetition of shuffled stratified
    k-fold cross-validation.

    The supplied pipeline is cloned before every fit. The original
    pipeline and the dataset are not modified.

    Split indices are zero-based positional indices into the training
    data, not values from the dataset's identifier column.

    This runner does not change estimator random states. If a future
    model is random, fix its random state in PipelineFactory so that
    changing the split seed changes only row assignment.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def evaluate(
        self,
        pipeline: Pipeline,
        dataset: Dataset,
        config: ExperimentConfig,
    ) -> ValidationResult:
        """Fit and score a fresh pipeline for every validation fold.

        Parameters
        ----------
        pipeline : Pipeline
            Pipeline template supporting predict_proba.
        dataset : Dataset
            Labelled training data and associated metadata. The
            competition test predictors are not used in evaluation.
        config : ExperimentConfig
            Experiment settings, including split count and seeds.

        Returns
        -------
        ValidationResult
            Validated fold-level scores and repeat-level summaries.

        Raises
        ------
        ValueError
            If the training data or fold assignments are invalid.
        TypeError
            If the pipeline does not support predict_proba.
        RuntimeError
            If validation coverage or the completed fit count is wrong.

        Notes
        -----
        After successful evaluation, split assignments are written to
        config.output_dir / "split_indices.json". Existing split output
        at that location is replaced.

        Score tables are saved separately using ValidationResult.save.
        """
        self._validate_inputs(pipeline, dataset, config)

        x = dataset.x_train
        y = dataset.y_train

        score_records: list[dict] = []
        split_records: list[dict] = []

        for repeat, split_seed in enumerate(config.split_seeds):
            splitter = StratifiedKFold(
                n_splits=config.n_splits,
                shuffle=True,
                random_state=split_seed,
            )

            validation_counts = np.zeros(len(x), dtype=np.int64)

            for fold, (train_indices, valid_indices) in enumerate(
                splitter.split(x, y)
            ):
                if np.intersect1d(train_indices, valid_indices).size:
                    raise RuntimeError(
                        f"Overlapping train/validation indices in "
                        f"repeat={repeat}, fold={fold}."
                    )

                validation_counts[valid_indices] += 1

                x_train = x.iloc[train_indices]
                y_train = y.iloc[train_indices]
                x_valid = x.iloc[valid_indices]
                y_valid = y.iloc[valid_indices]

                if y_train.nunique() != 2 or y_valid.nunique() != 2:
                    raise ValueError(
                        f"Both classes must be present in training and "
                        f"validation: repeat={repeat}, fold={fold}."
                    )

                fold_pipeline = clone(pipeline)
                fold_pipeline.fit(x_train, y_train)

                classes = np.asarray(fold_pipeline.classes_)
                positive_columns = np.flatnonzero(classes == 1)

                if len(classes) != 2 or len(positive_columns) != 1:
                    raise ValueError(
                        "The fitted classifier must have classes 0 and 1."
                    )

                probabilities = fold_pipeline.predict_proba(x_valid)
                positive_scores = probabilities[
                    :, int(positive_columns[0])
                ]

                valid_auc = float(
                    roc_auc_score(y_valid, positive_scores)
                )

                score_records.append(
                    {
                        "repeat": repeat,
                        "split_seed": split_seed,
                        "fold": fold,
                        "train_rows": len(train_indices),
                        "valid_rows": len(valid_indices),
                        "valid_auc": valid_auc,
                    }
                )

                split_records.append(
                    {
                        "repeat": repeat,
                        "split_seed": split_seed,
                        "fold": fold,
                        "train_indices": train_indices.tolist(),
                        "valid_indices": valid_indices.tolist(),
                    }
                )

                self._logger.info(
                    "Repeat %d/%d | fold %d/%d | validation AUC %.4f",
                    repeat + 1,
                    len(config.split_seeds),
                    fold + 1,
                    config.n_splits,
                    valid_auc,
                )

            if not np.all(validation_counts == 1):
                raise RuntimeError(
                    f"Every row must appear in validation exactly once "
                    f"in repetition {repeat}."
                )

        expected_fits = config.n_splits * len(config.split_seeds)

        if len(score_records) != expected_fits:
            raise RuntimeError(
                f"Expected {expected_fits} completed fits, "
                f"found {len(score_records)}."
            )

        result = ValidationResult(
            fold_scores=pd.DataFrame(score_records)
        )

        self._save_splits(
            split_records=split_records,
            dataset=dataset,
            config=config,
        )

        return result

    @staticmethod
    def _validate_inputs(
        pipeline: Pipeline,
        dataset: Dataset,
        config: ExperimentConfig,
    ) -> None:
        """Check assumptions required by binary stratified evaluation."""
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be a scikit-learn Pipeline.")

        if not callable(getattr(pipeline, "predict_proba", None)):
            raise TypeError(
                "The baseline pipeline must support predict_proba."
            )

        x = dataset.x_train
        y = dataset.y_train

        if x.empty:
            raise ValueError("Training predictors must not be empty.")

        if len(x) != len(y):
            raise ValueError(
                "Training predictors and labels have different lengths."
            )

        if not x.index.equals(y.index):
            raise ValueError(
                "Training predictor and label indices must align."
            )

        if y.isna().any() or set(y.unique()) != {0, 1}:
            raise ValueError(
                "Labels must contain both 0 and 1, with no missing values."
            )

        smallest_class = int(y.value_counts().min())

        if config.n_splits > smallest_class:
            raise ValueError(
                f"n_splits={config.n_splits} exceeds the smallest "
                f"class count ({smallest_class})."
            )

    @staticmethod
    def _save_splits(
        split_records: list[dict],
        dataset: Dataset,
        config: ExperimentConfig,
    ) -> None:
        """Save split positions and metadata for later comparisons."""
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "experiment_id": config.experiment_id,
            "index_type": "zero_based_row_position",
            "n_rows": len(dataset.x_train),
            "n_splits": config.n_splits,
            "split_seeds": list(config.split_seeds),
            "feature_names": dataset.feature_names,
            "splits": split_records,
        }

        (output_dir / "split_indices.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )