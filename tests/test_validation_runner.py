"""Tests for repeated stratified cross-validation."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config.experiment_config import ExperimentConfig
from src.data.dataset import Dataset
from src.validation.validation_result import ValidationResult
from src.validation.validation_runner import ValidationRunner


@pytest.fixture
def dataset() -> Dataset:
    """Create balanced synthetic data with non-default row indices."""
    rng = np.random.default_rng(42)

    x_train = pd.DataFrame(
        rng.normal(size=(60, 4)),
        columns=["a", "b", "c", "d"],
        index=np.arange(100, 160),
    )

    y_train = pd.Series(
        np.tile([0, 1], 30),
        index=x_train.index,
        name="target",
    )

    # Add a reproducible signal without making the task trivial.
    x_train["a"] += y_train.to_numpy() * 0.8

    x_test = pd.DataFrame(
        rng.normal(size=(8, 4)),
        columns=x_train.columns,
        index=np.arange(500, 508),
    )

    return Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        train_ids=pd.DataFrame(
            {"id": [str(i) for i in range(60)]},
            index=x_train.index,
        ),
        test_ids=pd.DataFrame(
            {"id": [str(i) for i in range(60, 68)]},
            index=x_test.index,
        ),
        target_column="target",
    )


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    """Use a small validation run independent of project JSON files."""
    return ExperimentConfig(
        experiment_id="test_validation",
        data_config_path=tmp_path / "data.json",
        model_name="logistic_regression",
        regularization_c=0.1,
        max_iter=2000,
        n_splits=3,
        split_seeds=(0, 1),
        output_dir=tmp_path / "results",
    )


@pytest.fixture
def pipeline() -> Pipeline:
    """Construct an unfitted deterministic baseline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    solver="lbfgs",
                    max_iter=2000,
                ),
            ),
        ]
    )


def read_splits(config: ExperimentConfig) -> dict:
    """Read the split assignments saved by the runner."""
    return json.loads(
        (config.output_dir / "split_indices.json").read_text(
            encoding="utf-8"
        )
    )


def test_evaluate_returns_complete_results(
    dataset: Dataset,
    config: ExperimentConfig,
    pipeline: Pipeline,
) -> None:
    result = ValidationRunner().evaluate(pipeline, dataset, config)

    assert isinstance(result, ValidationResult)

    table = result.fold_scores

    assert len(table) == 6
    assert table["repeat"].tolist() == [0, 0, 0, 1, 1, 1]
    assert table["split_seed"].tolist() == [0, 0, 0, 1, 1, 1]
    assert table["fold"].tolist() == [0, 1, 2, 0, 1, 2]
    assert table["train_rows"].eq(40).all()
    assert table["valid_rows"].eq(20).all()
    assert table["valid_auc"].between(0.0, 1.0).all()

    summary = result.summary()

    assert summary["n_model_fits"] == 6
    assert summary["n_repeats"] == 2
    assert summary["n_splits"] == 3


def test_saved_splits_have_correct_coverage(
    dataset: Dataset,
    config: ExperimentConfig,
    pipeline: Pipeline,
) -> None:
    ValidationRunner().evaluate(pipeline, dataset, config)

    payload = read_splits(config)

    assert payload["experiment_id"] == config.experiment_id
    assert payload["index_type"] == "zero_based_row_position"
    assert payload["n_rows"] == 60
    assert payload["n_splits"] == 3
    assert payload["split_seeds"] == [0, 1]
    assert payload["feature_names"] == dataset.feature_names
    assert len(payload["splits"]) == 6

    all_positions = set(range(len(dataset.x_train)))

    for repeat, seed in enumerate(config.split_seeds):
        records = [
            record
            for record in payload["splits"]
            if record["repeat"] == repeat
        ]

        assert len(records) == config.n_splits
        validation_positions = []

        for record in records:
            train_positions = record["train_indices"]
            valid_positions = record["valid_indices"]

            assert record["split_seed"] == seed
            assert len(train_positions) == len(set(train_positions))
            assert len(valid_positions) == len(set(valid_positions))

            assert set(train_positions).isdisjoint(valid_positions)
            assert set(train_positions) | set(valid_positions) == all_positions

            assert dataset.y_train.iloc[train_positions].nunique() == 2
            assert dataset.y_train.iloc[valid_positions].nunique() == 2

            validation_positions.extend(valid_positions)

        assert sorted(validation_positions) == list(range(60))


def test_scores_match_manual_probability_auc(
    dataset: Dataset,
    config: ExperimentConfig,
    pipeline: Pipeline,
) -> None:
    """Independently reproduce scores from the documented protocol."""
    result = ValidationRunner().evaluate(pipeline, dataset, config)
    expected_scores = []

    for seed in config.split_seeds:
        splitter = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=seed,
        )

        for train_indices, valid_indices in splitter.split(
            dataset.x_train,
            dataset.y_train,
        ):
            model = clone(pipeline)
            model.fit(
                dataset.x_train.iloc[train_indices],
                dataset.y_train.iloc[train_indices],
            )

            probabilities = model.predict_proba(
                dataset.x_train.iloc[valid_indices]
            )[:, 1]

            expected_scores.append(
                roc_auc_score(
                    dataset.y_train.iloc[valid_indices],
                    probabilities,
                )
            )

    np.testing.assert_allclose(
        result.fold_scores["valid_auc"].to_numpy(),
        expected_scores,
        rtol=0,
        atol=1e-12,
    )


def test_same_seeds_reproduce_scores_and_splits(
    dataset: Dataset,
    config: ExperimentConfig,
    pipeline: Pipeline,
) -> None:
    runner = ValidationRunner()

    first = runner.evaluate(pipeline, dataset, config)
    first_splits = read_splits(config)

    second_config = replace(
        config,
        output_dir=config.output_dir.parent / "second_run",
    )
    second = runner.evaluate(pipeline, dataset, second_config)
    second_splits = read_splits(second_config)

    pd.testing.assert_frame_equal(
        first.fold_scores,
        second.fold_scores,
    )
    assert first_splits == second_splits


def test_different_seeds_change_assignments(
    dataset: Dataset,
    config: ExperimentConfig,
    pipeline: Pipeline,
) -> None:
    ValidationRunner().evaluate(pipeline, dataset, config)
    records = read_splits(config)["splits"]

    first_repeat = [
        record["valid_indices"]
        for record in records
        if record["repeat"] == 0
    ]
    second_repeat = [
        record["valid_indices"]
        for record in records
        if record["repeat"] == 1
    ]

    assert first_repeat != second_repeat


def test_template_pipeline_remains_unfitted(
    dataset: Dataset,
    config: ExperimentConfig,
    pipeline: Pipeline,
) -> None:
    ValidationRunner().evaluate(pipeline, dataset, config)

    assert not hasattr(pipeline.named_steps["scaler"], "mean_")
    assert not hasattr(pipeline.named_steps["classifier"], "coef_")


def test_fresh_pipeline_and_training_only_scaling(
    dataset: Dataset,
    config: ExperimentConfig,
    pipeline: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe actual fits while retaining the real fitting behaviour."""
    original_fit = Pipeline.fit
    fitted_models = []
    fitted_indices = []

    def recording_fit(self, x, y=None, **kwargs):
        result = original_fit(self, x, y, **kwargs)

        fitted_models.append(self)
        fitted_indices.append(x.index.tolist())

        expected_mean = x.mean().to_numpy(dtype=float)

        np.testing.assert_allclose(
            self.named_steps["scaler"].mean_,
            expected_mean,
        )
        assert int(self.named_steps["scaler"].n_samples_seen_) == len(x)

        return result

    monkeypatch.setattr(Pipeline, "fit", recording_fit)

    ValidationRunner().evaluate(pipeline, dataset, config)

    expected_fits = config.n_splits * len(config.split_seeds)

    assert len(fitted_models) == expected_fits
    assert len({id(model) for model in fitted_models}) == expected_fits
    assert len(
        {id(model.named_steps["scaler"]) for model in fitted_models}
    ) == expected_fits
    assert len(
        {id(model.named_steps["classifier"]) for model in fitted_models}
    ) == expected_fits
    assert all(model is not pipeline for model in fitted_models)

    records = read_splits(config)["splits"]

    for actual_indices, record in zip(fitted_indices, records):
        expected_indices = dataset.x_train.iloc[
            record["train_indices"]
        ].index.tolist()

        assert actual_indices == expected_indices


def test_evaluate_does_not_modify_dataset(
    dataset: Dataset,
    config: ExperimentConfig,
    pipeline: Pipeline,
) -> None:
    x_train_before = dataset.x_train.copy(deep=True)
    y_before = dataset.y_train.copy(deep=True)
    x_test_before = dataset.x_test.copy(deep=True)
    train_ids_before = dataset.train_ids.copy(deep=True)
    test_ids_before = dataset.test_ids.copy(deep=True)

    ValidationRunner().evaluate(pipeline, dataset, config)

    pd.testing.assert_frame_equal(dataset.x_train, x_train_before)
    pd.testing.assert_series_equal(dataset.y_train, y_before)
    pd.testing.assert_frame_equal(dataset.x_test, x_test_before)
    pd.testing.assert_frame_equal(dataset.train_ids, train_ids_before)
    pd.testing.assert_frame_equal(dataset.test_ids, test_ids_before)


def test_competition_test_features_do_not_affect_scores(
    dataset: Dataset,
    config: ExperimentConfig,
    pipeline: Pipeline,
) -> None:
    first = ValidationRunner().evaluate(pipeline, dataset, config)

    changed_dataset = replace(
        dataset,
        x_test=pd.DataFrame(
            {"unusable_test_column": ["not a number"] * 8}
        ),
    )

    second_config = replace(
        config,
        output_dir=config.output_dir.parent / "changed_test",
    )

    second = ValidationRunner().evaluate(
        pipeline,
        changed_dataset,
        second_config,
    )

    pd.testing.assert_frame_equal(
        first.fold_scores,
        second.fold_scores,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("length", "different lengths"),
        ("index", "indices must align"),
        ("missing_label", "Labels must contain both"),
        ("invalid_label", "Labels must contain both"),
        ("single_class", "Labels must contain both"),
        ("empty_features", "must not be empty"),
    ],
)
def test_invalid_training_data_rejected(
    case: str,
    message: str,
    dataset: Dataset,
    config: ExperimentConfig,
    pipeline: Pipeline,
) -> None:
    if case == "length":
        dataset.y_train = dataset.y_train.iloc[:-1]

    elif case == "index":
        dataset.y_train = dataset.y_train.reset_index(drop=True)

    elif case == "missing_label":
        dataset.y_train = dataset.y_train.astype(float)
        dataset.y_train.iloc[0] = np.nan

    elif case == "invalid_label":
        dataset.y_train.iloc[0] = 2

    elif case == "single_class":
        dataset.y_train[:] = 0

    elif case == "empty_features":
        dataset.x_train = dataset.x_train.iloc[:, :0]

    with pytest.raises(ValueError, match=message):
        ValidationRunner().evaluate(pipeline, dataset, config)

    assert not (config.output_dir / "split_indices.json").exists()


def test_too_many_folds_rejected(
    dataset: Dataset,
    config: ExperimentConfig,
    pipeline: Pipeline,
) -> None:
    invalid_config = replace(config, n_splits=31)

    with pytest.raises(ValueError, match="smallest class count"):
        ValidationRunner().evaluate(
            pipeline,
            dataset,
            invalid_config,
        )


def test_non_pipeline_rejected(
    dataset: Dataset,
    config: ExperimentConfig,
) -> None:
    with pytest.raises(TypeError, match="scikit-learn Pipeline"):
        ValidationRunner().evaluate(
            LogisticRegression(),
            dataset,
            config,
        )


def test_pipeline_without_predict_proba_rejected(
    dataset: Dataset,
    config: ExperimentConfig,
) -> None:
    unsupported_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC()),
        ]
    )

    with pytest.raises(TypeError, match="predict_proba"):
        ValidationRunner().evaluate(
            unsupported_pipeline,
            dataset,
            config,
        )