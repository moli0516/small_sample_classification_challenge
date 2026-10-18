"""Integration tests for correlation-based feature-selection experiments."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.experiment_config import ExperimentConfig
from src.data.dataset import Dataset
from src.features import CorrelationSelector
from src.modeling.pipeline_factory import PipelineFactory
from src.validation.validation_runner import ValidationRunner


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    """Create a small experiment without loading project JSON files."""
    return ExperimentConfig(
        experiment_id="test_e2",
        data_config_path=tmp_path / "data.json",
        model_name="logistic_regression",
        regularization_c=0.001,
        max_iter=2000,
        n_splits=3,
        split_seeds=(0, 1),
        output_dir=tmp_path / "results",
    )


@pytest.fixture
def dataset() -> Dataset:
    """Create synthetic data with named columns and aligned indices."""
    rng = np.random.default_rng(42)

    x_train = pd.DataFrame(
        rng.normal(size=(60, 6)),
        columns=[f"feature_{index}" for index in range(6)],
        index=range(100, 160),
    )

    y_train = pd.Series(
        np.tile([0, 1], 30),
        index=x_train.index,
        name="target",
    )

    x_train["feature_0"] += (
        y_train.to_numpy() * 0.8
    )

    x_test = pd.DataFrame(
        rng.normal(size=(8, 6)),
        columns=x_train.columns,
    )

    return Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        train_ids=pd.DataFrame(
            {"id": [str(index) for index in range(60)]},
            index=x_train.index,
        ),
        test_ids=pd.DataFrame(
            {"id": [str(index) for index in range(60, 68)]}
        ),
        target_column="target",
    )


def test_old_json_without_selector_field_still_loads(
    config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    settings = config.to_dict()
    settings.pop("feature_selection_k")

    path = tmp_path / "old_config.json"
    path.write_text(
        json.dumps(settings),
        encoding="utf-8",
    )

    loaded = ExperimentConfig.from_json(path)
    pipeline = PipelineFactory().create(loaded)

    assert loaded.feature_selection_k is None
    assert list(pipeline.named_steps) == [
        "scaler",
        "classifier",
    ]


def test_selector_setting_round_trip(
    config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    selected = replace(
        config,
        feature_selection_k=3,
    )

    path = tmp_path / "selected_config.json"
    path.write_text(
        json.dumps(selected.to_dict()),
        encoding="utf-8",
    )

    restored = ExperimentConfig.from_json(path)

    assert restored == selected
    assert restored.feature_selection_k == 3


@pytest.mark.parametrize(
    "k",
    [True, 1.5, "3"],
)
def test_invalid_selector_setting_type(
    config: ExperimentConfig,
    k,
) -> None:
    with pytest.raises(
        TypeError,
        match="feature_selection_k",
    ):
        replace(config, feature_selection_k=k)


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_selector_setting_range(
    config: ExperimentConfig,
    k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="feature_selection_k",
    ):
        replace(config, feature_selection_k=k)


def test_factory_adds_unfitted_selector(
    config: ExperimentConfig,
) -> None:
    selected = replace(
        config,
        feature_selection_k=3,
    )

    pipeline = PipelineFactory().create(selected)

    assert list(pipeline.named_steps) == [
        "selector",
        "scaler",
        "classifier",
    ]

    selector = pipeline.named_steps["selector"]

    assert isinstance(selector, CorrelationSelector)
    assert selector.k == 3
    assert not hasattr(selector, "scores_")


def test_all_features_reproduce_baseline(
    config: ExperimentConfig,
    dataset: Dataset,
) -> None:
    factory = PipelineFactory()
    runner = ValidationRunner()

    baseline = runner.evaluate(
        factory.create(config),
        dataset,
        config,
    )

    selected_config = replace(
        config,
        experiment_id="test_e2_all_features",
        feature_selection_k=6,
        output_dir=config.output_dir.parent / "all_features",
    )

    selected = runner.evaluate(
        factory.create(selected_config),
        dataset,
        selected_config,
    )

    pd.testing.assert_frame_equal(
        baseline.fold_scores,
        selected.fold_scores,
    )

    baseline_splits = json.loads(
        (config.output_dir / "split_indices.json").read_text(
            encoding="utf-8"
        )
    )
    selected_splits = json.loads(
        (
            selected_config.output_dir
            / "split_indices.json"
        ).read_text(encoding="utf-8")
    )

    assert (
        baseline_splits["splits"]
        == selected_splits["splits"]
    )


def test_saved_features_match_training_fold_correlations(
    config: ExperimentConfig,
    dataset: Dataset,
) -> None:
    selected_config = replace(
        config,
        feature_selection_k=3,
    )

    pipeline = PipelineFactory().create(
        selected_config
    )

    ValidationRunner().evaluate(
        pipeline,
        dataset,
        selected_config,
    )

    feature_records = json.loads(
        (
            selected_config.output_dir
            / "selected_features.json"
        ).read_text(encoding="utf-8")
    )

    split_payload = json.loads(
        (
            selected_config.output_dir
            / "split_indices.json"
        ).read_text(encoding="utf-8")
    )

    split_lookup = {
        (
            record["repeat"],
            record["split_seed"],
            record["fold"],
        ): record
        for record in split_payload["splits"]
    }

    assert len(feature_records) == 6

    for record in feature_records:
        key = (
            record["repeat"],
            record["split_seed"],
            record["fold"],
        )

        train_indices = split_lookup[key][
            "train_indices"
        ]

        x_train = dataset.x_train.iloc[
            train_indices
        ]
        y_train = dataset.y_train.iloc[
            train_indices
        ]

        expected_scores = (
            x_train.corrwith(y_train)
            .abs()
            .fillna(0.0)
        )

        ranking = np.argsort(
            -expected_scores.to_numpy(),
            kind="stable",
        )

        selected_positions = np.sort(
            ranking[:3]
        )

        expected_names = (
            x_train.columns[selected_positions]
            .tolist()
        )

        assert record["k"] == 3
        assert record["selected_features"] == expected_names

        saved_scores = record[
            "selected_absolute_correlations"
        ]

        assert set(saved_scores) == set(expected_names)

        for name in expected_names:
            assert saved_scores[name] == pytest.approx(
                expected_scores[name],
                abs=1e-12,
            )

    # Evaluation fits clones, not the supplied template.
    assert not hasattr(
        pipeline.named_steps["selector"],
        "scores_",
    )


def test_baseline_does_not_write_feature_records(
    config: ExperimentConfig,
    dataset: Dataset,
) -> None:
    pipeline = PipelineFactory().create(config)

    ValidationRunner().evaluate(
        pipeline,
        dataset,
        config,
    )

    assert not (
        config.output_dir / "selected_features.json"
    ).exists()