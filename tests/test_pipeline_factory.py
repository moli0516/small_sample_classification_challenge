"""Tests for baseline pipeline construction."""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config.experiment_config import ExperimentConfig
from src.modeling.pipeline_factory import PipelineFactory


def test_create_baseline_pipeline() -> None:
    """The factory constructs the expected steps and model settings."""
    config = ExperimentConfig.from_json("configs/baseline.json")

    pipeline = PipelineFactory().create(config)

    assert list(pipeline.named_steps) == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)

    classifier = pipeline.named_steps["classifier"]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.C == config.regularization_c
    assert classifier.max_iter == config.max_iter
    assert classifier.solver == "lbfgs"


def test_pipeline_is_unfitted() -> None:
    """Construction alone must not learn scaling or model parameters."""
    config = ExperimentConfig.from_json("configs/baseline.json")

    pipeline = PipelineFactory().create(config)

    assert not hasattr(pipeline.named_steps["scaler"], "mean_")
    assert not hasattr(pipeline.named_steps["classifier"], "coef_")


def test_create_returns_independent_instances() -> None:
    """Separate experiments must not share estimator instances."""
    config = ExperimentConfig.from_json("configs/baseline.json")
    factory = PipelineFactory()

    first = factory.create(config)
    second = factory.create(config)

    assert first is not second
    assert first.named_steps["scaler"] is not second.named_steps["scaler"]
    assert (
        first.named_steps["classifier"]
        is not second.named_steps["classifier"]
    )