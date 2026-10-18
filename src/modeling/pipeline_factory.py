"""Construct unfitted preprocessing and classification pipelines."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.experiment_config import ExperimentConfig


class PipelineFactory:
    """Construct a fresh model pipeline from an experiment configuration.

    Notes
    -----
    This class does not load data, fit estimators, or evaluate models.
    Each call to create returns independent, unfitted components.
    """

    def create(self, config: ExperimentConfig) -> Pipeline:
        """Build the pipeline specified by the configuration.

        Parameters
        ----------
        config : ExperimentConfig
            Validated experiment settings.

        Returns
        -------
        Pipeline
            Unfitted preprocessing and classification pipeline.

        Raises
        ------
        ValueError
            If the requested model is not supported.
        """
        if config.model_name != "logistic_regression":
            raise ValueError(
                f"Unsupported model_name: {config.model_name!r}."
            )

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=config.regularization_c,
                        solver="lbfgs",
                        max_iter=config.max_iter,
                    ),
                ),
            ]
        )