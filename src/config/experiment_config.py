"""Configuration for reproducible classification experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Store and validate the settings for one experiment.

    Parameters
    ----------
    experiment_id : str
        Nonempty identifier for the experiment.
    data_config_path : Path
        Path to the data-loading configuration.
    model_name : str
        Supported model identifier.
    regularization_c : float
        Positive, finite inverse regularisation strength.
    max_iter : int
        Positive maximum number of solver iterations.
    n_splits : int
        Number of cross-validation folds, at least two.
    split_seeds : tuple of int
        Unique, nonnegative seeds controlling row assignments.
    output_dir : Path
        Directory for experiment results.

    Notes
    -----
    Relative paths are resolved against the project root inferred from
    this module's location:

        <project_root>/src/config/experiment_config.py

    Construction does not create directories or load the dataset.
    """

    experiment_id: str
    data_config_path: Path
    model_name: str
    regularization_c: float
    max_iter: int
    n_splits: int
    split_seeds: tuple[int, ...]
    output_dir: Path

    SUPPORTED_MODELS: ClassVar[frozenset[str]] = frozenset(
        {"logistic_regression"}
    )

    def __post_init__(self) -> None:
        """Validate values and normalise paths and seeds."""
        if not isinstance(self.experiment_id, str):
            raise TypeError("experiment_id must be a string.")

        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty.")

        if not isinstance(self.model_name, str):
            raise TypeError("model_name must be a string.")

        if self.model_name not in self.SUPPORTED_MODELS:
            supported = ", ".join(sorted(self.SUPPORTED_MODELS))
            raise ValueError(
                f"Unsupported model_name: {self.model_name!r}. "
                f"Supported models: {supported}."
            )

        if (
            isinstance(self.regularization_c, bool)
            or not isinstance(self.regularization_c, (int, float))
        ):
            raise TypeError("regularization_c must be a number.")

        if (
            not math.isfinite(self.regularization_c)
            or self.regularization_c <= 0
        ):
            raise ValueError(
                "regularization_c must be finite and greater than zero."
            )

        self._validate_integer("max_iter", self.max_iter, minimum=1)
        self._validate_integer("n_splits", self.n_splits, minimum=2)

        if not isinstance(self.split_seeds, (list, tuple)):
            raise TypeError("split_seeds must be a list or tuple of integers.")

        if not self.split_seeds:
            raise ValueError("split_seeds must not be empty.")

        for index, seed in enumerate(self.split_seeds):
            self._validate_integer(
                f"split_seeds[{index}]",
                seed,
                minimum=0,
            )

            if seed > 2**32 - 1:
                raise ValueError(
                    f"split_seeds[{index}] must not exceed {2**32 - 1}."
                )

        if len(set(self.split_seeds)) != len(self.split_seeds):
            raise ValueError("split_seeds must contain unique values.")

        # A tuple prevents accidental mutation of the seed collection.
        object.__setattr__(self, "split_seeds", tuple(self.split_seeds))

        object.__setattr__(
            self,
            "regularization_c",
            float(self.regularization_c),
        )

        for name in ("data_config_path", "output_dir"):
            value = getattr(self, name)

            if not isinstance(value, (str, Path)):
                raise TypeError(f"{name} must be a string or pathlib.Path.")

            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{name} must not be empty.")

            object.__setattr__(self, name, self.resolve_path(value))

    @staticmethod
    def _validate_integer(name: str, value: int, minimum: int) -> None:
        """Require an integer, explicitly rejecting boolean values."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")

    @staticmethod
    def project_root() -> Path:
        """Return the project root inferred from this module's location."""
        return Path(__file__).resolve().parents[2]

    @classmethod
    def resolve_path(cls, path: str | Path) -> Path:
        """Resolve a path, interpreting relative paths from the project root."""
        resolved = Path(path).expanduser()

        if not resolved.is_absolute():
            resolved = cls.project_root() / resolved

        return resolved.resolve()

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate an experiment configuration.

        Parameters
        ----------
        path : str or Path
            JSON file path. Relative paths start at the project root.

        Returns
        -------
        ExperimentConfig
            Validated configuration with absolute paths.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValueError
            If JSON is malformed, fields are missing or unknown, or
            configuration values are invalid.
        TypeError
            If a field has an incorrect type.
        """
        config_path = cls.resolve_path(path)

        try:
            with config_path.open("r", encoding="utf-8") as stream:
                settings = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {config_path}: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno})."
            ) from exc

        if not isinstance(settings, dict):
            raise ValueError(
                "Experiment configuration must be a JSON object."
            )

        required_fields = {
            "experiment_id",
            "data_config_path",
            "model_name",
            "regularization_c",
            "max_iter",
            "n_splits",
            "split_seeds",
            "output_dir",
        }

        missing_fields = required_fields - settings.keys()
        unknown_fields = settings.keys() - required_fields

        if missing_fields:
            raise ValueError(
                f"Missing configuration fields: {sorted(missing_fields)}."
            )

        if unknown_fields:
            raise ValueError(
                f"Unknown configuration fields: {sorted(unknown_fields)}."
            )

        return cls(**settings)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of the configuration."""
        return {
            "experiment_id": self.experiment_id,
            "data_config_path": str(self.data_config_path),
            "model_name": self.model_name,
            "regularization_c": self.regularization_c,
            "max_iter": self.max_iter,
            "n_splits": self.n_splits,
            "split_seeds": list(self.split_seeds),
            "output_dir": str(self.output_dir),
        }