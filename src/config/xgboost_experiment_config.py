"""Separate, validated settings for the bounded E5b experiment."""
from __future__ import annotations
from dataclasses import dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class XGBoostExperimentConfig:
    experiment_id: str = "e5b_xgboost"
    train_path: Path = Path("data/raw/train.csv")
    split_indices_path: Path = Path("outputs/results/e4_l1_c_1/split_indices.json")
    sparse_split_indices_path: Path = Path("outputs/results/e4_l1_c_0_1/split_indices.json")
    reference_scores_path: Path = Path("outputs/figures/e4_l1_regularization/e4_paired_repeat_scores.csv")
    output_dir: Path = Path("outputs/results/e5b_xgb_run_01")
    max_depth_values: tuple[int, ...] = (1, 2)
    n_estimators_values: tuple[int, ...] = (50, 100, 200)
    model_seed: int = 42
    expected_rows: int = 250
    expected_features: int = 150
    id_column: str = "id"
    target_column: str = "target"

    def __post_init__(self):
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string.")
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.experiment_id):
            raise ValueError("experiment_id must be a safe filename component.")
        for name in ("id_column", "target_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if self.id_column == self.target_column:
            raise ValueError("ID and target columns must differ.")
        for name in ("expected_rows", "expected_features", "model_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "model_seed" else 1):
                raise ValueError(f"Invalid integer setting: {name}.")
        if self.model_seed > 2**32 - 1:
            raise ValueError("model_seed is out of range.")
        for name in ("max_depth_values", "n_estimators_values"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{name} must be a non-empty list or tuple.")
            if any(type(v) is not int or v < 1 for v in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique positive integers.")
            object.__setattr__(self, name, tuple(values))
        for name in self.path_fields():
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"Invalid path: {name}.")
            value = Path(value).expanduser()
            object.__setattr__(self, name, (self.project_root() / value).resolve())

    @staticmethod
    def project_root():
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def path_fields():
        return ("train_path", "split_indices_path", "sparse_split_indices_path",
                "reference_scores_path", "output_dir")

    @classmethod
    def from_json(cls, path, project_root=None):
        root = Path(project_root).resolve() if project_root is not None else cls.project_root()
        path = (root / Path(path).expanduser()).resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a JSON object.")
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown E5b fields: {sorted(unknown)}.")
        defaults = cls()
        for name in cls.path_fields():
            value = raw.get(name, str(getattr(defaults, name).relative_to(cls.project_root()))
                            if getattr(defaults, name).is_relative_to(cls.project_root()) else str(getattr(defaults, name)))
            raw[name] = (root / Path(value).expanduser()).resolve()
        return cls(**raw)

    def candidates(self):
        for depth in self.max_depth_values:
            for trees in self.n_estimators_values:
                yield f"{self.experiment_id}_depth_{depth}_trees_{trees}", depth, trees

    def to_dict(self):
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            result[f.name] = str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        return result
