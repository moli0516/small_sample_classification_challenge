"""Backward-compatible experiment settings for E0–E4."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Validate model settings; interpret relative paths from project root."""
    experiment_id: str
    data_config_path: Path
    model_name: str
    regularization_c: float
    max_iter: int
    n_splits: int
    split_seeds: tuple[int, ...]
    output_dir: Path
    feature_selection_k: int | None = None
    regularization_type: str = 'l2'
    solver: str = 'lbfgs'
    model_seed: int | None = None
    tol: float = 1e-4

    SUPPORTED_MODELS: ClassVar[frozenset[str]] = frozenset({'logistic_regression'})

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str):
            raise TypeError('experiment_id must be a string.')
        if not self.experiment_id.strip():
            raise ValueError('experiment_id must not be empty.')
        if not isinstance(self.model_name, str):
            raise TypeError('model_name must be a string.')
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f'Unsupported model_name: {self.model_name!r}.')
        for name in ('regularization_c', 'tol'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f'{name} must be a number.')
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and greater than zero.')
            object.__setattr__(self, name, float(value))
        self._validate_integer('max_iter', self.max_iter, 1)
        self._validate_integer('n_splits', self.n_splits, 2)
        if not isinstance(self.split_seeds, (list, tuple)):
            raise TypeError('split_seeds must be a list or tuple of integers.')
        if not self.split_seeds:
            raise ValueError('split_seeds must not be empty.')
        for i, seed in enumerate(self.split_seeds):
            self._validate_seed(f'split_seeds[{i}]', seed)
        if len(set(self.split_seeds)) != len(self.split_seeds):
            raise ValueError('split_seeds must contain unique values.')
        object.__setattr__(self, 'split_seeds', tuple(self.split_seeds))
        if self.feature_selection_k is not None:
            self._validate_integer('feature_selection_k', self.feature_selection_k, 1)
        if self.model_seed is not None:
            self._validate_seed('model_seed', self.model_seed)
        if self.regularization_type not in ('l1', 'l2'):
            raise ValueError('regularization_type must be l1 or l2.')
        if self.solver not in ('lbfgs', 'saga'):
            raise ValueError('solver must be lbfgs or saga.')
        if self.regularization_type == 'l1' and self.solver != 'saga':
            raise ValueError('L1 experiments require solver=saga in this project.')
        if self.solver == 'saga' and self.model_seed is None:
            raise ValueError('saga requires a fixed model_seed for reproducibility.')
        for name in ('data_config_path', 'output_dir'):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)):
                raise TypeError(f'{name} must be a string or pathlib.Path.')
            if isinstance(value, str) and not value.strip():
                raise ValueError(f'{name} must not be empty.')
            object.__setattr__(self, name, self.resolve_path(value))

    @staticmethod
    def _validate_integer(name: str, value: int, minimum: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f'{name} must be an integer.')
        if value < minimum:
            raise ValueError(f'{name} must be at least {minimum}.')

    @classmethod
    def _validate_seed(cls, name: str, value: int) -> None:
        cls._validate_integer(name, value, 0)
        if value > 2**32 - 1:
            raise ValueError(f'{name} must not exceed {2**32 - 1}.')

    @staticmethod
    def project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def resolve_path(cls, path: str | Path) -> Path:
        result = Path(path).expanduser()
        return (result if result.is_absolute() else cls.project_root() / result).resolve()

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        path = cls.resolve_path(path)
        try:
            settings = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON in {path}: {exc.msg} '
                             f'(line {exc.lineno}, column {exc.colno}).') from exc
        if not isinstance(settings, dict):
            raise ValueError('Experiment configuration must be a JSON object.')
        required = {'experiment_id', 'data_config_path', 'model_name', 'regularization_c',
                    'max_iter', 'n_splits', 'split_seeds', 'output_dir'}
        optional = {'feature_selection_k', 'regularization_type', 'solver', 'model_seed', 'tol'}
        missing, unknown = required - settings.keys(), settings.keys() - required - optional
        if missing:
            raise ValueError(f'Missing configuration fields: {sorted(missing)}.')
        if unknown:
            raise ValueError(f'Unknown configuration fields: {sorted(unknown)}.')
        return cls(**settings)

    def to_dict(self) -> dict[str, object]:
        return {
            'experiment_id': self.experiment_id,
            'data_config_path': str(self.data_config_path),
            'model_name': self.model_name,
            'regularization_c': self.regularization_c,
            'max_iter': self.max_iter,
            'n_splits': self.n_splits,
            'split_seeds': list(self.split_seeds),
            'output_dir': str(self.output_dir),
            'feature_selection_k': self.feature_selection_k,
            'regularization_type': self.regularization_type,
            'solver': self.solver,
            'model_seed': self.model_seed,
            'tol': self.tol,
        }
