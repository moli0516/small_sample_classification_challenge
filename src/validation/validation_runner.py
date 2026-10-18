"""Run repeated stratified validation and preserve model diagnostics."""
from __future__ import annotations
import json
import logging
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config.experiment_config import ExperimentConfig
from src.data.dataset import Dataset
from src.validation.validation_result import ValidationResult


class ValidationRunner:
    """Evaluate clones without changing supplied pipelines or data.

    No fit failure is silently skipped. Convergence warnings are recorded
    and re-emitted. A warning-bearing run remains a development diagnostic,
    not automatically an accepted candidate. Metadata files are overwritten
    on an explicit rerun in the same output directory.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def evaluate(self, pipeline: Pipeline, dataset: Dataset,
                 config: ExperimentConfig) -> ValidationResult:
        self._validate_inputs(pipeline, dataset, config)
        x, y = dataset.x_train, dataset.y_train
        score_records, split_records = [], []
        feature_records, coefficient_records = [], []
        for repeat, split_seed in enumerate(config.split_seeds):
            splitter = StratifiedKFold(config.n_splits, shuffle=True, random_state=split_seed)
            coverage = np.zeros(len(x), dtype=int)
            for fold, (train, valid) in enumerate(splitter.split(x, y)):
                if np.intersect1d(train, valid).size:
                    raise RuntimeError('Overlapping train/validation indices.')
                coverage[valid] += 1
                x_train, y_train = x.iloc[train], y.iloc[train]
                x_valid, y_valid = x.iloc[valid], y.iloc[valid]
                if y_train.nunique() != 2 or y_valid.nunique() != 2:
                    raise ValueError('Both classes must be present in training and validation.')
                model = clone(pipeline)
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always', ConvergenceWarning)
                    model.fit(x_train, y_train)
                convergence_messages = [str(w.message) for w in captured
                                        if issubclass(w.category, ConvergenceWarning)]
                for warning in captured:
                    warnings.warn(str(warning.message), warning.category, stacklevel=2)
                classes = np.asarray(model.classes_)
                positive = np.flatnonzero(classes == 1)
                if len(classes) != 2 or len(positive) != 1:
                    raise ValueError('The fitted classifier must have classes 0 and 1.')
                scores = model.predict_proba(x_valid)[:, int(positive[0])]
                auc = float(roc_auc_score(y_valid, scores))
                key = {'repeat': repeat, 'split_seed': split_seed, 'fold': fold}
                score_records.append({**key, 'train_rows': len(train),
                                      'valid_rows': len(valid), 'valid_auc': auc})
                split_records.append({**key, 'train_indices': train.tolist(),
                                      'valid_indices': valid.tolist()})
                selector = model.named_steps.get('selector')
                names = dataset.feature_names
                if selector is not None:
                    indices = selector.get_support(indices=True)
                    names = selector.get_feature_names_out().tolist()
                    feature_records.append({
                        **key, 'k': int(selector.k), 'selected_features': names,
                        'selected_absolute_correlations': {
                            str(selector.feature_names_in_[i]): float(selector.scores_[i]) for i in indices
                        },
                    })
                classifier = model.named_steps.get('classifier')
                if classifier is not None and hasattr(classifier, 'coef_'):
                    coefficients = np.asarray(classifier.coef_, dtype=float)
                    if coefficients.shape != (1, len(names)):
                        raise ValueError('Binary coefficients do not align with feature names.')
                    values = coefficients[0]
                    if not np.isfinite(values).all():
                        raise ValueError('Fitted coefficients must be finite.')
                    effective = np.abs(values) > 1e-8
                    coefficient_records.append({
                        **key,
                        'feature_names': list(names),
                        'coefficients': values.tolist(),
                        'intercept': np.asarray(classifier.intercept_).tolist(),
                        'coefficient_scale': 'standardized_predictors',
                        'exact_nonzero_count': int(np.count_nonzero(values)),
                        'effective_threshold': 1e-8,
                        'effective_nonzero_count': int(effective.sum()),
                        'effective_nonzero_features': np.asarray(names)[effective].tolist(),
                        'n_iter': np.asarray(classifier.n_iter_).tolist(),
                        'convergence_warning': bool(convergence_messages),
                        'convergence_messages': convergence_messages,
                    })
                self._logger.info('Repeat %d/%d | fold %d/%d | AUC %.4f%s',
                                  repeat + 1, len(config.split_seeds), fold + 1,
                                  config.n_splits, auc,
                                  ' | CONVERGENCE WARNING' if convergence_messages else '')
            if not np.all(coverage == 1):
                raise RuntimeError('Every row must appear in validation exactly once per repetition.')
        expected = config.n_splits * len(config.split_seeds)
        if len(score_records) != expected:
            raise RuntimeError(f'Expected {expected} completed fits, found {len(score_records)}.')
        if feature_records and len(feature_records) != expected:
            raise RuntimeError('Feature-selection records are incomplete.')
        if coefficient_records and len(coefficient_records) != expected:
            raise RuntimeError('Coefficient records are incomplete.')
        result = ValidationResult(pd.DataFrame(score_records))
        self._save_splits(split_records, dataset, config)
        output_dir = Path(config.output_dir)
        self._save_json(output_dir / 'environment.json', {
            'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__,
            'estimator_parameters': model.named_steps['classifier'].get_params(deep=False)
            if 'classifier' in model.named_steps else {},
        })
        if feature_records:
            self._save_json(output_dir / 'selected_features.json', feature_records)
        if coefficient_records:
            self._save_json(output_dir / 'coefficient_records.json', coefficient_records)
            self._save_json(output_dir / 'convergence_summary.json', {
                'n_model_fits': expected,
                'fits_with_convergence_warning': sum(r['convergence_warning'] for r in coefficient_records),
                'all_fits_without_convergence_warning': not any(r['convergence_warning'] for r in coefficient_records),
                'max_reported_iterations': max(max(r['n_iter']) for r in coefficient_records),
                'note': 'Absence of warnings is not proof of an exact optimum. Review diagnostics before acceptance.',
            })
        return result

    @staticmethod
    def _validate_inputs(pipeline: Pipeline, dataset: Dataset,
                         config: ExperimentConfig) -> None:
        if not isinstance(pipeline, Pipeline):
            raise TypeError('pipeline must be a scikit-learn Pipeline.')
        if not callable(getattr(pipeline, 'predict_proba', None)):
            raise TypeError('The baseline pipeline must support predict_proba.')
        x, y = dataset.x_train, dataset.y_train
        if x.empty:
            raise ValueError('Training predictors must not be empty.')
        if len(x) != len(y):
            raise ValueError('Training predictors and labels have different lengths.')
        if not x.index.equals(y.index):
            raise ValueError('Training predictor and label indices must align.')
        if y.isna().any() or set(y.unique()) != {0, 1}:
            raise ValueError('Labels must contain both 0 and 1, with no missing values.')
        smallest = int(y.value_counts().min())
        if config.n_splits > smallest:
            raise ValueError(f'n_splits={config.n_splits} exceeds the smallest class count ({smallest}).')

    @staticmethod
    def _save_json(path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    @classmethod
    def _save_splits(cls, split_records: list[dict], dataset: Dataset,
                     config: ExperimentConfig) -> None:
        cls._save_json(Path(config.output_dir) / 'split_indices.json', {
            'experiment_id': config.experiment_id,
            'index_type': 'zero_based_row_position', 'n_rows': len(dataset.x_train),
            'n_splits': config.n_splits, 'split_seeds': list(config.split_seeds),
            'feature_names': dataset.feature_names, 'splits': split_records,
        })
