import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.data.dataset import Dataset


class DataLoader:
    """Read binary-classification CSVs without fitting any transformations.

    Parameters
    ----------
    target_column : str or None
        Explicit label name. Infer only when exactly one train-only column exists.
    id_columns : sequence of str
        Explicit identifiers excluded from predictors. Never inferred silently.
    expected_features : int or None
        Required predictor count, or None to disable the check.
    """

    def __init__(
        self,
        target_column: str | None = None,
        id_columns: Sequence[str] = (),
        expected_features: int | None = 150,
    ) -> None:
        if isinstance(id_columns, str):
            raise TypeError('id_columns must be a sequence, not a single string.')
        if len(set(id_columns)) != len(id_columns):
            raise ValueError('id_columns contains duplicates.')
        self.target_column = target_column
        self.id_columns = list(id_columns)
        self.expected_features = expected_features

    def _read_csv(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        with path.open('r', encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream), [])
        if not header or any(not name.strip() for name in header):
            raise ValueError(f'{path.name}: empty file or blank column names.')
        if len(set(header)) != len(header):
            raise ValueError(f'{path.name}: duplicate column names.')
        frame = pd.read_csv(path, dtype={name: 'string' for name in self.id_columns})
        if frame.empty:
            raise ValueError(f'{path.name}: no data rows.')
        return frame

    def load(self, train_path: str | Path, test_path: str | Path) -> Dataset:
        """Validate schema and return aligned, untransformed data.

        Missing predictor values are retained; infinite values are rejected.
        Labels must already be encoded as 0 and 1; no mapping is guessed.
        """
        train = self._read_csv(train_path)
        test = self._read_csv(test_path)
        target = self.target_column
        if target is None:
            candidates = [name for name in train if name not in test]
            if len(candidates) != 1:
                raise ValueError(
                    f'Cannot infer target unambiguously: {candidates}. '
                    'Set target_column explicitly.'
                )
            target = candidates[0]
        if target not in train:
            raise ValueError(f'Target {target!r} is absent from training data.')
        if target in test:
            raise ValueError('Target must not appear in the unlabelled test file.')
        if target in self.id_columns:
            raise ValueError('Target cannot also be an identifier.')
        for name in self.id_columns:
            if name not in train or name not in test:
                raise ValueError(f'Identifier {name!r} must exist in both files.')
        features = [name for name in train if name not in [target, *self.id_columns]]
        test_features = [name for name in test if name not in self.id_columns]
        if set(features) != set(test_features):
            raise ValueError(
                f'Feature mismatch: missing={sorted(set(features) - set(test_features))}, '
                f'extra={sorted(set(test_features) - set(features))}'
            )
        if not features:
            raise ValueError('No predictor columns remain.')
        if self.expected_features is not None and len(features) != self.expected_features:
            raise ValueError(
                f'Expected {self.expected_features} predictors, found {len(features)}. '
                'Check identifier configuration.'
            )
        y = train[target].copy()
        if y.isna().any() or set(y.unique()) != {0, 1}:
            raise ValueError('Target must contain both 0 and 1, with no missing labels.')
        for label, frame in [('train', train), ('test', test)]:
            invalid = [name for name in features if not is_numeric_dtype(frame[name])]
            if invalid:
                raise ValueError(f'{label}: nonnumeric predictors: {invalid}')
            if np.isinf(frame[features].to_numpy(dtype=float)).any():
                raise ValueError(f'{label}: infinite predictor values detected.')
        return Dataset(
            x_train=train[features].copy(),
            y_train=y.astype('int64'),
            x_test=test[features].copy(),
            train_ids=train[self.id_columns].copy(),
            test_ids=test[self.id_columns].copy(),
            target_column=target,
        )
