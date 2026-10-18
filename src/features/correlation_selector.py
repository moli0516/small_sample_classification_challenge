"""Select numeric DataFrame columns by absolute Pearson correlation."""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CorrelationSelector(TransformerMixin, BaseEstimator):
    """Select the k most correlated predictors using training labels only.

    Parameters
    ----------
    k : int, default=10
        Number of predictors to retain, between 1 and the input width.

    Notes
    -----
    Input must be a numeric pandas DataFrame with unique string columns.
    Missing or infinite values are rejected. Labels must contain both 0
    and 1. A Series label vector must share the DataFrame's row index;
    a one-dimensional array is interpreted positionally.

    Constant predictors receive correlation zero. Exact score ties are
    resolved by original column position. Output columns retain original
    input order, not ranking order. transform never recomputes rankings.
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a pandas DataFrame.')
        if X.empty:
            raise ValueError('X must contain rows and predictors.')
        if X.columns.duplicated().any():
            raise ValueError('X must have unique column names.')
        if not all(isinstance(name, str) for name in X.columns):
            raise TypeError('Column names must be strings.')
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise TypeError('All predictors must be numeric.')
        if any(pd.api.types.is_complex_dtype(dtype) for dtype in X.dtypes):
            raise TypeError('Complex predictors are not supported.')
        if X.isna().any().any():
            raise ValueError('Predictors must not contain missing values.')
        values = X.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Predictors must be finite.')
        return values

    def fit(self, X: pd.DataFrame, y) -> 'CorrelationSelector':
        """Learn signed correlations and selected columns from X and y."""
        values = self._validate_frame(X)
        if isinstance(self.k, bool) or not isinstance(self.k, Integral):
            raise TypeError('k must be an integer.')
        if not 1 <= self.k <= X.shape[1]:
            raise ValueError('k must be between 1 and the number of predictors.')
        if isinstance(y, pd.Series) and not X.index.equals(y.index):
            raise ValueError('X and y indices must align.')
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(X):
            raise ValueError('y must be one-dimensional and match X length.')
        if pd.isna(target).any() or not np.isin(target, [0, 1]).all():
            raise ValueError('y must contain only 0 and 1, without missing labels.')
        if len(np.unique(target)) != 2:
            raise ValueError('y must contain both classes 0 and 1.')

        # Rescale first to limit overflow when centering large finite values.
        scales = np.max(np.abs(values), axis=0)
        scaled = np.divide(values, scales, out=np.zeros_like(values), where=scales > 0)
        centered = scaled - scaled.mean(axis=0)
        centered_y = target.astype(float) - target.astype(float).mean()
        denominator = np.linalg.norm(centered, axis=0) * np.linalg.norm(centered_y)
        correlations = np.divide(
            centered.T @ centered_y,
            denominator,
            out=np.zeros(values.shape[1], dtype=float),
            where=denominator > 0,
        )
        correlations = np.clip(correlations, -1.0, 1.0)
        scores = np.abs(correlations)
        ranking = np.argsort(-scores, kind='stable')
        support = np.zeros(values.shape[1], dtype=bool)
        support[ranking[:int(self.k)]] = True

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy(copy=True)
        self.correlations_ = correlations
        self.scores_ = scores
        self.ranking_indices_ = ranking
        self.support_ = support
        self.selected_features_ = self.feature_names_in_[support].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return selected columns without changing row order or fitting."""
        check_is_fitted(self, attributes=['support_', 'feature_names_in_'])
        self._validate_frame(X)
        if X.columns.tolist() != self.feature_names_in_.tolist():
            raise ValueError('Transform columns and order must match fitted input.')
        return X.loc[:, self.selected_features_].copy()

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return the selection mask or selected positional indices."""
        check_is_fitted(self, attributes=['support_'])
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return selected feature names in their original column order."""
        check_is_fitted(self, attributes=['feature_names_in_', 'support_'])
        if input_features is not None:
            if list(input_features) != self.feature_names_in_.tolist():
                raise ValueError('input_features must match fitted feature names.')
        return self.feature_names_in_[self.support_].copy()
