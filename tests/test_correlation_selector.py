"""Tests for fold-local correlation selection and pipeline compatibility."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features.correlation_selector import CorrelationSelector


@pytest.fixture
def data():
    rng = np.random.default_rng(4)
    y = pd.Series(np.tile([0, 1], 30), index=range(100, 160))
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=['a', 'b', 'c', 'd'], index=y.index)
    X['b'] = y.to_numpy()
    X['d'] = -y.to_numpy()
    return X, y


def test_signed_correlation_matches_pandas(data):
    X, y = data
    selector = CorrelationSelector(2).fit(X, y)
    np.testing.assert_allclose(selector.correlations_, X.corrwith(y).to_numpy(), atol=1e-12)
    assert selector.selected_features_ == ['b', 'd']
    assert selector.fit(X, y) is selector


def test_original_column_and_row_order_preserved(data):
    X, y = data
    result = CorrelationSelector(2).fit_transform(X, y)
    pd.testing.assert_frame_equal(result, X[['b', 'd']])


def test_all_features_identity(data):
    X, y = data
    pd.testing.assert_frame_equal(CorrelationSelector(4).fit_transform(X, y), X)


def test_exact_ties_use_original_position():
    y = pd.Series([0, 1, 0, 1])
    X = pd.DataFrame({'first': y, 'second': y, 'constant': 7})
    selector = CorrelationSelector(1).fit(X, y)
    assert selector.selected_features_ == ['first']
    assert selector.scores_[2] == 0


def test_support_and_feature_names(data):
    X, y = data
    selector = CorrelationSelector(2).fit(X, y)
    assert selector.get_support().tolist() == [False, True, False, True]
    assert selector.get_support(indices=True).tolist() == [1, 3]
    assert selector.get_feature_names_out().tolist() == ['b', 'd']
    mask = selector.get_support()
    mask[:] = False
    assert selector.get_support().sum() == 2
    with pytest.raises(ValueError, match='input_features'):
        selector.get_feature_names_out(['wrong'])

def test_package_exports_correlation_selector() -> None:
    """Package-level import must expose the original class."""
    from src.features import CorrelationSelector as ExportedSelector
    from src.features.correlation_selector import CorrelationSelector

    assert ExportedSelector is CorrelationSelector


def test_transform_uses_fitted_selection_only(data):
    X, y = data
    selector = CorrelationSelector(2).fit(X, y)
    before = selector.scores_.copy()
    changed = X.iloc[::-1].copy()
    changed['a'] = 1000
    pd.testing.assert_frame_equal(selector.transform(changed), changed[['b', 'd']])
    np.testing.assert_array_equal(selector.scores_, before)


def test_no_mutation(data):
    X, y = data
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    output = CorrelationSelector(2).fit_transform(X, y)
    output.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


@pytest.mark.parametrize('k', [0, -1, 5])
def test_invalid_k_range(data, k):
    with pytest.raises(ValueError, match='k must be between'):
        CorrelationSelector(k).fit(*data)


@pytest.mark.parametrize('k', [True, 1.5, '2', None])
def test_invalid_k_type(data, k):
    with pytest.raises(TypeError, match='k must be an integer'):
        CorrelationSelector(k).fit(*data)


def test_unfitted_rejected(data):
    with pytest.raises(NotFittedError):
        CorrelationSelector(2).transform(data[0])


@pytest.mark.parametrize('case', ['missing', 'infinite', 'text', 'duplicate', 'array'])
def test_invalid_predictors(data, case):
    X, y = data
    if case == 'missing': X.iloc[0, 0] = np.nan
    elif case == 'infinite': X.iloc[0, 0] = np.inf
    elif case == 'text': X['a'] = 'bad'
    elif case == 'duplicate': X.columns = ['a', 'a', 'c', 'd']
    elif case == 'array': X = X.to_numpy()
    with pytest.raises((ValueError, TypeError)):
        CorrelationSelector(2).fit(X, y)


@pytest.mark.parametrize('case', ['length', 'index', 'missing', 'one_class', 'invalid', 'two_dimensional'])
def test_invalid_labels(data, case):
    X, y = data
    if case == 'length': y = y.iloc[:-1].to_numpy()
    elif case == 'index': y = y.reset_index(drop=True)
    elif case == 'missing':
        y = y.astype(float)
        y.iloc[0] = np.nan
    elif case == 'one_class': y[:] = 0
    elif case == 'invalid': y.iloc[0] = 2
    elif case == 'two_dimensional': y = y.to_numpy().reshape(-1, 1)
    with pytest.raises(ValueError):
        CorrelationSelector(2).fit(X, y)


def test_transform_schema_mismatch(data):
    X, y = data
    selector = CorrelationSelector(2).fit(X, y)
    with pytest.raises(ValueError, match='columns and order'):
        selector.transform(X[['d', 'c', 'b', 'a']])
    with pytest.raises(ValueError, match='columns and order'):
        selector.transform(X.drop(columns='a'))


def test_clone_and_serialization(data, tmp_path):
    X, y = data
    selector = CorrelationSelector(2).fit(X, y)
    fresh = clone(selector)
    assert fresh.k == 2
    assert not hasattr(fresh, 'scores_')
    path = tmp_path / 'selector.joblib'
    joblib.dump(selector, path)
    restored = joblib.load(path)
    pd.testing.assert_frame_equal(restored.transform(X), selector.transform(X))


def test_pipeline_cross_validation_and_all_features_control(data):
    X, y = data
    baseline = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(C=0.001))])
    selected = Pipeline([('selector', CorrelationSelector(4)), ('scaler', StandardScaler()), ('classifier', LogisticRegression(C=0.001))])
    splits = list(StratifiedKFold(3, shuffle=True, random_state=2).split(X, y))
    a = cross_val_score(baseline, X, y, cv=splits, scoring='roc_auc')
    b = cross_val_score(selected, X, y, cv=splits, scoring='roc_auc')
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
    assert not hasattr(selected.named_steps['selector'], 'scores_')


def test_pipeline_selector_fits_training_rows_only(data, monkeypatch):
    X, y = data
    splits = list(StratifiedKFold(3, shuffle=True, random_state=2).split(X, y))
    calls = []
    original_fit = CorrelationSelector.fit
    def recording_fit(self, X, y):
        calls.append((X.index.tolist(), y.index.tolist()))
        return original_fit(self, X, y)
    monkeypatch.setattr(CorrelationSelector, 'fit', recording_fit)
    pipeline = Pipeline([('selector', CorrelationSelector(2)), ('scaler', StandardScaler()), ('classifier', LogisticRegression())])
    cross_val_score(pipeline, X, y, cv=splits, scoring='roc_auc', error_score='raise')
    assert len(calls) == 3
    for (x_indices, y_indices), (train, valid) in zip(calls, splits):
        assert x_indices == X.iloc[train].index.tolist()
        assert y_indices == y.iloc[train].index.tolist()
        assert set(x_indices).isdisjoint(X.iloc[valid].index)
