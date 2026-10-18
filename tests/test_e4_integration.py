"""E4 compatibility, diagnostics, and deterministic-run tests."""
from dataclasses import replace
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.experiment_config import ExperimentConfig
from src.data.dataset import Dataset
from src.modeling.pipeline_factory import PipelineFactory
from src.validation.validation_runner import ValidationRunner


@pytest.fixture
def config(tmp_path):
    return ExperimentConfig('test_e4', tmp_path / 'data.json', 'logistic_regression',
                            0.1, 10000, 3, (0, 1), tmp_path / 'results',
                            regularization_type='l1', solver='saga', model_seed=42)


@pytest.fixture
def dataset():
    rng = np.random.default_rng(5)
    y = pd.Series(np.tile([0, 1], 40), index=range(100, 180))
    X = pd.DataFrame(rng.normal(size=(80, 6)), columns=list('abcdef'), index=y.index)
    X['a'] += y.to_numpy() * 1.5
    test = X.iloc[:5].copy()
    return Dataset(X, y, test, pd.DataFrame(index=X.index), pd.DataFrame(index=test.index), 'target')


def test_old_json_defaults_preserved(config, tmp_path):
    payload = config.to_dict()
    for field in ['regularization_type', 'solver', 'model_seed', 'tol', 'feature_selection_k']:
        payload.pop(field)
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(payload))
    old = ExperimentConfig.from_json(path)
    assert old.regularization_type == 'l2'
    assert old.solver == 'lbfgs'
    assert old.model_seed is None
    assert old.tol == 1e-4
    assert old.feature_selection_k is None


def test_json_round_trip(config, tmp_path):
    path = tmp_path / 'roundtrip.json'
    path.write_text(json.dumps(config.to_dict()))
    assert ExperimentConfig.from_json(path) == config


@pytest.mark.parametrize('changes', [
    {'regularization_type': 'elasticnet'}, {'solver': 'invalid'},
    {'solver': 'lbfgs'}, {'model_seed': None}, {'model_seed': True},
    {'model_seed': -1}, {'tol': 0}, {'tol': float('nan')}, {'tol': True},
])
def test_invalid_settings(config, changes):
    with pytest.raises((TypeError, ValueError)):
        replace(config, **changes)


def test_l1_factory(config):
    model = PipelineFactory().create(config)
    assert list(model.named_steps) == ['scaler', 'classifier']
    clf = model.named_steps['classifier']
    assert clf.solver == 'saga'
    assert clf.random_state == 42
    assert clf.max_iter == 10000
    assert clf.tol == 1e-4
    if 'penalty' in clf.get_params() and clf.get_params()['penalty'] != 'deprecated':
        assert clf.penalty == 'l1'
    else:
        assert clf.l1_ratio == 1


def test_legacy_l2_scores_unchanged(config, dataset):
    old = replace(config, regularization_type='l2', solver='lbfgs', model_seed=None, max_iter=2000)
    result = ValidationRunner().evaluate(PipelineFactory().create(old), dataset, old)
    expected = []
    for seed in old.split_seeds:
        for train, valid in StratifiedKFold(3, shuffle=True, random_state=seed).split(dataset.x_train, dataset.y_train):
            model = Pipeline([('scaler', StandardScaler()),
                              ('classifier', LogisticRegression(C=old.regularization_c, solver='lbfgs', max_iter=2000))])
            model.fit(dataset.x_train.iloc[train], dataset.y_train.iloc[train])
            expected.append(roc_auc_score(dataset.y_train.iloc[valid], model.predict_proba(dataset.x_train.iloc[valid])[:, 1]))
    np.testing.assert_allclose(result.fold_scores.valid_auc, expected, rtol=0, atol=1e-12)


def test_coefficients_and_diagnostics(config, dataset):
    template = PipelineFactory().create(config)
    result = ValidationRunner().evaluate(template, dataset, config)
    records = json.loads((config.output_dir / 'coefficient_records.json').read_text())
    assert len(records) == len(result.fold_scores) == 6
    for record in records:
        assert record['feature_names'] == dataset.feature_names
        values = np.array(record['coefficients'])
        assert len(values) == 6
        assert record['exact_nonzero_count'] == int(np.count_nonzero(values))
        assert record['effective_nonzero_count'] == int((abs(values) > 1e-8).sum())
        assert record['effective_nonzero_features'] == np.array(dataset.feature_names)[abs(values) > 1e-8].tolist()
        assert isinstance(record['convergence_warning'], bool)
        assert len(record['n_iter']) == 1
    assert not hasattr(template.named_steps['classifier'], 'coef_')
    assert not (config.output_dir / 'selected_features.json').exists()
    assert (config.output_dir / 'convergence_summary.json').exists()
    environment = json.loads((config.output_dir / 'environment.json').read_text())
    assert environment['estimator_parameters']['random_state'] == 42


def test_reproducible_l1_scores(config, dataset):
    runner = ValidationRunner()
    factory = PipelineFactory()
    a = runner.evaluate(factory.create(config), dataset, config)
    other = replace(config, output_dir=config.output_dir.parent / 'second')
    b = runner.evaluate(factory.create(other), dataset, other)
    pd.testing.assert_frame_equal(a.fold_scores, b.fold_scores)


def test_zero_coefficients_are_retained(config, dataset):
    small = replace(config, regularization_c=1e-10)
    result = ValidationRunner().evaluate(PipelineFactory().create(small), dataset, small)
    records = json.loads((small.output_dir / 'coefficient_records.json').read_text())
    assert all(r['exact_nonzero_count'] == 0 for r in records)
    np.testing.assert_allclose(result.fold_scores.valid_auc, 0.5)


def test_convergence_warning_saved_and_reemitted(config, dataset, monkeypatch):
    original_fit = Pipeline.fit
    def warned_fit(self, X, y=None, **kwargs):
        result = original_fit(self, X, y, **kwargs)
        warnings.warn('synthetic convergence diagnostic', ConvergenceWarning)
        return result
    monkeypatch.setattr(Pipeline, 'fit', warned_fit)
    with pytest.warns(ConvergenceWarning, match='synthetic convergence diagnostic'):
        ValidationRunner().evaluate(PipelineFactory().create(config), dataset, config)
    records = json.loads((config.output_dir / 'coefficient_records.json').read_text())
    assert all(r['convergence_warning'] for r in records)
    assert all('synthetic convergence diagnostic' in r['convergence_messages'] for r in records)
    summary = json.loads((config.output_dir / 'convergence_summary.json').read_text())
    assert summary['fits_with_convergence_warning'] == 6
    assert not summary['all_fits_without_convergence_warning']


def test_selector_records_still_supported(config, dataset):
    selected = replace(config, feature_selection_k=3)
    ValidationRunner().evaluate(PipelineFactory().create(selected), dataset, selected)
    features = json.loads((selected.output_dir / 'selected_features.json').read_text())
    coefficients = json.loads((selected.output_dir / 'coefficient_records.json').read_text())
    assert len(features) == len(coefficients) == 6
    for feature, coef in zip(features, coefficients):
        assert feature['selected_features'] == coef['feature_names']
        assert len(coef['coefficients']) == 3


def test_modern_logistic_api_mapping(config, monkeypatch):
    import src.modeling.pipeline_factory as factory_module
    original = LogisticRegression
    captured = {}
    def modern(*, C=1.0, solver='lbfgs', max_iter=100, random_state=None,
               tol=1e-4, penalty='deprecated', l1_ratio=0.0):
        captured.update(penalty=penalty, l1_ratio=l1_ratio)
        return original(C=C, solver=solver, max_iter=max_iter, random_state=random_state,
                        tol=tol, penalty='l1' if l1_ratio == 1 else 'l2')
    monkeypatch.setattr(factory_module, 'LogisticRegression', modern)
    factory_module.PipelineFactory().create(config)
    assert captured == {'penalty': 'deprecated', 'l1_ratio': 1.0}
    l2 = replace(config, regularization_type='l2')
    factory_module.PipelineFactory().create(l2)
    assert captured['l1_ratio'] == 0.0


def test_config_generator_and_no_overwrite(config, tmp_path, monkeypatch):
    import sys
    from scripts.create_e4_configs import main
    base = replace(config, n_splits=5, split_seeds=tuple(range(10)),
                   regularization_type='l2', solver='lbfgs', model_seed=None,
                   max_iter=2000, regularization_c=0.001)
    source = tmp_path / 'base.json'
    source.write_text(json.dumps(base.to_dict()))
    monkeypatch.setattr(ExperimentConfig, 'project_root', staticmethod(lambda: tmp_path))
    monkeypatch.setattr(sys, 'argv', ['create_e4_configs', '--base-config', str(source)])
    main()
    paths = sorted((tmp_path/'configs').glob('e4_l1_c_*.json'))
    assert len(paths) == 4
    configs = [ExperimentConfig.from_json(path) for path in paths]
    assert sorted(item.regularization_c for item in configs) == [0.01, 0.1, 1.0, 10.0]
    assert all(item.feature_selection_k is None and item.solver == 'saga'
               and item.model_seed == 42 and item.max_iter == 10000 for item in configs)
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(FileExistsError):
        main()
    assert all(path.read_bytes() == before[path] for path in paths)
