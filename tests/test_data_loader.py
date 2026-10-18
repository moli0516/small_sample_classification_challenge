import pandas as pd
import pytest
from src.data.data_loader import DataLoader


def write_pair(tmp_path, train=None, test=None):
    if train is None:
        train = pd.DataFrame({'id': ['001', '002', '003', '004'],
                              'a': [1., 2., 3., 4.], 'b': [2., 1., 4., 3.],
                              'target': [0, 1, 0, 1]})
    if test is None:
        test = pd.DataFrame({'b': [4., 5.], 'id': ['005', '006'], 'a': [5., 6.]})
    a, b = tmp_path / 'train.csv', tmp_path / 'test.csv'
    train.to_csv(a, index=False)
    test.to_csv(b, index=False)
    return a, b


def test_alignment_and_inference(tmp_path):
    ds = DataLoader(id_columns=['id'], expected_features=2).load(*write_pair(tmp_path))
    assert ds.feature_names == ['a', 'b']
    assert ds.x_test.columns.tolist() == ['a', 'b']
    assert ds.target_column == 'target'
    assert ds.train_ids['id'].tolist() == ['001', '002', '003', '004']
    assert ds.x_test['a'].tolist() == [5., 6.]
    assert ds.y_train.tolist() == [0, 1, 0, 1]


def test_feature_count(tmp_path):
    with pytest.raises(ValueError, match='Expected 150'):
        DataLoader(id_columns=['id']).load(*write_pair(tmp_path))


@pytest.mark.parametrize('change, message', [
    ('missing_feature', 'Feature mismatch'),
    ('invalid_label', 'both 0 and 1'),
    ('missing_label', 'both 0 and 1'),
    ('text_feature', 'nonnumeric'),
    ('infinity', 'infinite'),
    ('target_in_test', 'Target must not appear'),
])
def test_invalid_data(tmp_path, change, message):
    train = pd.DataFrame({'a': [1., 2., 3., 4.], 'target': [0., 1., 0., 1.]})
    test = pd.DataFrame({'a': [5., 6.]})
    if change == 'missing_feature':
        test = test.rename(columns={'a': 'wrong'})
    elif change == 'invalid_label':
        train.loc[0, 'target'] = 2
    elif change == 'missing_label':
        train.loc[0, 'target'] = float('nan')
    elif change == 'text_feature':
        test['a'] = ['bad', 'data']
    elif change == 'infinity':
        test.loc[0, 'a'] = float('inf')
    elif change == 'target_in_test':
        test['target'] = 0
    with pytest.raises(ValueError, match=message):
        DataLoader(target_column='target', expected_features=1).load(
            *write_pair(tmp_path, train, test))


def test_missing_predictors_preserved(tmp_path):
    train = pd.DataFrame({'a': [1., float('nan')], 'target': [0, 1]})
    test = pd.DataFrame({'a': [2.]})
    ds = DataLoader(expected_features=1).load(*write_pair(tmp_path, train, test))
    assert ds.x_train.isna().sum().sum() == 1


def test_duplicate_header_rejected(tmp_path):
    a, b = write_pair(tmp_path)
    a.write_text('a,a,target\n1,2,0\n3,4,1\n')
    with pytest.raises(ValueError, match='duplicate column'):
        DataLoader(expected_features=2).load(a, b)


def test_ambiguous_target(tmp_path):
    train = pd.DataFrame({'a': [1, 2], 'target': [0, 1], 'extra': [3, 4]})
    test = pd.DataFrame({'a': [3]})
    with pytest.raises(ValueError, match='unambiguously'):
        DataLoader(expected_features=1).load(*write_pair(tmp_path, train, test))
