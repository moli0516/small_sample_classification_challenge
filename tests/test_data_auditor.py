import json
import pandas as pd
from src.data.dataset import Dataset
from src.data.data_auditor import DataAuditor


def test_audit_counts_and_no_mutation(tmp_path):
    train = pd.DataFrame({'a': [1., 1., 2., 3.], 'constant': [5., 5., 5., 5.]})
    test = pd.DataFrame({'a': [1., 9.], 'constant': [5., 5.]})
    ds = Dataset(train, pd.Series([0, 1, 0, 1]), test,
                 pd.DataFrame(index=train.index), pd.DataFrame(index=test.index), 'target')
    before = train.copy(deep=True)
    summary = DataAuditor().save(ds, tmp_path)
    assert summary['train_duplicate_predictor_rows'] == 1
    assert summary['conflicting_label_feature_groups'] == 1
    assert summary['test_rows_matching_train_predictors'] == 1
    assert summary['constant_train_features'] == ['constant']
    pd.testing.assert_frame_equal(ds.x_train, before)
    assert json.loads((tmp_path / 'summary.json').read_text()) == summary
    assert len(pd.read_csv(tmp_path / 'feature_summary.csv')) == 2
