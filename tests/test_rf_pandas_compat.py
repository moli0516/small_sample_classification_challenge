"""Regression: no deprecated set_index keyword; duplicate safeguards stay enabled."""
import warnings
import numpy as np
import pandas as pd
import pytest
import src.experiments.rf_stabilizer as m


@pytest.fixture
def compatible_source(monkeypatch):
    original = pd.DataFrame.set_index
    def checked(self, *args, **kwargs):
        if "verify_integrity" in kwargs:
            warnings.warn("deprecated verify_integrity keyword", FutureWarning)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "set_index", checked)
    x = pd.DataFrame({"feature": np.arange(10)})
    y = pd.Series([0, 1] * 5)
    ids = pd.Series([f"id{i}" for i in range(10)])
    splits, rows, scores = [], [], []
    for repeat in range(10):
        for fold in range(5):
            va = [2 * fold, 2 * fold + 1]
            key = dict(repeat=repeat, split_seed=repeat, fold=fold)
            splits.append(dict(key, valid_indices=va))
            scores.append(dict(key, l1=1., xgboost=1., lightgbm=1., l1_xgb_lgbm=1.))
            for i in va:
                p = .2 if y.iloc[i] == 0 else .8
                rows.append(dict(key, row_index=i, id=ids.iloc[i], target=int(y.iloc[i]),
                    l1=p, xgboost=p, lightgbm=p, l1_xgb_lgbm=p))
    frame, score = pd.DataFrame(rows), pd.DataFrame(scores)
    c = dict(x=x, y=y, ids=ids, splits=splits, cache=frame.set_index(m.KEYS),
             expected=score.set_index(["repeat", "fold"]).xgboost)
    return frame, score, c


def test_no_deprecated_keyword_with_warnings_as_errors(compatible_source):
    frame, scores, c = compatible_source
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = m.validate_oof(frame, scores, c)
    assert result.index.is_unique and len(result) == 100


def test_duplicate_oof_still_rejected(compatible_source):
    frame, scores, c = compatible_source
    frame = pd.concat([frame.iloc[[0]], frame.iloc[:-1]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate OOF"):
        m.validate_oof(frame, scores, c)


def test_duplicate_score_keys_still_rejected(compatible_source):
    frame, scores, c = compatible_source
    scores = pd.concat([scores.iloc[[0]], scores.iloc[:-1]], ignore_index=True)
    with pytest.raises(ValueError, match="fold-score keys"):
        m.validate_oof(frame, scores, c)


def test_bad_blend_still_rejected(compatible_source):
    frame, scores, c = compatible_source
    frame.loc[0, "l1_xgb_lgbm"] = .9
    with pytest.raises(ValueError, match="equal-weight"):
        m.validate_oof(frame, scores, c)


def test_source_auc_still_checked(compatible_source):
    frame, scores, c = compatible_source
    scores.loc[0, "lightgbm"] = .5
    with pytest.raises(ValueError, match="AUC mismatch"):
        m.validate_oof(frame, scores, c)
