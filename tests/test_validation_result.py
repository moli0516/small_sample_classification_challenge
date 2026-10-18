"""Tests for cross-validation result handling."""

import pandas as pd
import pytest

from src.validation.validation_result import ValidationResult


def make_scores() -> pd.DataFrame:
    """Return synthetic results for two repetitions of two-fold CV."""
    return pd.DataFrame(
        {
            "repeat": [0, 0, 1, 1],
            "split_seed": [0, 0, 1, 1],
            "fold": [0, 1, 0, 1],
            "train_rows": [125, 125, 125, 125],
            "valid_rows": [125, 125, 125, 125],
            "valid_auc": [0.70, 0.80, 0.80, 0.90],
        }
    )


def test_repeat_scores() -> None:
    result = ValidationResult(make_scores())

    scores = result.repeat_scores()

    assert scores["mean_auc"].tolist() == pytest.approx([0.75, 0.85])
    assert scores["n_folds"].tolist() == [2, 2]


def test_summary() -> None:
    summary = ValidationResult(make_scores()).summary()

    assert summary["n_repeats"] == 2
    assert summary["n_splits"] == 2
    assert summary["n_model_fits"] == 4
    assert summary["mean_auc"] == pytest.approx(0.80)
    assert summary["std_auc"] == pytest.approx(0.0707106781)
    assert summary["min_auc"] == pytest.approx(0.75)
    assert summary["max_auc"] == pytest.approx(0.85)


def test_single_repeat_has_no_across_repeat_std() -> None:
    table = make_scores().query("repeat == 0")

    result = ValidationResult(table)

    assert result.summary()["std_auc"] is None


def test_duplicate_fold_rejected() -> None:
    table = make_scores()
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="repeat/fold"):
        ValidationResult(table)


def test_invalid_auc_rejected() -> None:
    table = make_scores()
    table.loc[0, "valid_auc"] = 1.2

    with pytest.raises(ValueError, match="finite values"):
        ValidationResult(table)


def test_input_not_modified() -> None:
    table = make_scores().iloc[::-1].copy()
    original = table.copy(deep=True)

    ValidationResult(table)

    pd.testing.assert_frame_equal(table, original)


def test_save(tmp_path) -> None:
    result = ValidationResult(make_scores())

    result.save(tmp_path)

    assert (tmp_path / "fold_scores.csv").is_file()
    assert (tmp_path / "repeat_scores.csv").is_file()
    assert (tmp_path / "summary.json").is_file()