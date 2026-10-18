from dataclasses import dataclass
import pandas as pd


@dataclass
class Dataset:
    """Container preserving feature names, original row order, and identifiers."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    train_ids: pd.DataFrame
    test_ids: pd.DataFrame
    target_column: str

    @property
    def feature_names(self) -> list[str]:
        """Return predictors in training-column order."""
        return self.x_train.columns.tolist()
