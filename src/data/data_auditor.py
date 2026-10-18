import json
from pathlib import Path

import pandas as pd

from src.data.dataset import Dataset


class DataAuditor:
    """Describe raw data without modifying rows or learning preprocessing."""

    def feature_summary(self, dataset: Dataset) -> pd.DataFrame:
        """Return per-feature types, missingness, uniqueness, and train statistics."""
        train, test = dataset.x_train, dataset.x_test
        table = pd.DataFrame(index=train.columns.copy())
        table.index.name = 'feature'
        table['train_dtype'] = train.dtypes.astype(str)
        table['test_dtype'] = test.dtypes.astype(str)
        table['train_missing'] = train.isna().sum()
        table['test_missing'] = test.isna().sum()
        table['train_missing_fraction'] = train.isna().mean()
        table['test_missing_fraction'] = test.isna().mean()
        table['train_unique_nonmissing'] = train.nunique(dropna=True)
        table['train_mean'] = train.mean()
        table['train_std'] = train.std()
        table['train_min'] = train.min()
        table['train_max'] = train.max()
        return table

    def summarize(self, dataset: Dataset) -> dict:
        """Return JSON-compatible counts and diagnostic warnings.

        Duplicate counts exclude the first occurrence. Train/test overlap counts
        test rows exactly matching any train predictor row, ignoring identifiers.
        """
        train, test, y = dataset.x_train, dataset.x_test, dataset.y_train
        constant = train.columns[train.nunique(dropna=True).eq(1)].tolist()
        all_missing = train.columns[train.isna().all()].tolist()
        label_name = '__audit_target__'
        while label_name in train.columns:
            label_name += '_'
        grouped = train.assign(**{label_name: y}).groupby(
            dataset.feature_names, dropna=False
        )[label_name].nunique()
        overlap = test.merge(train.drop_duplicates(), how='left', indicator=True)
        warnings = []
        if len(train) != 250 or len(test) != 19750:
            warnings.append('Row counts differ from the assignment: 250 train, 19750 test.')
        if constant:
            warnings.append('Constant training predictors detected; not removed.')
        if all_missing:
            warnings.append('All-missing training predictors detected; not removed.')
        if train.isna().any().any() or test.isna().any().any():
            warnings.append('Missing predictors retained; handle within model pipelines.')
        if train.duplicated().any():
            warnings.append('Duplicate predictor rows detected; review before choosing splits.')
        suspicious = [c for c in train if c.lower() in {'id', 'index', 'row_id'}
                      or c.lower().startswith('unnamed:')]
        if suspicious:
            warnings.append(f'Possible identifiers still in predictors: {suspicious}. Review manually.')
        id_summary = {}
        for name in dataset.train_ids.columns:
            a, b = dataset.train_ids[name], dataset.test_ids[name]
            id_summary[name] = {
                'train_missing': int(a.isna().sum()),
                'test_missing': int(b.isna().sum()),
                'train_duplicate_rows': int(a.duplicated().sum()),
                'test_duplicate_rows': int(b.duplicated().sum()),
            }
        return {
            'target_column': dataset.target_column,
            'train_rows': len(train),
            'test_rows': len(test),
            'feature_count': len(train.columns),
            'class_counts': {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            'positive_fraction': float(y.mean()),
            'train_missing_cells': int(train.isna().sum().sum()),
            'test_missing_cells': int(test.isna().sum().sum()),
            'train_duplicate_predictor_rows': int(train.duplicated().sum()),
            'test_duplicate_predictor_rows': int(test.duplicated().sum()),
            'conflicting_label_feature_groups': int(grouped.gt(1).sum()),
            'test_rows_matching_train_predictors': int(overlap['_merge'].eq('both').sum()),
            'constant_train_features': constant,
            'all_missing_train_features': all_missing,
            'identifiers': id_summary,
            'warnings': warnings,
        }

    def save(self, dataset: Dataset, output_dir: str | Path) -> dict:
        """Write summary JSON and feature CSV; return the summary."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summarize(dataset)
        (output_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8'
        )
        self.feature_summary(dataset).to_csv(output_dir / 'feature_summary.csv')
        return summary
