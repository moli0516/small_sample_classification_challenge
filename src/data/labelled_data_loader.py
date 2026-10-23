"""Load labelled predictors only; never open the competition test file."""
import csv
import numpy as np
import pandas as pd


class LabelledDataLoader:
    def load(self, config):
        with config.train_path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if not header or any(not s.strip() for s in header) or len(set(header)) != len(header):
            raise ValueError("Empty, blank or duplicate training CSV headers.")
        frame = pd.read_csv(config.train_path, dtype={config.id_column: "string"})
        if config.id_column not in frame or config.target_column not in frame:
            raise ValueError("Missing explicit ID or target column.")
        if len(frame) != config.expected_rows:
            raise ValueError("Unexpected labelled-row count.")
        ids = frame[config.id_column]
        if ids.isna().any() or not ids.is_unique or ids.str.strip().eq("").any():
            raise ValueError("IDs must be non-empty and unique.")
        features = [c for c in frame if c not in (config.id_column, config.target_column)]
        if len(features) != config.expected_features:
            raise ValueError("Unexpected predictor count.")
        if any(not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]) for c in features):
            raise ValueError("Predictors must be numeric.")
        x = frame[features].copy()
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError("This frozen E5b protocol rejects NaN/Inf predictors.")
        y = frame[config.target_column]
        if pd.api.types.is_bool_dtype(y) or y.isna().any() or set(y.unique()) != {0, 1}:
            raise ValueError("Target must contain numeric labels 0 and 1.")
        return x, y.astype("int64"), ids.copy(), features
