"""Validate and load native saved split indices; never generate new splits."""
import hashlib
import json
from pathlib import Path
import numpy as np


class SavedSplitLoader:
    @staticmethod
    def integer_array(values, name):
        if not isinstance(values, (list, tuple, np.ndarray)) or any(isinstance(v, (bool, np.bool_)) for v in values):
            raise ValueError(f"{name} must be an integer sequence.")
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1 or not np.isfinite(arr).all() or not (arr == np.floor(arr)).all():
            raise ValueError(f"{name} must contain finite integers.")
        return arr.astype(int)

    def load(self, path, y, feature_names):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("index_type") != "zero_based_row_position":
            raise ValueError("Expected native metadata-wrapped, zero-based split JSON.")
        if raw.get("n_rows") != len(y) or raw.get("n_splits") != 5:
            raise ValueError("Saved split dimensions mismatch.")
        if raw.get("split_seeds") != list(range(10)):
            raise ValueError("Expected split seeds 0–9.")
        if raw.get("feature_names") != list(feature_names):
            raise ValueError("Saved feature names/order differ from training data.")
        source = raw.get("splits")
        if not isinstance(source, list) or len(source) != 50:
            raise ValueError("Expected 50 complete fold records.")
        y = np.asarray(y)
        result, seen = [], set()
        for r in source:
            if not isinstance(r, dict):
                raise ValueError("Each fold record must be an object.")
            repeat, seed, fold = map(int, self.integer_array(
                [r["repeat"], r["split_seed"], r["fold"]], "fold identifiers"))
            if repeat != seed or repeat not in range(10) or fold not in range(5) or (repeat, fold) in seen:
                raise ValueError("Invalid or duplicate repeat/seed/fold.")
            seen.add((repeat, fold))
            tr = self.integer_array(r["train_indices"], "train_indices")
            validation = r.get("valid_indices", r.get("val_indices"))
            if "valid_indices" in r and "val_indices" in r and r["valid_indices"] != r["val_indices"]:
                raise ValueError("Conflicting validation-index fields.")
            va = self.integer_array(validation, "valid_indices")
            if len(set(tr)) != len(tr) or len(set(va)) != len(va):
                raise ValueError("Duplicate row indices.")
            if set(tr) & set(va) or set(tr) | set(va) != set(range(len(y))):
                raise ValueError("Fold must partition all rows without overlap.")
            if set(y[tr]) != {0, 1} or set(y[va]) != {0, 1}:
                raise ValueError("Both classes required in train and validation.")
            result.append(dict(repeat=repeat, split_seed=seed, fold=fold,
                               train_indices=tr.tolist(), valid_indices=va.tolist()))
        for repeat in range(10):
            group = [r for r in result if r["repeat"] == repeat]
            indices = [i for r in group for i in r["valid_indices"]]
            sizes = [len(r["valid_indices"]) for r in group]
            counts = [np.bincount(y[r["valid_indices"]], minlength=2) for r in group]
            if sorted(indices) != list(range(len(y))) or max(sizes)-min(sizes) > 1 or (np.ptp(counts, axis=0) > 1).any():
                raise ValueError("Invalid repeated stratified five-fold coverage.")
        return sorted(result, key=lambda r: (r["repeat"], r["fold"]))

    @staticmethod
    def signature(records):
        normalized = [dict(repeat=r["repeat"], split_seed=r["split_seed"], fold=r["fold"],
                           train_indices=sorted(r["train_indices"]), valid_indices=sorted(r["valid_indices"]))
                      for r in sorted(records, key=lambda r: (r["repeat"], r["fold"]))]
        return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
