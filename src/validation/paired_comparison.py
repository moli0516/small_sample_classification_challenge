"""Full-precision paired repeat comparisons against the two retained L1 models."""
from pathlib import Path
import numpy as np
import pandas as pd


class PairedComparison:
    TOLERANCE = 1e-12

    def load_references(self, path):
        frame = pd.read_csv(Path(path))
        required = {"C", "repeat", "split_seed", "n_folds", "mean_auc"}
        if not required.issubset(frame):
            raise ValueError("Expected E4 paired repeat-score table, not its summary.")
        for c in required:
            frame[c] = pd.to_numeric(frame[c], errors="raise")
            if not np.isfinite(frame[c]).all():
                raise ValueError("Reference values must be finite.")
        for c in ("repeat", "split_seed", "n_folds"):
            if not (frame[c] == np.floor(frame[c])).all():
                raise ValueError("Reference identifiers must be integers.")
            frame[c] = frame[c].astype(int)
        if not (frame["n_folds"] == 5).all() or not (frame["repeat"] == frame["split_seed"]).all():
            raise ValueError("Reference validation protocol mismatch.")
        if not frame["mean_auc"].between(0, 1).all():
            raise ValueError("Reference AUC outside [0,1].")
        tables = []
        for c, label in ((1., "e4_leader_auc"), (.1, "e4_sparse_auc")):
            a = frame.loc[np.isclose(frame["C"], c, rtol=0, atol=self.TOLERANCE),
                          ["repeat", "split_seed", "mean_auc"]].copy()
            if len(a) != 10 or set(a["repeat"]) != set(range(10)) or a["repeat"].duplicated().any():
                raise ValueError(f"Reference C={c} requires one score for each repeat 0–9.")
            tables.append(a.rename(columns={"mean_auc": label}))
        return tables[0].merge(tables[1], on=["repeat", "split_seed"], validate="one_to_one").sort_values("repeat")

    def compare(self, repeat_scores, references):
        keys = ["repeat", "split_seed"]
        scores = repeat_scores.copy()
        if len(scores) != 10 or set(scores["repeat"]) != set(range(10)) or scores.duplicated(keys).any():
            raise ValueError("Candidate needs ten complete repetition scores.")
        if not (scores["n_folds"] == 5).all():
            raise ValueError("Candidate repetitions must each have five folds.")
        scores = scores.merge(references, on=keys, how="left", validate="one_to_one")
        if scores[["e4_leader_auc", "e4_sparse_auc"]].isna().any().any():
            raise ValueError("Missing paired reference matches.")
        summary = {}
        for label in ("e4_leader", "e4_sparse"):
            difference = scores["mean_auc"] - scores[f"{label}_auc"]
            scores[f"difference_vs_{label}"] = difference
            summary.update({f"mean_difference_vs_{label}": float(difference.mean()),
                            f"sd_difference_vs_{label}": float(difference.std(ddof=1)),
                            f"min_difference_vs_{label}": float(difference.min()),
                            f"max_difference_vs_{label}": float(difference.max()),
                            f"wins_vs_{label}": int((difference > self.TOLERANCE).sum()),
                            f"losses_vs_{label}": int((difference < -self.TOLERANCE).sum()),
                            f"ties_vs_{label}": int((abs(difference) <= self.TOLERANCE).sum())})
        return scores, summary
