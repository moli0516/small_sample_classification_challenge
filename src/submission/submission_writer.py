"""Validate and export predictions using the official submission template."""

from pathlib import Path

import numpy as np
import pandas as pd


class SubmissionWriter:
    """Write class-1 probabilities in the official id/target format."""

    def __init__(self, sample_path: str | Path) -> None:
        self.sample_path = Path(sample_path)

    def write(
        self,
        test_ids: pd.DataFrame,
        probabilities: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        """Align predictions by ID and write a validated submission.

        Parameters
        ----------
        test_ids : pd.DataFrame
            Identifiers in the same positional order as probabilities.
        probabilities : np.ndarray
            One-dimensional class-1 probabilities.
        output_path : str or Path
            Destination CSV. Existing files are not overwritten.

        Returns
        -------
        Path
            Path to the generated submission.
        """
        sample = pd.read_csv(
            self.sample_path,
            dtype={"id": "string"},
        )

        if sample.columns.tolist() != ["id", "target"]:
            raise ValueError(
                "Sample submission must have columns ['id', 'target']."
            )

        if sample.empty:
            raise ValueError("Sample submission must not be empty.")

        if "id" not in test_ids.columns:
            raise ValueError("test_ids must contain an 'id' column.")

        ids = test_ids["id"].astype("string").reset_index(drop=True)
        scores = np.asarray(probabilities, dtype=float)

        if scores.ndim != 1:
            raise ValueError("probabilities must be one-dimensional.")

        if len(ids) != len(scores) or len(ids) != len(sample):
            raise ValueError(
                "Test IDs, probabilities, and sample must have equal lengths."
            )

        if ids.isna().any() or sample["id"].isna().any():
            raise ValueError("Identifiers must not contain missing values.")

        if ids.duplicated().any() or sample["id"].duplicated().any():
            raise ValueError("Identifiers must be unique.")

        if set(ids) != set(sample["id"]):
            raise ValueError("Test IDs do not match sample submission IDs.")

        if not np.isfinite(scores).all():
            raise ValueError("Probabilities must be finite.")

        if ((scores < 0.0) | (scores > 1.0)).any():
            raise ValueError("Probabilities must be between 0 and 1.")

        predictions = pd.Series(
            scores,
            index=pd.Index(ids),
            name="target",
        )

        submission = sample.copy()
        submission["target"] = (
            sample["id"].map(predictions).to_numpy()
        )

        if submission["target"].isna().any():
            raise ValueError("Missing predictions after ID alignment.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Exclusive creation prevents accidental overwriting.
        with output_path.open(
            "x",
            encoding="utf-8",
            newline="",
        ) as stream:
            submission.to_csv(stream, index=False)

        return output_path