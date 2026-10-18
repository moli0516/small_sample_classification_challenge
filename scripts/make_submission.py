"""Refit a configured pipeline and generate a Kaggle submission."""

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np

from src.config.experiment_config import ExperimentConfig
from src.data.data_loader import DataLoader
from src.modeling.pipeline_factory import PipelineFactory
from src.submission.submission_writer import SubmissionWriter


def main() -> None:
    """Train on all labelled rows and save submission artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/logistic_c_0_001.json",
    )
    parser.add_argument(
        "--sample",
        default="data/raw/sample_submission.csv",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)

    data_config = json.loads(
        config.data_config_path.read_text(encoding="utf-8")
    )

    loader = DataLoader(
        target_column=data_config.get("target_column"),
        id_columns=data_config.get("id_columns", []),
        expected_features=data_config.get("expected_features", 150),
    )

    dataset = loader.load(
        config.resolve_path(data_config["train_path"]),
        config.resolve_path(data_config["test_path"]),
    )

    pipeline = PipelineFactory().create(config)
    pipeline.fit(dataset.x_train, dataset.y_train)

    positive_columns = np.flatnonzero(
        np.asarray(pipeline.classes_) == 1
    )

    if len(positive_columns) != 1:
        raise ValueError("The fitted model must include positive class 1.")

    probabilities = pipeline.predict_proba(dataset.x_test)[
        :, int(positive_columns[0])
    ]

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    artifact_dir = (
        config.project_root()
        / "outputs"
        / "submissions"
        / f"{config.experiment_id}_{timestamp}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)

    writer = SubmissionWriter(
        sample_path=config.resolve_path(args.sample)
    )

    submission_path = writer.write(
        test_ids=dataset.test_ids,
        probabilities=probabilities,
        output_path=artifact_dir / "submission.csv",
    )

    joblib.dump(pipeline, artifact_dir / "model.joblib")

    (artifact_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Training rows: {len(dataset.x_train)}")
    print(f"Prediction rows: {len(probabilities)}")
    print(
        "Probability range: "
        f"{probabilities.min():.6f}–{probabilities.max():.6f}"
    )
    print(f"Submission saved: {submission_path}")


if __name__ == "__main__":
    main()