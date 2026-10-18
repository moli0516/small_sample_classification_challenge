"""Run the fixed logistic-regression baseline."""

import json
import logging
import argparse

from src.config.experiment_config import ExperimentConfig
from src.data.data_loader import DataLoader
from src.modeling.pipeline_factory import PipelineFactory
from src.validation.validation_runner import ValidationRunner


def main() -> None:
    """Load data, evaluate the baseline, and save the results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run a configured classification experiment."
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.json",
        help="Experiment configuration path, relative to the project root.",
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

    project_root = config.project_root()

    dataset = loader.load(
        project_root / data_config["train_path"],
        project_root / data_config["test_path"],
    )

    pipeline = PipelineFactory().create(config)

    result = ValidationRunner().evaluate(
        pipeline=pipeline,
        dataset=dataset,
        config=config,
    )

    result.save(config.output_dir)

    (config.output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nValidation summary")
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()