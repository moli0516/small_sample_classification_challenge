"""Generate the five fixed-C configurations for E2."""

import argparse
import json

from src.config.experiment_config import ExperimentConfig


def main() -> None:
    """Create E2 configuration files from the selected baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/logistic_c_0_001.json",
    )
    args = parser.parse_args()

    base_path = ExperimentConfig.resolve_path(
        args.base_config
    )
    base = json.loads(
        base_path.read_text(encoding="utf-8")
    )

    # Validate the source configuration before generating new ones.
    ExperimentConfig.from_json(base_path)

    configs_dir = (
        ExperimentConfig.project_root() / "configs"
    )
    configs_dir.mkdir(parents=True, exist_ok=True)

    pending = []

    for k in (5, 10, 20, 50, 150):
        experiment_id = f"e2_correlation_k_{k}"
        path = configs_dir / f"{experiment_id}.json"

        settings = {
            **base,
            "experiment_id": experiment_id,
            "regularization_c": 0.001,
            "feature_selection_k": k,
            "output_dir": (
                f"outputs/results/{experiment_id}"
            ),
        }

        ExperimentConfig(**settings)

        pending.append((path, settings))

    # Check every destination before creating any files.
    existing = [
        str(path)
        for path, _ in pending
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "E2 configuration files already exist. "
            f"Review them before replacing: {existing}"
        )

    for path, settings in pending:
        with path.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                settings,
                stream,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")

        print(f"Created: {path}")


if __name__ == "__main__":
    main()