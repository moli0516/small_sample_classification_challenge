"""Generate the four new configurations for the E3 experiment."""

import json

from src.config.experiment_config import ExperimentConfig


def main() -> None:
    """Create E3 configurations without overwriting existing files."""
    project_root = ExperimentConfig.project_root()
    configs_dir = project_root / "configs"

    c_candidates = (
        ("0_01", 0.01),
        ("0_1", 0.1),
    )

    pending = []

    for k in (10, 20):
        source_path = (
            configs_dir / f"e2_correlation_k_{k}.json"
        )

        source_config = ExperimentConfig.from_json(
            source_path
        )

        if source_config.feature_selection_k != k:
            raise ValueError(
                f"Unexpected feature_selection_k in {source_path}."
            )

        if source_config.regularization_c != 0.001:
            raise ValueError(
                f"E2 source must use C=0.001: {source_path}."
            )

        if (
            source_config.n_splits != 5
            or source_config.split_seeds != tuple(range(10))
        ):
            raise ValueError(
                f"Unexpected E2 validation settings: {source_path}."
            )

        source_settings = json.loads(
            source_path.read_text(encoding="utf-8")
        )

        for c_label, c_value in c_candidates:
            experiment_id = (
                f"e3_correlation_k_{k}_c_{c_label}"
            )

            settings = {
                **source_settings,
                "experiment_id": experiment_id,
                "regularization_c": c_value,
                "output_dir": (
                    f"outputs/results/{experiment_id}"
                ),
            }

            # Validate every configuration before writing files.
            ExperimentConfig(**settings)

            destination = (
                configs_dir / f"{experiment_id}.json"
            )

            pending.append((destination, settings))

    existing = [
        str(path)
        for path, _ in pending
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "E3 configurations already exist. "
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