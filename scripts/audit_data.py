import argparse
import json
from pathlib import Path

from src.data.data_auditor import DataAuditor
from src.data.data_loader import DataLoader


def main() -> None:
    """Audit the competition CSVs using a project-root-relative configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/data.json')
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = json.loads(config_path.read_text(encoding='utf-8'))
    loader = DataLoader(
        target_column=config.get('target_column'),
        id_columns=config.get('id_columns', []),
        expected_features=config.get('expected_features', 150),
    )
    dataset = loader.load(
        project_root / config['train_path'], project_root / config['test_path']
    )
    summary = DataAuditor().save(dataset, project_root / config['audit_output_dir'])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
