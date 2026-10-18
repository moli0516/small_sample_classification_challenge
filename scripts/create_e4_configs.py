"""Create the four predefined E4 L1 experiments without training."""
import argparse
import json
from src.config.experiment_config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', default='configs/logistic_c_0_001.json')
    args = parser.parse_args()
    source = ExperimentConfig.resolve_path(args.base_config)
    source_config = ExperimentConfig.from_json(source)
    if source_config.n_splits != 5 or source_config.split_seeds != tuple(range(10)):
        raise ValueError('E4 requires the established five folds and split seeds 0–9.')
    base = json.loads(source.read_text(encoding='utf-8'))
    pending = []
    for label, c in [('0_01', 0.01), ('0_1', 0.1), ('1', 1.0), ('10', 10.0)]:
        experiment_id = f'e4_l1_c_{label}'
        settings = {**base, 'experiment_id': experiment_id,
                    'regularization_c': c, 'feature_selection_k': None,
                    'regularization_type': 'l1', 'solver': 'saga',
                    'model_seed': 42, 'max_iter': 10000, 'tol': 1e-4,
                    'output_dir': f'outputs/results/{experiment_id}'}
        ExperimentConfig(**settings)
        path = ExperimentConfig.project_root() / 'configs' / f'{experiment_id}.json'
        pending.append((path, settings))
    existing = [str(path) for path, _ in pending if path.exists()]
    if existing:
        raise FileExistsError(f'Review existing E4 configurations before replacing: {existing}')
    for path, settings in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(settings, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(f'Created: {path}')


if __name__ == '__main__':
    main()
