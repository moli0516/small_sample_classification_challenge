"""Build baseline, correlation-filtered, and L1 pipelines."""
from inspect import signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config.experiment_config import ExperimentConfig
from src.features import CorrelationSelector


class PipelineFactory:
    """Construct unfitted pipelines without altering old L2 settings."""

    def create(self, config: ExperimentConfig) -> Pipeline:
        if config.model_name != 'logistic_regression':
            raise ValueError(f'Unsupported model_name: {config.model_name!r}.')
        parameters = {
            'C': config.regularization_c,
            'solver': config.solver,
            'max_iter': config.max_iter,
            'random_state': config.model_seed,
            'tol': config.tol,
        }
        api = signature(LogisticRegression).parameters
        penalty_parameter = api.get('penalty')
        new_api = penalty_parameter is None or penalty_parameter.default == 'deprecated'
        if new_api:
            parameters['l1_ratio'] = 1.0 if config.regularization_type == 'l1' else 0.0
        else:
            parameters['penalty'] = config.regularization_type
        steps = []
        if config.feature_selection_k is not None:
            steps.append(('selector', CorrelationSelector(k=config.feature_selection_k)))
        steps.extend([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**parameters)),
        ])
        return Pipeline(steps)
