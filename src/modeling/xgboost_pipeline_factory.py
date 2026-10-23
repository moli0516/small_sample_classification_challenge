"""Build XGBoost pipelines; preserve E5b defaults and allow validated trial overrides."""
import math
from sklearn.pipeline import Pipeline


class XGBoostPipelineFactory:
    FIXED_PARAMETERS = dict(
        booster="gbtree", objective="binary:logistic", eval_metric="auc",
        tree_method="hist", learning_rate=0.05, min_child_weight=5,
        reg_lambda=10.0, reg_alpha=0.0, gamma=0.0, subsample=1.0,
        colsample_bytree=1.0, n_jobs=1, verbosity=1,
    )
    OVERRIDABLE = frozenset({"learning_rate", "min_child_weight", "reg_lambda",
                             "subsample", "colsample_bytree"})

    def __init__(self, overrides=None):
        if overrides is not None and not isinstance(overrides, dict):
            raise TypeError("overrides must be a dictionary or None.")
        values = dict(overrides or {})
        unknown = set(values) - self.OVERRIDABLE
        if unknown:
            raise ValueError(f"Unsupported XGBoost overrides: {sorted(unknown)}.")
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero.")
            if name in ("learning_rate", "subsample", "colsample_bytree") and value > 1:
                raise ValueError(f"{name} must not exceed one.")
        self._overrides = values

    @staticmethod
    def check_environment():
        import xgboost
        return xgboost.__version__

    def parameters(self, config, max_depth, n_estimators):
        parameters = dict(self.FIXED_PARAMETERS)
        parameters.update(self._overrides)
        parameters.update(max_depth=max_depth, n_estimators=n_estimators,
                          random_state=config.model_seed)
        return parameters

    def create(self, config, max_depth, n_estimators):
        from xgboost import XGBClassifier
        return Pipeline([("classifier", XGBClassifier(
            **self.parameters(config, max_depth, n_estimators)))])
