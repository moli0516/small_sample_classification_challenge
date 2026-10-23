"""Complete model procedures; optional native libraries import only when used."""
from inspect import signature
from src.modeling.model_registry import RegisteredModel, ModelRegistry


@ModelRegistry.register("l1_pipeline")
class L1Model(RegisteredModel):
    def __init__(self, *, context=None, parameters=None):
        if parameters:
            raise ValueError("L1 uses the existing resolved configuration; do not override it here.")
        self.config = None if context is None else context["linear"]

    def fit(self, X, y):
        from src.modeling.pipeline_factory import PipelineFactory
        self._remember(X)
        self.model = PipelineFactory().create(self.config)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        self._check(X)
        return self.model.predict_proba(X)


@ModelRegistry.register("xgb_topk")
class XGBTopKModel(RegisteredModel):
    def __init__(self, *, context=None, parameters=None):
        if parameters:
            raise ValueError("XGBoost uses frozen constructor parameters.")
        self.parameters = {} if context is None else dict(context["candidate"]["estimator_parameters"])
        self.k = 20 if context is None else context["candidate"]["k"]

    def fit(self, X, y):
        from xgboost import XGBClassifier
        from src.features.gain_top_k_selector import GainTopKSelector
        self._remember(X)
        self.ranker = XGBClassifier(**self.parameters).fit(X, y)
        self.selector = GainTopKSelector(k=self.k).fit_from_booster(self.ranker.get_booster(), self.features_)
        self.selected_ = self.selector.get_feature_names_out().tolist()
        self.model = XGBClassifier(**self.parameters).fit(self.selector.transform(X), y)
        for fitted in (self.ranker, self.model):
            if fitted.get_booster().num_boosted_rounds() != self.parameters["n_estimators"]:
                raise ValueError("XGBoost boosting-round count changed.")
        return self

    def predict_proba(self, X):
        self._check(X)
        return self.model.predict_proba(self.selector.transform(X))

    def diagnostics(self):
        return dict(super().diagnostics(), selected_features=self.selected_,
                    actual_rounds=self.model.get_booster().num_boosted_rounds())


@ModelRegistry.register("lgbm_classifier")
class LightGBMModel(RegisteredModel):
    def __init__(self, *, context=None, parameters=None):
        self.parameters = dict(parameters or {})

    def fit(self, X, y):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("Install the optional dependency: python -m pip install -r requirements-registry.txt") from exc
        supported = set(signature(LGBMClassifier).parameters) | {"verbosity", "deterministic", "force_col_wise"}
        if set(self.parameters) - supported:
            raise ValueError(f"Unsupported LightGBM constructor keys: {set(self.parameters) - supported}")
        self._remember(X)
        self.model = LGBMClassifier(**self.parameters)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        self._check(X)
        return self.model.predict_proba(X)

    def diagnostics(self):
        return dict(super().diagnostics(), actual_rounds=int(self.model.n_estimators_),
                    actual_trees=int(self.model.booster_.num_trees()))
