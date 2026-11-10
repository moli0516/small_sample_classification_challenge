"""Unregistered frozen Elastic-net procedure; one class per module."""
from inspect import signature
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.modeling.model_registry import RegisteredModel


class ElasticnetTrialModel(RegisteredModel):
    def __init__(self, parameters):
        if set(parameters) != {"C", "l1_ratio"}:
            raise ValueError("Expected only C and l1_ratio")
        self.parameters = dict(parameters)

    def fit(self, x, y):
        self._remember(x)
        params = dict(self.parameters, solver="saga", max_iter=20000, tol=1e-4, random_state=42)
        penalty = signature(LogisticRegression).parameters.get("penalty")
        if penalty is not None and penalty.default != "deprecated":
            params["penalty"] = "elasticnet"
        self.model = Pipeline([("scaler", StandardScaler()),
                               ("classifier", LogisticRegression(**params))])
        self.model.fit(x, y)
        return self

    def predict_proba(self, x):
        self._check(x)
        return self.model.predict_proba(x)

    def diagnostics(self):
        fitted = self.model.named_steps["classifier"]
        return dict(super().diagnostics(), parameters=self.parameters,
                    nonzero_coefficients=int(np.count_nonzero(fitted.coef_)),
                    n_iter=int(fitted.n_iter_.max()))
