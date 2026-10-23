"""Lazy, strict model registration and checked probability averaging."""
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
import joblib
import numpy as np


class RegisteredModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Fit the complete training-only procedure."""

    @abstractmethod
    def predict_proba(self, X):
        """Return an n-by-2 probability matrix ordered by classes_."""

    def positive_probability(self, X):
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("Expected class order [0, 1].")
        p = np.asarray(self.predict_proba(X), dtype=np.float64)
        if p.shape != (len(X), 2) or not np.isfinite(p).all():
            raise ValueError("Invalid probability shape or nonfinite values.")
        if ((p < 0) | (p > 1)).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Invalid probability range/sums.")
        return p[:, 1]

    @property
    def classes_(self):
        return self.model.classes_

    def _remember(self, X):
        if not hasattr(X, "columns") or not X.columns.is_unique:
            raise ValueError("Expected a DataFrame with unique predictor names.")
        self.features_ = list(X.columns)

    def _check(self, X):
        if list(X.columns) != self.features_:
            raise ValueError("Predictor names/order differ from training.")

    def diagnostics(self):
        return {"feature_count": len(self.features_)}

    def save(self, path):
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path):
        model = joblib.load(Path(path))
        if not isinstance(model, RegisteredModel):
            raise TypeError("Artifact is not a registered model.")
        return model


class ModelRegistry:
    _models = {}

    @classmethod
    def register(cls, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model name must not be empty.")
        def decorate(model_type):
            if not isinstance(model_type, type) or not issubclass(model_type, RegisteredModel):
                raise TypeError("Registered classes must inherit RegisteredModel.")
            if name in cls._models:
                raise ValueError(f"Duplicate model registration: {name}")
            cls._models[name] = model_type
            return model_type
        return decorate

    @classmethod
    def names(cls):
        importlib.import_module("src.modeling.registered_models")
        return sorted(cls._models)

    @classmethod
    def create(cls, name, *, context=None, parameters=None):
        cls.names()
        if name not in cls._models:
            raise ValueError(f"Unknown model {name!r}; available: {cls.names()}")
        return cls._models[name](context=context, parameters=parameters)


def soft_vote(probabilities, members, weights):
    if not members or len(members) != len(set(members)) or len(members) != len(weights):
        raise ValueError("Members/weights must have equal nonzero length and unique names.")
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not np.isfinite(w).all() or (w < 0).any() or w.max(initial=0) <= 0:
        raise ValueError("Weights must be finite, nonnegative and have positive total.")
    if any(name not in probabilities for name in members):
        raise ValueError("Missing member predictions.")
    arrays = [np.asarray(probabilities[name], dtype=np.float64) for name in members]
    if any(a.ndim != 1 or a.size == 0 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Inconsistent/empty prediction vectors.")
    if any(not np.isfinite(a).all() or ((a < 0) | (a > 1)).any() for a in arrays):
        raise ValueError("Invalid probability values.")
    scaled = w / w.max()
    return np.average(np.stack(arrays), axis=0, weights=scaled / scaled.sum())
