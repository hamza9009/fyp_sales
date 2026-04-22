"""Simple averaging ensemble over fitted forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.models.base import BaseForecaster


class AveragingEnsembleForecaster(BaseForecaster):
    """Average predictions from multiple fitted forecasting models."""

    def __init__(
        self,
        models: list[BaseForecaster],
        *,
        name: str = "LightGBM + XGBoost Ensemble",
        already_fitted: bool = False,
    ) -> None:
        if not models:
            raise ValueError("AveragingEnsembleForecaster requires at least one base model.")
        self._models = models
        self._name = name
        self._is_fitted = already_fitted

    @property
    def name(self) -> str:
        return self._name

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "AveragingEnsembleForecaster":
        if not self._is_fitted:
            for model in self._models:
                model.fit(X_train, y_train)
            self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        stacked = np.vstack([model.predict(X) for model in self._models])
        return np.clip(stacked.mean(axis=0), 0.0, None)

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_feature_importance().")

        importances = [model.get_feature_importance() for model in self._models]
        importances = [imp for imp in importances if imp]
        if not importances:
            return None

        feature_keys = list(importances[0].keys())
        averaged = {
            key: float(np.mean([imp.get(key, 0.0) for imp in importances]))
            for key in feature_keys
        }
        total = sum(averaged.values()) or 1.0
        return {key: value / total for key, value in averaged.items()}

    def get_training_metadata(self) -> dict[str, int | float | bool] | None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_training_metadata().")
        return {
            "component_count": len(self._models),
            "averaging_method": "mean",
        }
