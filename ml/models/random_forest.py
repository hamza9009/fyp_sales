"""
Random Forest demand forecaster — Phase 3.

Applies a ``log1p`` transform to the training target and ``expm1`` to
predictions so that RMSE is computed on a symmetric, low-skew scale.
Predictions are always returned on the *original* (unit) scale.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml.config import FEATURE_COLS, RF_PARAMS, USE_LOG_TRANSFORM
from ml.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class RandomForestForecaster(BaseForecaster):
    """Random Forest regressor for daily demand forecasting.

    Args:
        params: Hyperparameter dict for
                :class:`sklearn.ensemble.RandomForestRegressor`.
                Defaults to :data:`ml.config.RF_PARAMS`.
        use_log_transform: If ``True`` (default when
                           :data:`ml.config.USE_LOG_TRANSFORM` is set),
                           apply ``log1p`` to y before fitting and
                           ``expm1`` to predictions.
    """

    def __init__(
        self,
        params: dict | None = None,
        use_log_transform: bool = USE_LOG_TRANSFORM,
    ) -> None:
        self._model = RandomForestRegressor(**(params if params is not None else RF_PARAMS))
        self._feature_cols: list[str] = FEATURE_COLS
        self._use_log: bool = use_log_transform
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        return "Random Forest"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestForecaster":
        logger.info(
            "[%s] Training on %d rows × %d features (log_transform=%s)",
            self.name, len(X_train), len(self._feature_cols), self._use_log,
        )
        y = np.log1p(y_train.to_numpy()) if self._use_log else y_train.to_numpy()
        self._model.fit(X_train[self._feature_cols], y)
        self._is_fitted = True
        logger.info("[%s] Training complete", self.name)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        raw = self._model.predict(X[self._feature_cols])
        preds = np.expm1(raw) if self._use_log else raw
        return np.clip(preds, 0.0, None)

    def get_feature_importance(self) -> dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_feature_importance().")
        return dict(zip(self._feature_cols, self._model.feature_importances_.tolist()))
