"""
XGBoost demand forecaster — Phase 3.

Applies ``log1p`` target transform. During final refit it uses a
chronological validation slice to determine the best boosting round, then
retrains on the full training window using the selected round count.
"""

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ml.config import (
    FEATURE_COLS,
    TARGET_ENCODING_FEATURE_COL,
    TARGET_ENCODING_KEY_COL,
    USE_LOG_TRANSFORM,
    VALIDATION_RATIO,
    XGB_PARAMS,
)
from ml.models.base import BaseForecaster
from ml.target_encoding import append_target_encoding_feature, build_target_encoding

logger = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """XGBoost regressor for daily demand forecasting.

    Args:
        params: Hyperparameter dict for :class:`xgboost.XGBRegressor`.
                ``early_stopping_rounds`` is included here (XGBoost 2.x+ API).
                Defaults to :data:`ml.config.XGB_PARAMS`.
        use_log_transform: Apply ``log1p`` / ``expm1`` transform.
    """

    def __init__(
        self,
        params: dict | None = None,
        use_log_transform: bool = USE_LOG_TRANSFORM,
    ) -> None:
        resolved = dict(params if params is not None else XGB_PARAMS)
        if "early_stopping_rounds" not in resolved:
            resolved["early_stopping_rounds"] = 50
        self._params = resolved
        self._model = XGBRegressor(**resolved)
        self._base_feature_cols: list[str] = FEATURE_COLS
        self._feature_cols: list[str] = [*FEATURE_COLS, TARGET_ENCODING_FEATURE_COL]
        self._use_log: bool = use_log_transform
        self._is_fitted: bool = False
        self._training_metadata: dict[str, int | float | bool] = {}
        self._target_encoding_map: dict[int | float | str, float] = {}
        self._target_encoding_global_mean: float = 0.0

    @property
    def name(self) -> str:
        return "XGBoost"

    def _build_model(self, params: dict | None = None) -> XGBRegressor:
        return XGBRegressor(**(params if params is not None else self._params))

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostForecaster":
        logger.info(
            "[%s] Training on %d rows × %d features (log_transform=%s)",
            self.name, len(X_train), len(self._feature_cols), self._use_log,
        )
        X = X_train[self._base_feature_cols]
        y_raw = y_train.to_numpy()
        y = np.log1p(y_raw) if self._use_log else y_raw
        self._target_encoding_map, self._target_encoding_global_mean = build_target_encoding(
            X[TARGET_ENCODING_KEY_COL],
            y,
        )
        X_encoded = append_target_encoding_feature(
            X,
            mapping=self._target_encoding_map,
            global_mean=self._target_encoding_global_mean,
            key_col=TARGET_ENCODING_KEY_COL,
            feature_col=TARGET_ENCODING_FEATURE_COL,
        )
        fit_params = dict(self._params)
        early_stopping_rounds = int(fit_params.pop("early_stopping_rounds", 0))

        validation_rows = max(1, int(len(X_encoded) * VALIDATION_RATIO))
        used_early_stopping = early_stopping_rounds > 0 and validation_rows < len(X_encoded)
        selected_estimators = int(fit_params.get("n_estimators", 100))

        if used_early_stopping:
            X_fit, X_val = X_encoded.iloc[:-validation_rows], X_encoded.iloc[-validation_rows:]
            y_fit, y_val = y[:-validation_rows], y[-validation_rows:]
            selection_model = self._build_model({**fit_params, "early_stopping_rounds": early_stopping_rounds})
            selection_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
            best_iteration = getattr(selection_model, "best_iteration", None)
            if best_iteration is not None:
                selected_estimators = max(1, int(best_iteration) + 1)

        final_params = dict(fit_params)
        final_params["n_estimators"] = selected_estimators
        self._model = self._build_model(final_params)
        self._model.fit(X_encoded, y, verbose=False)

        logger.info("[%s] Training complete | selected_n_estimators=%d", self.name, selected_estimators)
        self._training_metadata = {
            "used_early_stopping": used_early_stopping,
            "validation_rows": int(validation_rows if used_early_stopping else 0),
            "selected_n_estimators": int(selected_estimators),
        }
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X_encoded = append_target_encoding_feature(
            X[self._base_feature_cols],
            mapping=self._target_encoding_map,
            global_mean=self._target_encoding_global_mean,
            key_col=TARGET_ENCODING_KEY_COL,
            feature_col=TARGET_ENCODING_FEATURE_COL,
        )
        raw = self._model.predict(X_encoded[self._feature_cols])
        preds = np.expm1(raw) if self._use_log else raw
        return np.clip(preds, 0.0, None)

    def get_feature_importance(self) -> dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_feature_importance().")
        scores = self._model.get_booster().get_score(importance_type="gain")
        total = sum(scores.values()) or 1.0
        return {k: scores.get(k, 0.0) / total for k in self._feature_cols}

    def get_training_metadata(self) -> dict[str, int | float | bool] | None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_training_metadata().")
        return dict(self._training_metadata)
