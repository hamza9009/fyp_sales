"""
LightGBM demand forecaster.

Applies log1p target transform. During final refit it uses a chronological
validation slice to determine the best boosting round via early stopping,
then retrains on the full training window using the selected round count.
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.config import (
    FEATURE_COLS,
    LGB_PARAMS,
    TARGET_ENCODING_FEATURE_COL,
    TARGET_ENCODING_KEY_COL,
    USE_LOG_TRANSFORM,
    VALIDATION_RATIO,
)
from ml.models.base import BaseForecaster
from ml.target_encoding import append_target_encoding_feature, build_target_encoding

logger = logging.getLogger(__name__)


class LightGBMForecaster(BaseForecaster):
    """LightGBM gradient boosting forecaster for daily demand.

    Args:
        params: Hyperparameter dict for :class:`lightgbm.LGBMRegressor`.
                ``early_stopping_rounds`` is extracted and passed via callback.
                Defaults to :data:`ml.config.LGB_PARAMS`.
        use_log_transform: Apply ``log1p`` / ``expm1`` transform on the target.
    """

    def __init__(
        self,
        params: dict | None = None,
        use_log_transform: bool = USE_LOG_TRANSFORM,
    ) -> None:
        resolved = dict(params if params is not None else LGB_PARAMS)
        self._early_stopping_rounds = resolved.pop("early_stopping_rounds", 50)
        self._params = resolved
        self._model = lgb.LGBMRegressor(**resolved)
        self._base_feature_cols: list[str] = FEATURE_COLS
        self._feature_cols: list[str] = [*FEATURE_COLS, TARGET_ENCODING_FEATURE_COL]
        self._use_log: bool = use_log_transform
        self._is_fitted: bool = False
        self._training_metadata: dict[str, int | float | bool] = {}
        self._target_encoding_map: dict[int | float | str, float] = {}
        self._target_encoding_global_mean: float = 0.0

    @property
    def name(self) -> str:
        return "LightGBM"

    def _build_model(self, params: dict | None = None) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(**(params if params is not None else self._params))

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LightGBMForecaster":
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

        validation_rows = max(1, int(len(X_encoded) * VALIDATION_RATIO))
        used_early_stopping = self._early_stopping_rounds > 0 and validation_rows < len(X_encoded)
        selected_estimators = int(self._params.get("n_estimators", 100))

        if used_early_stopping:
            X_fit, X_val = X_encoded.iloc[:-validation_rows], X_encoded.iloc[-validation_rows:]
            y_fit, y_val = y[:-validation_rows], y[-validation_rows:]
            selection_model = self._build_model()
            selection_model.fit(
                X_fit,
                y_fit,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(self._early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            selected_estimators = max(
                1,
                int(getattr(selection_model, "best_iteration_", selected_estimators)),
            )

        final_params = dict(self._params)
        final_params["n_estimators"] = selected_estimators
        self._model = self._build_model(final_params)
        self._model.fit(X_encoded, y)

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
        scores = dict(zip(self._feature_cols, self._model.feature_importances_))
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    def get_training_metadata(self) -> dict[str, int | float | bool] | None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_training_metadata().")
        return dict(self._training_metadata)
