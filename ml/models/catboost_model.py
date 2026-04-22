"""
CatBoost demand forecaster.

Uses CatBoost's ordered boosting with native categorical feature support for
product SKU, weekday, month, quarter, and weekend/month-end flags.
Applies log1p target transform. During final refit it uses a chronological
validation slice to select the best number of trees, then retrains on the
full training window using the chosen iteration count.
"""

import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from ml.config import CATBOOST_CAT_FEATURES, CATBOOST_PARAMS, FEATURE_COLS, USE_LOG_TRANSFORM, VALIDATION_RATIO
from ml.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class CatBoostForecaster(BaseForecaster):
    """CatBoost regressor for daily demand forecasting.

    Marks product identity and calendar columns as categorical so CatBoost
    applies its ordered target encoding — this avoids leakage and often
    outperforms label-encoded integer features.

    Args:
        params: Hyperparameter dict for :class:`catboost.CatBoostRegressor`.
                Defaults to :data:`ml.config.CATBOOST_PARAMS`.
        use_log_transform: Apply ``log1p`` / ``expm1`` transform on the target.
    """

    def __init__(
        self,
        params: dict | None = None,
        use_log_transform: bool = USE_LOG_TRANSFORM,
    ) -> None:
        self._params = dict(params if params is not None else CATBOOST_PARAMS)
        self._feature_cols: list[str] = FEATURE_COLS
        self._cat_feature_names: list[str] = CATBOOST_CAT_FEATURES
        # Precompute column indices for Pool construction
        self._cat_indices: list[int] = [
            self._feature_cols.index(c)
            for c in self._cat_feature_names
            if c in self._feature_cols
        ]
        self._use_log: bool = use_log_transform
        self._model: CatBoostRegressor | None = None
        self._is_fitted: bool = False
        self._training_metadata: dict[str, int | float | bool] = {}

    @property
    def name(self) -> str:
        return "CatBoost"

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cast categorical columns to int (CatBoost requires no float NaN)."""
        X = X[self._feature_cols].copy()
        for col in self._cat_feature_names:
            if col in X.columns:
                X[col] = X[col].fillna(0).astype(int)
        return X

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CatBoostForecaster":
        logger.info(
            "[%s] Training on %d rows × %d features (log_transform=%s, cat=%s)",
            self.name, len(X_train), len(self._feature_cols), self._use_log,
            self._cat_feature_names,
        )
        X = self._prepare_X(X_train)
        y_raw = y_train.to_numpy()
        y = np.log1p(y_raw) if self._use_log else y_raw

        fit_params = dict(self._params)
        early_stopping_rounds = int(fit_params.pop("early_stopping_rounds", 0))

        validation_rows = max(1, int(len(X) * VALIDATION_RATIO))
        used_early_stopping = early_stopping_rounds > 0 and validation_rows < len(X)
        selected_iterations = int(fit_params.get("iterations", 100))

        if used_early_stopping:
            X_fit, X_val = X.iloc[:-validation_rows], X.iloc[-validation_rows:]
            y_fit, y_val = y[:-validation_rows], y[-validation_rows:]
            train_pool = Pool(X_fit, label=y_fit, cat_features=self._cat_indices)
            eval_pool = Pool(X_val, label=y_val, cat_features=self._cat_indices)

            selection_model = CatBoostRegressor(
                **fit_params,
                early_stopping_rounds=early_stopping_rounds,
            )
            selection_model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
            best_iteration = selection_model.get_best_iteration()
            if best_iteration is not None and best_iteration >= 0:
                selected_iterations = max(1, int(best_iteration) + 1)

        full_pool = Pool(X, label=y, cat_features=self._cat_indices)
        self._model = CatBoostRegressor(**{**fit_params, "iterations": selected_iterations})
        self._model.fit(full_pool)

        logger.info("[%s] Training complete | selected_iterations=%d", self.name, selected_iterations)
        self._training_metadata = {
            "used_early_stopping": used_early_stopping,
            "validation_rows": int(validation_rows if used_early_stopping else 0),
            "selected_iterations": int(selected_iterations),
        }
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Call fit() before predict().")
        X_pred = self._prepare_X(X)
        pool = Pool(X_pred, cat_features=self._cat_indices)
        raw = self._model.predict(pool)
        preds = np.expm1(raw) if self._use_log else raw
        return np.clip(preds, 0.0, None)

    def get_feature_importance(self) -> dict[str, float]:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Call fit() before get_feature_importance().")
        scores = dict(zip(self._feature_cols, self._model.get_feature_importance()))
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    def get_training_metadata(self) -> dict[str, int | float | bool] | None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Call fit() before get_training_metadata().")
        return dict(self._training_metadata)
