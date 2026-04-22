"""Leakage-safe target encoding utilities for stock-code features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from ml.config import (
    TARGET_ENCODING_FEATURE_COL,
    TARGET_ENCODING_KEY_COL,
    TARGET_ENCODING_SMOOTHING,
)


def build_target_encoding(
    keys: pd.Series,
    targets: pd.Series | np.ndarray,
    *,
    smoothing: float = TARGET_ENCODING_SMOOTHING,
) -> tuple[dict[Any, float], float]:
    """Fit a smoothed target-encoding map for a categorical key."""
    key_series = pd.Series(keys).reset_index(drop=True)
    target_series = pd.Series(targets).reset_index(drop=True).astype(float)
    global_mean = float(target_series.mean())

    stats = (
        pd.DataFrame({"key": key_series, "target": target_series})
        .groupby("key", dropna=False)["target"]
        .agg(["mean", "count"])
    )
    smoothed = (
        (stats["mean"] * stats["count"]) + (global_mean * smoothing)
    ) / (stats["count"] + smoothing)
    return smoothed.to_dict(), global_mean


def append_target_encoding_feature(
    X: pd.DataFrame,
    *,
    mapping: dict[Any, float],
    global_mean: float,
    key_col: str = TARGET_ENCODING_KEY_COL,
    feature_col: str = TARGET_ENCODING_FEATURE_COL,
) -> pd.DataFrame:
    """Append the fitted target-encoding feature to a feature frame."""
    if key_col not in X.columns:
        raise ValueError(f"Target encoding requires '{key_col}' in features; got {list(X.columns)}")

    X_encoded = X.copy()
    encoded = X_encoded[key_col].map(mapping)
    X_encoded[feature_col] = encoded.fillna(global_mean).astype(float)
    return X_encoded


class TargetEncodingRegressor(BaseEstimator, RegressorMixin):
    """Wrap a regressor and add a smoothed target-encoding feature in fit/predict."""

    def __init__(
        self,
        base_estimator: RegressorMixin,
        *,
        key_col: str = TARGET_ENCODING_KEY_COL,
        feature_col: str = TARGET_ENCODING_FEATURE_COL,
        smoothing: float = TARGET_ENCODING_SMOOTHING,
    ) -> None:
        self.base_estimator = base_estimator
        self.key_col = key_col
        self.feature_col = feature_col
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "TargetEncodingRegressor":
        X_df = pd.DataFrame(X).copy()
        self._mapping, self._global_mean = build_target_encoding(
            X_df[self.key_col],
            y,
            smoothing=self.smoothing,
        )
        self._model = clone(self.base_estimator)
        self._model.fit(
            append_target_encoding_feature(
                X_df,
                mapping=self._mapping,
                global_mean=self._global_mean,
                key_col=self.key_col,
                feature_col=self.feature_col,
            ),
            y,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_df = pd.DataFrame(X).copy()
        preds = self._model.predict(
            append_target_encoding_feature(
                X_df,
                mapping=self._mapping,
                global_mean=self._global_mean,
                key_col=self.key_col,
                feature_col=self.feature_col,
            )
        )
        return np.asarray(preds, dtype=float)
