"""
Naive lag-1 baseline forecaster — Phase 3.

Predicts tomorrow's demand as today's observed demand (i.e. the ``lag_1``
feature).  Requires no training and serves as the lower-bound benchmark
that tree-based models must beat.
"""

import logging

import numpy as np
import pandas as pd

from ml.models.base import BaseForecaster

logger = logging.getLogger(__name__)

_LAG_COL = "lag_1"


class NaiveForecaster(BaseForecaster):
    """Predicts demand as the previous day's quantity (``lag_1``).

    This is a zero-parameter model — ``fit`` is a no-op.  The model is
    included so that evaluation infrastructure can treat all models
    identically.
    """

    @property
    def name(self) -> str:
        return "Naive (lag-1)"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "NaiveForecaster":
        """No-op — the naive model has no parameters to learn.

        Args:
            X_train: Unused (kept for interface compatibility).
            y_train: Unused.

        Returns:
            Self.
        """
        if _LAG_COL not in X_train.columns:
            raise ValueError(
                f"NaiveForecaster requires '{_LAG_COL}' column in X_train; "
                f"got columns: {list(X_train.columns)}"
            )
        logger.info("[%s] fit() — no-op baseline", self.name)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the ``lag_1`` column as the prediction.

        Args:
            X: Feature DataFrame containing at least the ``lag_1`` column.

        Returns:
            Array of predicted quantities (clipped to ≥ 0).
        """
        if _LAG_COL not in X.columns:
            raise ValueError(
                f"NaiveForecaster requires '{_LAG_COL}' column; "
                f"got: {list(X.columns)}"
            )
        preds = X[_LAG_COL].to_numpy(dtype=float)
        return np.clip(preds, 0.0, None)
