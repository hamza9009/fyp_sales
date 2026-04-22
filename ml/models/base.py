"""
Abstract base class for all Phase 3 forecasting models.

Every model must implement ``fit``, ``predict``, and expose a ``name``
property so that the training orchestrator can treat them uniformly.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Common interface for all demand forecasting models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier (e.g. ``"XGBoost"``)."""

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseForecaster":
        """Train the model on the supplied feature matrix and target.

        Args:
            X_train: Training feature matrix (rows × feature_cols).
            y_train: Training target series (``total_quantity``).

        Returns:
            Self — enables method chaining.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the supplied feature matrix.

        Predictions are clipped to ``≥ 0`` (demand cannot be negative).

        Args:
            X: Feature matrix with the same columns used during ``fit``.

        Returns:
            1-D NumPy array of predicted quantities.
        """

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return feature importance scores keyed by column name.

        Tree-based models should override this.  Returns ``None`` by default
        (e.g. for the naive baseline which has no learnable importances).
        """
        return None

    def get_training_metadata(self) -> dict[str, int | float | bool] | None:
        """Return model-specific fit metadata for reporting.

        Boosted models override this to expose the selected number of boosting
        rounds and whether early stopping was used during refit.
        """
        return None
