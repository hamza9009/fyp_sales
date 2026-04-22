"""
Model evaluation utilities — Phase 3.

Computes MAE and RMSE for each model and produces a comparison
table sorted by RMSE (primary metric for demand forecasting).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    model_name: str = "",
) -> dict[str, float]:
    """Compute MAE and RMSE for a single model's predictions.

    Args:
        y_true: Ground-truth quantity values.
        y_pred: Predicted quantity values (same length as ``y_true``).
        model_name: Optional label used only for logging.

    Returns:
        Dict with keys ``"mae"`` and ``"rmse"``.
        All values are rounded to 4 decimal places.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different lengths.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; "
            f"got {y_true_arr.shape} vs {y_pred_arr.shape}"
        )

    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }

    label = f"[{model_name}] " if model_name else ""
    logger.info(
        "%sMAE=%.4f  RMSE=%.4f",
        label, metrics["mae"], metrics["rmse"],
    )
    return metrics


def compare_models(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a ranked comparison table from per-model metric dicts.

    Args:
        results: Mapping of model name → ``{"mae": …, "rmse": …}``.

    Returns:
        DataFrame sorted by ``rmse`` ascending, with a ``rank`` column and
        a ``best`` boolean column marking the top model.
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "model": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }
        )

    comparison = (
        pd.DataFrame(rows)
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    comparison.insert(0, "rank", range(1, len(comparison) + 1))
    comparison["best"] = comparison["rank"] == 1

    logger.info("Model comparison (sorted by RMSE):\n%s", comparison.to_string(index=False))
    return comparison
