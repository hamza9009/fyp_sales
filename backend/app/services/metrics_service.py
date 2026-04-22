"""
Model metrics service — Phase 4.

Reads the Phase 3 metrics report JSON and returns a structured
comparison of all trained models.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_METRICS_JSON = _PROJECT_ROOT / "ml" / "artifacts" / "metrics_report.json"


def get_model_metrics() -> dict:
    """Load and structure the Phase 3 model metrics report.

    Returns:
        Dict matching :class:`~app.schemas.models.ModelMetricsResponse`.

    Raises:
        FileNotFoundError: If the metrics report has not been generated yet.
    """
    if not _METRICS_JSON.exists():
        raise FileNotFoundError(
            f"Metrics report not found at {_METRICS_JSON}. "
            "Run 'python -m ml.pipeline' first."
        )

    with _METRICS_JSON.open() as fh:
        report = json.load(fh)

    best_model = report.get("best_model", "Unknown")
    split = report.get("split", {})
    models_raw = report.get("models", {})

    # Sort by RMSE ascending (best first) and assign ranks
    sorted_models = sorted(models_raw.items(), key=lambda kv: kv[1].get("rmse", float("inf")))

    models = [
        {
            "model_name": name,
            "mae": float(metrics.get("mae", 0.0)),
            "rmse": float(metrics.get("rmse", 0.0)),
            "train_time_sec": float(metrics.get("train_time_sec", 0.0)),
            "rank": rank + 1,
            "is_best": name == best_model,
        }
        for rank, (name, metrics) in enumerate(sorted_models)
    ]

    return {
        "best_model": best_model,
        "split": {
            "cutoff_date": split.get("cutoff_date", ""),
            "train_rows": int(split.get("train_rows", 0)),
            "test_rows": int(split.get("test_rows", 0)),
            "train_products": int(split.get("train_products", 0)),
            "test_products": int(split.get("test_products", 0)),
        },
        "models": models,
    }
