"""
Model loader service — Phase 4.

Loads the best trained model artifact and the stock_code LabelEncoder once
at application startup and caches both for the lifetime of the process.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

# ── Feature columns (must match ml/config.py exactly) ─────────────────────────
FEATURE_COLS: list[str] = [
    "stock_code_encoded",
    "unit_price",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
    "day_of_week",
    "month",
    "quarter",
    "is_weekend",
    "is_month_end",
    "day_of_year",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ARTIFACTS   = _PROJECT_ROOT / "ml" / "artifacts"
_MODEL_PATH  = _ARTIFACTS / "best_model.joblib"
_ENCODER_PATH = _ARTIFACTS / "stock_code_encoder.joblib"
_METRICS_JSON = _ARTIFACTS / "metrics_report.json"

# Ensure ml.* classes are importable when joblib unpickles the model
_project_root_str = str(_PROJECT_ROOT)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

_model: Any | None = None
_encoder: Any | None = None
_model_name: str = "Unknown"


def load_model(artifact_path: str | Path | None = None) -> None:
    """Load model + encoder into the module-level cache.

    Called once from the FastAPI lifespan startup hook.
    """
    global _model, _encoder, _model_name

    model_path = Path(artifact_path) if artifact_path else _MODEL_PATH
    enc_path   = _ENCODER_PATH

    for p in (model_path, enc_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Artifact not found: {p}. "
                "Run 'python -m ml.pipeline' first."
            )

    logger.info("Loading model from %s", model_path)
    _model = joblib.load(model_path)

    logger.info("Loading encoder from %s", enc_path)
    _encoder = joblib.load(enc_path)

    try:
        with _METRICS_JSON.open() as fh:
            _model_name = json.load(fh).get("best_model", "Unknown")
    except Exception:
        _model_name = "Best Model"

    logger.info("Model loaded: %s | encoder classes: %d", _model_name, len(_encoder.classes_))


def get_model() -> Any:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")
    return _model


def get_encoder() -> Any:
    if _encoder is None:
        raise RuntimeError("Encoder not loaded. Call load_model() at startup.")
    return _encoder


def get_model_name() -> str:
    return _model_name


def encode_stock_code(stock_code: str) -> int:
    """Encode a stock_code string to its training-time integer.

    Returns -1 for codes not seen during training (graceful degradation).
    """
    enc = get_encoder()
    try:
        return int(enc.transform([stock_code])[0])
    except ValueError:
        return -1
