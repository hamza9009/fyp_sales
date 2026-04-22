"""
Forecast service — Phase 4.

Generates a multi-step demand forecast for a given product using the
Phase 3 trained model.

Algorithm
---------
1. Load the last 28 actual daily quantities for the product (from the
   Phase 2 sales_daily Parquet).
2. Starting from the day after the last observed date, predict one day
   at a time, feeding each prediction back into the rolling window.
3. Lag and rolling features are recomputed from the rolling window at
   every step — no target leakage.
4. Time calendar features (day_of_week, month, etc.) are computed from
   the actual forecast date.
"""

import logging
from collections import deque
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.services import data_store, model_loader

logger = logging.getLogger(__name__)

_LAGS = [1, 7, 14, 28]
_WINDOWS = [7, 14, 28]


def _is_month_end(d: date) -> int:
    """Return 1 if ``d`` is the last day of its month, else 0."""
    next_day = d + timedelta(days=1)
    return int(next_day.month != d.month)


def _build_feature_row(
    window: deque,
    forecast_date: date,
    stock_code_encoded: int,
    unit_price: float,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame from the rolling window + date.

    Args:
        window: Deque of daily quantities (oldest → newest), length ≤ 28.
        forecast_date: The date being predicted.
        stock_code_encoded: Integer-encoded stock code from the training encoder.
        unit_price: Unit price for the product.

    Returns:
        Single-row DataFrame with all columns in
        :data:`~app.services.model_loader.FEATURE_COLS`.
    """
    w = list(window)
    n = len(w)

    def lag(k: int) -> float:
        return float(w[-k]) if n >= k else 0.0

    def rolling_mean(size: int) -> float:
        chunk = w[-size:] if n >= size else w
        return float(np.mean(chunk)) if chunk else 0.0

    def rolling_std(size: int) -> float:
        chunk = w[-size:] if n >= size else w
        return float(np.std(chunk, ddof=0)) if len(chunk) > 1 else 0.0

    return pd.DataFrame([{
        "stock_code_encoded": stock_code_encoded,
        "unit_price": unit_price,
        "lag_1": lag(1),
        "lag_7": lag(7),
        "lag_14": lag(14),
        "lag_28": lag(28),
        "rolling_mean_7": rolling_mean(7),
        "rolling_std_7": rolling_std(7),
        "rolling_mean_14": rolling_mean(14),
        "rolling_std_14": rolling_std(14),
        "rolling_mean_28": rolling_mean(28),
        "rolling_std_28": rolling_std(28),
        "day_of_week": forecast_date.weekday(),
        "month": forecast_date.month,
        "quarter": (forecast_date.month - 1) // 3 + 1,
        "is_weekend": int(forecast_date.weekday() >= 5),
        "is_month_end": _is_month_end(forecast_date),
        "day_of_year": forecast_date.timetuple().tm_yday,
    }])


def generate_forecast(
    stock_code: str,
    horizon: int = 7,
) -> list[dict]:
    """Produce a ``horizon``-day demand forecast for ``stock_code``.

    Each step uses the *previous step's prediction* as a lag feature,
    so the forecast is truly multi-step with no future-data leakage.

    Args:
        stock_code: Product identifier (StockCode).
        horizon: Number of days to forecast (1–30).

    Returns:
        List of dicts with keys ``forecast_date``, ``predicted_quantity``,
        ``predicted_revenue`` (may be None).

    Raises:
        KeyError: If ``stock_code`` is not found in the data store.
    """
    model = model_loader.get_model()

    history_df = data_store.get_product_sales_history(stock_code, last_n=28)
    if history_df.empty:
        raise KeyError(f"No sales history found for stock_code={stock_code!r}")

    # Static per-product features — resolved once, reused at every forecast step
    sc_encoded: int = model_loader.encode_stock_code(stock_code)
    unit_price: float = data_store.get_unit_price(stock_code)

    product_info = data_store.get_product_info(stock_code)
    unit_price_display: float | None = product_info.get("unit_price") if product_info else None

    # Initialise rolling window from actual history
    qty_history = history_df["total_quantity"].tolist()
    window: deque = deque(qty_history, maxlen=28)

    last_date: date = history_df["sale_date"].max().date()

    results = []
    for i in range(horizon):
        forecast_date = last_date + timedelta(days=i + 1)
        features = _build_feature_row(window, forecast_date, sc_encoded, unit_price)

        pred_qty = float(np.clip(model.predict(features)[0], 0.0, None))
        pred_qty = round(pred_qty, 2)

        pred_revenue: float | None = None
        if unit_price_display is not None:
            pred_revenue = round(pred_qty * unit_price_display, 4)

        results.append({
            "forecast_date": forecast_date,
            "predicted_quantity": pred_qty,
            "predicted_revenue": pred_revenue,
        })

        # Feed prediction back into the window for the next step
        window.append(pred_qty)

    logger.debug(
        "Forecast for %s: %d days from %s",
        stock_code, horizon, last_date + timedelta(days=1),
    )
    return results
