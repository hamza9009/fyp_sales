"""
Dashboard service — Phase 4.

Aggregates sales data and model metrics into a single summary payload
for the dashboard overview endpoint.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from app.services import data_store

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_METRICS_JSON = _PROJECT_ROOT / "ml" / "artifacts" / "metrics_report.json"


def _load_metrics_report() -> dict:
    """Load the Phase 3 metrics JSON report.

    Returns an empty dict if the file is missing (graceful degradation).
    """
    if not _METRICS_JSON.exists():
        logger.warning("Metrics report not found at %s", _METRICS_JSON)
        return {}
    with _METRICS_JSON.open() as fh:
        return json.load(fh)


def get_dashboard_summary() -> dict:
    """Build the full dashboard summary payload.

    Returns:
        Dict matching :class:`~app.schemas.dashboard.DashboardSummaryResponse`.
    """
    sales = data_store.get_sales()
    products = data_store.get_products()

    # ── Dataset-level overview ───────────────────────────────────────────────
    total_products = int(sales["stock_code"].nunique())
    total_revenue = float(sales["total_revenue"].sum())
    total_quantity = int(sales["total_quantity"].sum())
    date_range_start = sales["sale_date"].min().date()
    date_range_end = sales["sale_date"].max().date()
    total_days = (date_range_end - date_range_start).days + 1

    # ── Last 30 days trend ───────────────────────────────────────────────────
    cutoff = sales["sale_date"].max() - pd.Timedelta(days=29)
    recent = sales[sales["sale_date"] >= cutoff]
    daily_trend = (
        recent
        .groupby("sale_date", as_index=False)
        .agg(total_quantity=("total_quantity", "sum"), total_revenue=("total_revenue", "sum"))
        .sort_values("sale_date")
    )
    last_30_days = [
        {
            "sale_date": row["sale_date"].date(),
            "total_quantity": int(row["total_quantity"]),
            "total_revenue": round(float(row["total_revenue"]), 4),
        }
        for _, row in daily_trend.iterrows()
    ]

    # ── Top 10 products by total revenue ────────────────────────────────────
    product_agg = (
        sales
        .groupby("stock_code", as_index=False)
        .agg(
            total_revenue=("total_revenue", "sum"),
            total_quantity=("total_quantity", "sum"),
            num_days_active=("sale_date", "nunique"),
        )
        .sort_values("total_revenue", ascending=False)
        .head(10)
    )

    # Join with description from products catalogue
    desc_map: dict[str, str | None] = {}
    if "description" in products.columns:
        desc_map = (
            products.dropna(subset=["description"])
            .drop_duplicates(subset=["stock_code"])
            .set_index("stock_code")["description"]
            .to_dict()
        )

    top_products = [
        {
            "stock_code": row["stock_code"],
            "description": desc_map.get(row["stock_code"]),
            "total_revenue": round(float(row["total_revenue"]), 4),
            "total_quantity": int(row["total_quantity"]),
            "num_days_active": int(row["num_days_active"]),
        }
        for _, row in product_agg.iterrows()
    ]

    # ── Model performance snapshot ───────────────────────────────────────────
    report = _load_metrics_report()
    best_model_name = report.get("best_model", "Unknown")
    split_info = report.get("split", {})
    models_info = report.get("models", {})
    best_metrics = models_info.get(best_model_name, {})

    best_model = {
        "model_name": best_model_name,
        "mae": best_metrics.get("mae", 0.0),
        "rmse": best_metrics.get("rmse", 0.0),
        "train_rows": split_info.get("train_rows", 0),
        "test_rows": split_info.get("test_rows", 0),
        "cutoff_date": split_info.get("cutoff_date", ""),
    }

    return {
        "total_products": total_products,
        "total_revenue": round(total_revenue, 4),
        "total_quantity": total_quantity,
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "total_days": total_days,
        "last_30_days_trend": last_30_days,
        "top_products": top_products,
        "best_model": best_model,
    }
