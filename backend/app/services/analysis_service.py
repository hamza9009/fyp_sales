"""
Analysis service — univariate & bivariate data analysis.

Derives all statistics from the in-memory Parquet DataFrames loaded by
data_store at startup.  All functions are synchronous and cheap because
the data is already in RAM.
"""

import logging

import numpy as np
import pandas as pd

from app.services import data_store

logger = logging.getLogger(__name__)

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _histogram(series: pd.Series, n_bins: int = 20) -> list[dict]:
    """Build histogram bins, excluding top 1% outliers for readability."""
    s = series.dropna()
    if s.empty:
        return []
    cap = float(s.quantile(0.99))
    s = s[s <= cap]
    counts, edges = np.histogram(s, bins=n_bins)
    return [
        {"bin_start": round(float(edges[i]), 4), "bin_end": round(float(edges[i + 1]), 4), "count": int(counts[i])}
        for i in range(len(counts))
        if counts[i] > 0
    ]


def get_univariate_analysis() -> dict:
    sales = data_store.get_sales()
    products = data_store.get_products()

    # Quantity histogram — daily totals per product
    qty_hist = _histogram(sales["total_quantity"], n_bins=25)

    # Revenue histogram — daily totals per product
    rev_hist = _histogram(sales["total_revenue"], n_bins=25)

    # Price histogram — unit price per product
    price_hist = _histogram(products["unit_price"].dropna(), n_bins=20)

    # Sales by day of week — avg daily quantity
    features = data_store.get_features()
    dow_avg = (
        features.groupby("day_of_week")["total_quantity"]
        .mean()
        .reindex(range(7), fill_value=0)
    )
    sales_by_dow = [
        {"label": _DAY_NAMES[i], "value": round(float(dow_avg[i]), 2)}
        for i in range(7)
    ]

    # Sales by month — avg daily quantity
    month_avg = (
        features.groupby("month")["total_quantity"]
        .mean()
        .reindex(range(1, 13), fill_value=0)
    )
    sales_by_month = [
        {"label": _MONTH_NAMES[m - 1], "value": round(float(month_avg[m]), 2)}
        for m in range(1, 13)
    ]

    # Products by activity (# unique days each product appears)
    days_active = sales.groupby("stock_code")["sale_date"].nunique()
    buckets = pd.cut(days_active, bins=[0, 30, 90, 180, 270, 365, 500, 10000],
                     labels=["1-30", "31-90", "91-180", "181-270", "271-365", "366-500", "500+"])
    activity_counts = buckets.value_counts().sort_index()
    products_by_activity = [
        {"label": str(label), "value": float(count)}
        for label, count in activity_counts.items()
    ]

    return {
        "quantity_histogram": qty_hist,
        "revenue_histogram": rev_hist,
        "price_histogram": price_hist,
        "sales_by_day_of_week": sales_by_dow,
        "sales_by_month": sales_by_month,
        "products_by_activity": products_by_activity,
    }


def get_bivariate_analysis() -> dict:
    sales = data_store.get_sales()
    products = data_store.get_products()
    features = data_store.get_features()

    # Revenue by month — aggregate across all years
    sales_copy = sales.copy()
    sales_copy["month"] = sales_copy["sale_date"].dt.month
    rev_by_month = (
        sales_copy.groupby("month")["total_revenue"]
        .sum()
        .reindex(range(1, 13), fill_value=0)
    )
    revenue_by_month = [
        {"label": _MONTH_NAMES[m - 1], "value": round(float(rev_by_month[m]), 2)}
        for m in range(1, 13)
    ]

    # Quantity by day of week
    dow_qty = (
        features.groupby("day_of_week")["total_quantity"]
        .mean()
        .reindex(range(7), fill_value=0)
    )
    quantity_by_dow = [
        {"label": _DAY_NAMES[i], "value": round(float(dow_qty[i]), 2)}
        for i in range(7)
    ]

    # Revenue by quarter
    sales_copy["quarter"] = sales_copy["sale_date"].dt.quarter
    rev_by_q = (
        sales_copy.groupby("quarter")["total_revenue"]
        .sum()
        .reindex([1, 2, 3, 4], fill_value=0)
    )
    revenue_by_quarter = [
        {"label": f"Q{q}", "value": round(float(rev_by_q[q]), 2)}
        for q in [1, 2, 3, 4]
    ]

    # Price vs avg quantity — join unit_price to features, bucket by price
    price_map = (
        products.dropna(subset=["unit_price"])
        .drop_duplicates("stock_code")
        .set_index("stock_code")["unit_price"]
    )
    feat_with_price = features.copy()
    feat_with_price["unit_price"] = feat_with_price["stock_code"].map(price_map)
    feat_with_price = feat_with_price.dropna(subset=["unit_price"])

    # Use quantile-based bins so each bucket has data
    try:
        feat_with_price["price_bucket"] = pd.qcut(
            feat_with_price["unit_price"], q=8, duplicates="drop"
        )
        price_qty = (
            feat_with_price.groupby("price_bucket", observed=True)["total_quantity"]
            .mean()
        )
        price_vs_qty = [
            {
                "x": round(float(interval.mid), 2),
                "y": round(float(val), 2),
                "label": f"£{interval.left:.2f}–£{interval.right:.2f}",
            }
            for interval, val in price_qty.items()
        ]
    except Exception:
        price_vs_qty = []

    # Monthly quantity trend (month-year, all data)
    sales_copy["month_year"] = sales_copy["sale_date"].dt.to_period("M")
    monthly_qty = (
        sales_copy.groupby("month_year")["total_quantity"]
        .sum()
        .sort_index()
    )
    monthly_quantity_trend = [
        {"label": str(period), "value": float(qty)}
        for period, qty in monthly_qty.items()
    ]

    return {
        "revenue_by_month": revenue_by_month,
        "quantity_by_day_of_week": quantity_by_dow,
        "revenue_by_quarter": revenue_by_quarter,
        "price_vs_quantity_buckets": price_vs_qty,
        "monthly_quantity_trend": monthly_quantity_trend,
    }
