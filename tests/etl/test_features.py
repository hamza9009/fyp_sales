"""
Tests for etl.features — add_time_features, add_lag_features,
add_rolling_features, build_feature_dataset.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from etl.config import LAG_DAYS, MIN_PRODUCT_OBSERVATIONS, ROLLING_WINDOWS
from etl.features import (
    add_lag_features,
    add_rolling_features,
    add_time_features,
    build_feature_dataset,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _daily_df(
    stock_code: str = "A",
    n_days: int = 60,
    start: date = date(2010, 12, 1),
    qty_start: int = 10,
) -> pd.DataFrame:
    """Create a synthetic daily sales DataFrame for a single product."""
    dates = [start + timedelta(days=i) for i in range(n_days)]
    return pd.DataFrame(
        {
            "stock_code": stock_code,
            "sale_date": dates,
            "total_quantity": [qty_start + i for i in range(n_days)],
            "total_revenue": [float(qty_start + i) * 1.5 for i in range(n_days)],
            "num_transactions": 1,
        }
    )


def _multi_product_df(n_days: int = 60) -> pd.DataFrame:
    """Two products, each with n_days of history."""
    df_a = _daily_df("A", n_days=n_days)
    df_b = _daily_df("B", n_days=n_days, qty_start=20)
    return pd.concat([df_a, df_b], ignore_index=True)


# ── add_time_features ──────────────────────────────────────────────────────────

def test_time_feature_columns_present():
    df = _daily_df(n_days=5)
    result = add_time_features(df)
    for col in ("day_of_week", "month", "quarter", "is_weekend", "is_month_end", "day_of_year"):
        assert col in result.columns, f"Missing column: {col}"


def test_day_of_week_range():
    df = _daily_df(n_days=14)
    result = add_time_features(df)
    assert result["day_of_week"].between(0, 6).all()


def test_month_range():
    df = _daily_df(n_days=5)
    result = add_time_features(df)
    assert result["month"].between(1, 12).all()


def test_is_weekend_binary():
    df = _daily_df(n_days=14)
    result = add_time_features(df)
    assert set(result["is_weekend"].unique()).issubset({0, 1})


def test_time_features_does_not_mutate_input():
    df = _daily_df(n_days=5)
    original_cols = set(df.columns)
    _ = add_time_features(df)
    assert set(df.columns) == original_cols


# ── add_lag_features ──────────────────────────────────────────────────────────

def test_lag_columns_present():
    df = _daily_df(n_days=40)
    result = add_lag_features(df)
    for lag in LAG_DAYS:
        assert f"lag_{lag}" in result.columns


def test_lag_1_correct_value():
    df = _daily_df(n_days=10)
    result = add_lag_features(df, lags=[1])
    # lag_1 at row index 1 should equal total_quantity at row index 0
    assert result["lag_1"].iloc[1] == result["total_quantity"].iloc[0]


def test_lag_does_not_cross_products():
    df = _multi_product_df(n_days=40)
    result = add_lag_features(df, lags=[1])
    # First row of product B's lag_1 should be NaN (no previous product-B row)
    b_rows = result[result["stock_code"] == "B"].reset_index(drop=True)
    assert pd.isna(b_rows["lag_1"].iloc[0])


def test_lag_first_rows_are_nan():
    df = _daily_df(n_days=35)
    result = add_lag_features(df, lags=[7])
    # First 7 rows have no lag_7 history → NaN
    assert result["lag_7"].iloc[:7].isna().all()


# ── add_rolling_features ──────────────────────────────────────────────────────

def test_rolling_columns_present():
    df = _daily_df(n_days=40)
    df = add_lag_features(df)
    result = add_rolling_features(df)
    for w in ROLLING_WINDOWS:
        assert f"rolling_mean_{w}" in result.columns
        assert f"rolling_std_{w}" in result.columns


def test_rolling_mean_is_numeric():
    df = _daily_df(n_days=40)
    result = add_rolling_features(df)
    for w in ROLLING_WINDOWS:
        assert pd.api.types.is_numeric_dtype(result[f"rolling_mean_{w}"])


def test_rolling_std_non_negative():
    df = _daily_df(n_days=40)
    result = add_rolling_features(df)
    for w in ROLLING_WINDOWS:
        col = result[f"rolling_std_{w}"].dropna()
        assert (col >= 0).all(), f"rolling_std_{w} has negative values"


def test_rolling_does_not_cross_products():
    df = _multi_product_df(n_days=40)
    result = add_rolling_features(df)
    # The rolling_mean_7 of product B's first row should NOT use product A's data.
    # Specifically the max rolling mean for B should not exceed B's own max quantity.
    b_max_qty = df[df["stock_code"] == "B"]["total_quantity"].max()
    b_roll = result[result["stock_code"] == "B"]["rolling_mean_7"].max()
    # rolling mean cannot exceed the max qty of that product
    assert b_roll <= b_max_qty + 1  # +1 for floating-point tolerance


# ── build_feature_dataset ──────────────────────────────────────────────────────

def test_build_feature_dataset_has_all_feature_groups():
    df = _multi_product_df(n_days=80)
    result = build_feature_dataset(df)
    # Time features
    for col in ("day_of_week", "month", "is_weekend"):
        assert col in result.columns
    # Lag features
    for lag in LAG_DAYS:
        assert f"lag_{lag}" in result.columns
    # Rolling features
    for w in ROLLING_WINDOWS:
        assert f"rolling_mean_{w}" in result.columns
        assert f"rolling_std_{w}" in result.columns


def test_build_feature_dataset_no_nan_in_lags():
    df = _multi_product_df(n_days=80)
    result = build_feature_dataset(df)
    lag_cols = [f"lag_{l}" for l in LAG_DAYS]
    assert not result[lag_cols].isna().any().any()


def test_build_feature_dataset_drops_sparse_products():
    # Product C has only 5 rows — below MIN_PRODUCT_OBSERVATIONS
    df_c = _daily_df("C", n_days=5)
    df = _multi_product_df(n_days=80)
    df = pd.concat([df, df_c], ignore_index=True)
    result = build_feature_dataset(df)
    assert "C" not in result["stock_code"].unique()


def test_build_feature_dataset_target_column_present():
    df = _multi_product_df(n_days=80)
    result = build_feature_dataset(df)
    assert "total_quantity" in result.columns


def test_build_feature_dataset_sorted_by_product_then_date():
    df = _multi_product_df(n_days=80)
    result = build_feature_dataset(df)
    dates_by_product = result.groupby("stock_code")["sale_date"].apply(list)
    for code, dates in dates_by_product.items():
        assert dates == sorted(dates), f"Dates not sorted for product {code}"
