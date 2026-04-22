"""
ETL Feature Engineering step.

Builds the ML-ready feature dataset from the daily aggregated sales table.

Features produced
-----------------
Time features (from ``sale_date``):
  day_of_week, month, quarter, is_weekend, is_month_end, day_of_year

Lag features (target shifted per product):
  lag_1, lag_7, lag_14, lag_28

Rolling features (computed on the *previous* day's history per product):
  rolling_mean_7,  rolling_std_7
  rolling_mean_14, rolling_std_14
  rolling_mean_28, rolling_std_28

Target variable: ``total_quantity`` (daily units sold per product).

Design note
-----------
All lag and rolling computations use ``shift(1)`` before the window so that
the feature at time t uses only information available at time t-1.
This prevents target leakage.
"""

import logging

import pandas as pd

from etl.config import LAG_DAYS, MIN_PRODUCT_OBSERVATIONS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar features from the ``sale_date`` column.

    Args:
        df: DataFrame with a ``sale_date`` column (``date`` or ``datetime``).

    Returns:
        Copy of ``df`` with additional integer time-feature columns appended.
    """
    df = df.copy()
    dates = pd.to_datetime(df["sale_date"])
    df["day_of_week"] = dates.dt.dayofweek          # 0 = Monday, 6 = Sunday
    df["month"] = dates.dt.month                    # 1–12
    df["quarter"] = dates.dt.quarter                # 1–4
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    df["is_month_end"] = dates.dt.is_month_end.astype(int)
    df["day_of_year"] = dates.dt.dayofyear          # 1–366
    return df


def add_lag_features(
    df: pd.DataFrame, lags: list[int] | None = None
) -> pd.DataFrame:
    """Add lag features for ``total_quantity`` within each product group.

    Lags are computed inside each ``stock_code`` partition after sorting
    chronologically, guaranteeing no leakage across products and no leakage
    from future rows.

    Args:
        df: Daily aggregated DataFrame. Must be sorted by
            (stock_code, sale_date) or will be sorted internally.
        lags: Lag sizes in days. Defaults to :data:`etl.config.LAG_DAYS`.

    Returns:
        DataFrame with ``lag_{n}`` columns appended.
        Rows where the lag is undefined (start of product history) will
        contain ``NaN`` — these are removed later in
        :func:`build_feature_dataset`.
    """
    if lags is None:
        lags = LAG_DAYS

    df = (
        df.copy()
        .sort_values(["stock_code", "sale_date"])
        .reset_index(drop=True)
    )

    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby("stock_code")["total_quantity"].shift(lag)
        )

    logger.debug("Lag features added: %s", [f"lag_{l}" for l in lags])
    return df


def add_rolling_features(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Add rolling mean and standard deviation features per product.

    The rolling window is applied to ``shift(1)`` of ``total_quantity``
    so the feature at time t summarises history up to t-1 only.

    ``min_periods=1`` avoids dropping rows at the beginning of a product's
    history; the std will be 0 for a single observation, which is correct.

    Args:
        df: DataFrame already sorted by (stock_code, sale_date).
        windows: Rolling window sizes in days. Defaults to
                 :data:`etl.config.ROLLING_WINDOWS`.

    Returns:
        DataFrame with ``rolling_mean_{w}`` and ``rolling_std_{w}`` columns
        appended for each window size.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    df = (
        df.copy()
        .sort_values(["stock_code", "sale_date"])
        .reset_index(drop=True)
    )

    for w in windows:
        shifted = df.groupby("stock_code")["total_quantity"].shift(1)
        df[f"rolling_mean_{w}"] = (
            shifted.groupby(df["stock_code"])
            .transform(lambda s, _w=w: s.rolling(_w, min_periods=1).mean())
        )
        df[f"rolling_std_{w}"] = (
            shifted.groupby(df["stock_code"])
            .transform(
                lambda s, _w=w: s.rolling(_w, min_periods=1).std().fillna(0.0)
            )
        )

    logger.debug("Rolling features added: windows=%s", windows)
    return df


def build_feature_dataset(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all feature engineering into a single ML-ready dataset.

    Pipeline:
    1. Add time features (day_of_week, month, quarter, …)
    2. Add lag features (lag_1, lag_7, lag_14, lag_28)
    3. Add rolling features (rolling_mean/std for 7/14/28-day windows)
    4. Drop products with fewer than ``MIN_PRODUCT_OBSERVATIONS`` rows
       (insufficient history for reliable features)
    5. Drop rows where any lag column is NaN
       (unavoidable at the start of each product's history)

    Args:
        daily_df: Output of :func:`etl.transform.aggregate_daily`.

    Returns:
        ML-ready DataFrame sorted by (stock_code, sale_date).
        The target variable is ``total_quantity``.
        Feature columns include: all ``lag_*``, ``rolling_*``,
        ``day_of_week``, ``month``, ``quarter``, ``is_weekend``,
        ``is_month_end``, ``day_of_year``.
    """
    df = add_time_features(daily_df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Drop products with insufficient history
    obs_counts = df.groupby("stock_code")["sale_date"].transform("count")
    before_filter = len(df)
    df = df[obs_counts >= MIN_PRODUCT_OBSERVATIONS].copy()
    logger.info(
        "Products filtered to >= %d observations: %d → %d rows",
        MIN_PRODUCT_OBSERVATIONS,
        before_filter,
        len(df),
    )

    # Drop rows where the longest lag is NaN (insufficient look-back at product start)
    lag_cols = [f"lag_{l}" for l in LAG_DAYS]
    before_dropna = len(df)
    df = df.dropna(subset=lag_cols)
    logger.info(
        "Dropped %d rows with NaN lag features", before_dropna - len(df)
    )

    df = (
        df.sort_values(["stock_code", "sale_date"])
        .reset_index(drop=True)
    )
    logger.info(
        "Feature dataset ready: %d rows × %d columns, %d products",
        len(df),
        len(df.columns),
        df["stock_code"].nunique(),
    )
    return df
