"""Tests for ml.splitter — time-based train/test split."""

import pandas as pd
import pytest

from ml.splitter import time_split


def _make_df(n_dates: int = 100, n_products: int = 3) -> pd.DataFrame:
    """Build a minimal feature-like DataFrame for testing."""
    import numpy as np

    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    rows = []
    for stock in [f"P{i}" for i in range(n_products)]:
        for d in dates:
            rows.append({"stock_code": stock, "sale_date": d, "total_quantity": np.random.randint(1, 50)})
    return pd.DataFrame(rows)


class TestTimeSplit:
    def test_default_ratio(self):
        df = _make_df()
        result = time_split(df)
        assert len(result.train) > 0
        assert len(result.test) > 0

    def test_no_date_overlap(self):
        df = _make_df()
        result = time_split(df)
        train_dates = set(result.train["sale_date"])
        test_dates = set(result.test["sale_date"])
        assert train_dates.isdisjoint(test_dates), "Train and test share dates — leakage!"

    def test_train_comes_before_test(self):
        df = _make_df()
        result = time_split(df)
        assert result.train["sale_date"].max() < result.test["sale_date"].min()

    def test_custom_ratio(self):
        df = _make_df(n_dates=200)
        result = time_split(df, train_ratio=0.7)
        n_unique_dates = df["sale_date"].nunique()
        expected_train_dates = int(n_unique_dates * 0.7)
        # Allow ±1 date for rounding
        actual_train_dates = result.train["sale_date"].nunique()
        assert abs(actual_train_dates - expected_train_dates) <= 1

    def test_row_count_adds_up(self):
        df = _make_df()
        result = time_split(df)
        assert len(result.train) + len(result.test) == len(df)

    def test_invalid_ratio_raises(self):
        df = _make_df()
        with pytest.raises(ValueError, match="train_ratio"):
            time_split(df, train_ratio=1.5)

    def test_too_few_dates_raises(self):
        df = pd.DataFrame(
            [{"stock_code": "A", "sale_date": pd.Timestamp("2020-01-01"), "total_quantity": 5}]
        )
        with pytest.raises(ValueError, match="unique date"):
            time_split(df)

    def test_split_metadata(self):
        df = _make_df()
        result = time_split(df)
        assert result.cutoff_date == result.test["sale_date"].min()
        assert result.n_train_products > 0
        assert result.n_test_products > 0
