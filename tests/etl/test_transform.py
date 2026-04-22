"""
Tests for etl.transform — add_revenue, aggregate_daily, extract_products.
"""

from datetime import date

import pandas as pd
import pytest

from etl.transform import add_revenue, aggregate_daily, extract_products


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sales_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _base_row(**overrides) -> dict:
    base = {
        "invoice_no": "536365",
        "stock_code": "85123A",
        "description": "WHITE HEART",
        "quantity": 6.0,
        "invoice_date": pd.Timestamp("2010-12-01 08:26:00"),
        "unit_price": 2.55,
        "revenue": 15.30,
        "customer_id": "17850",
        "country": "United Kingdom",
    }
    base.update(overrides)
    return base


# ── add_revenue ────────────────────────────────────────────────────────────────

def test_revenue_column_added():
    df = _sales_df({"quantity": 6.0, "unit_price": 2.55})
    result = add_revenue(df)
    assert "revenue" in result.columns


def test_revenue_value_correct():
    df = _sales_df({"quantity": 6.0, "unit_price": 2.55})
    result = add_revenue(df)
    assert abs(result["revenue"].iloc[0] - 15.30) < 1e-9


def test_revenue_does_not_mutate_input():
    df = _sales_df({"quantity": 6.0, "unit_price": 2.55})
    _ = add_revenue(df)
    assert "revenue" not in df.columns


def test_revenue_multiple_rows():
    df = _sales_df(
        {"quantity": 3.0, "unit_price": 1.0},
        {"quantity": 4.0, "unit_price": 2.0},
    )
    result = add_revenue(df)
    assert result["revenue"].tolist() == [3.0, 8.0]


# ── aggregate_daily ────────────────────────────────────────────────────────────

def _agg_input(*rows: dict) -> pd.DataFrame:
    """Build a minimal DataFrame for aggregate_daily."""
    return pd.DataFrame(list(rows))


def test_aggregate_daily_one_row_per_product_date():
    df = _agg_input(
        {"stock_code": "A", "invoice_date": pd.Timestamp("2010-12-01"),
         "quantity": 3.0, "revenue": 6.0, "invoice_no": "INV1"},
        {"stock_code": "A", "invoice_date": pd.Timestamp("2010-12-01"),
         "quantity": 2.0, "revenue": 4.0, "invoice_no": "INV2"},
    )
    result = aggregate_daily(df)
    assert len(result) == 1
    assert result.iloc[0]["total_quantity"] == 5
    assert abs(result.iloc[0]["total_revenue"] - 10.0) < 1e-6


def test_aggregate_daily_counts_unique_invoices():
    df = _agg_input(
        {"stock_code": "A", "invoice_date": pd.Timestamp("2010-12-01"),
         "quantity": 1.0, "revenue": 1.0, "invoice_no": "INV1"},
        {"stock_code": "A", "invoice_date": pd.Timestamp("2010-12-01"),
         "quantity": 1.0, "revenue": 1.0, "invoice_no": "INV1"},  # same invoice
        {"stock_code": "A", "invoice_date": pd.Timestamp("2010-12-01"),
         "quantity": 1.0, "revenue": 1.0, "invoice_no": "INV2"},
    )
    result = aggregate_daily(df)
    # Two unique invoices on that day
    assert result.iloc[0]["num_transactions"] == 2


def test_aggregate_daily_sale_date_is_date_type():
    df = _agg_input(
        {"stock_code": "A", "invoice_date": pd.Timestamp("2010-12-01 09:00:00"),
         "quantity": 2.0, "revenue": 4.0, "invoice_no": "INV1"},
    )
    result = aggregate_daily(df)
    assert isinstance(result.iloc[0]["sale_date"], date)


def test_aggregate_daily_separates_by_product():
    df = _agg_input(
        {"stock_code": "A", "invoice_date": pd.Timestamp("2010-12-01"),
         "quantity": 5.0, "revenue": 10.0, "invoice_no": "INV1"},
        {"stock_code": "B", "invoice_date": pd.Timestamp("2010-12-01"),
         "quantity": 3.0, "revenue": 6.0, "invoice_no": "INV2"},
    )
    result = aggregate_daily(df)
    assert len(result) == 2
    codes = set(result["stock_code"])
    assert codes == {"A", "B"}


def test_aggregate_daily_separates_by_date():
    df = _agg_input(
        {"stock_code": "A", "invoice_date": pd.Timestamp("2010-12-01"),
         "quantity": 2.0, "revenue": 4.0, "invoice_no": "INV1"},
        {"stock_code": "A", "invoice_date": pd.Timestamp("2010-12-02"),
         "quantity": 3.0, "revenue": 6.0, "invoice_no": "INV2"},
    )
    result = aggregate_daily(df)
    assert len(result) == 2


# ── extract_products ───────────────────────────────────────────────────────────

def _product_input(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_extract_products_one_row_per_stock_code():
    df = _product_input(
        {"stock_code": "A", "description": "Item A", "unit_price": 1.0, "country": "UK"},
        {"stock_code": "A", "description": "Item A", "unit_price": 1.5, "country": "UK"},
        {"stock_code": "B", "description": "Item B", "unit_price": 2.0, "country": "DE"},
    )
    result = extract_products(df)
    assert len(result) == 2


def test_extract_products_unit_price_is_median():
    df = _product_input(
        {"stock_code": "A", "description": "X", "unit_price": 1.0, "country": "UK"},
        {"stock_code": "A", "description": "X", "unit_price": 3.0, "country": "UK"},
        {"stock_code": "A", "description": "X", "unit_price": 5.0, "country": "UK"},
    )
    result = extract_products(df)
    # Median of [1, 3, 5] = 3
    assert abs(result.iloc[0]["unit_price"] - 3.0) < 1e-6


def test_extract_products_description_is_mode():
    df = _product_input(
        {"stock_code": "A", "description": "Correct", "unit_price": 1.0, "country": "UK"},
        {"stock_code": "A", "description": "Correct", "unit_price": 1.0, "country": "UK"},
        {"stock_code": "A", "description": "Wrong", "unit_price": 1.0, "country": "UK"},
    )
    result = extract_products(df)
    assert result.iloc[0]["description"] == "Correct"


def test_extract_products_required_columns_present():
    df = _product_input(
        {"stock_code": "A", "description": "X", "unit_price": 1.0, "country": "UK"},
    )
    result = extract_products(df)
    assert {"stock_code", "description", "unit_price", "country"}.issubset(set(result.columns))
