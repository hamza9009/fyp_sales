"""
Tests for etl.clean — clean_data.

Synthetic DataFrames are built inline so no real dataset is needed.
"""

import pandas as pd
import pytest

from etl.clean import clean_data


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_df(**overrides) -> pd.DataFrame:
    """Return a single-row valid DataFrame, optionally overriding columns."""
    base = {
        "invoice_no": "536365",
        "stock_code": "85123A",
        "description": "WHITE HEART",
        "quantity": 6.0,
        "invoice_date": pd.Timestamp("2010-12-01"),
        "unit_price": 2.55,
        "customer_id": "17850",
        "country": "United Kingdom",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _make_multi(*rows: dict) -> pd.DataFrame:
    """Build a DataFrame from a list of row dicts."""
    return pd.DataFrame(list(rows))


# ── Tests: cancellation removal ────────────────────────────────────────────────

def test_cancellation_invoice_removed():
    df = _make_df(invoice_no="C536365")
    result = clean_data(df)
    assert len(result) == 0


def test_normal_invoice_kept():
    df = _make_df(invoice_no="536365")
    result = clean_data(df)
    assert len(result) == 1


def test_mixed_cancellations():
    df = _make_multi(
        {"invoice_no": "536365", "stock_code": "A", "description": "Item A",
         "quantity": 5.0, "invoice_date": pd.Timestamp("2010-12-01"),
         "unit_price": 1.0, "customer_id": "111", "country": "UK"},
        {"invoice_no": "C536366", "stock_code": "B", "description": "Item B",
         "quantity": 5.0, "invoice_date": pd.Timestamp("2010-12-01"),
         "unit_price": 1.0, "customer_id": "222", "country": "UK"},
    )
    result = clean_data(df)
    assert len(result) == 1
    assert result.iloc[0]["invoice_no"] == "536365"


# ── Tests: null / empty stock_code or description ──────────────────────────────

@pytest.mark.parametrize("bad_value", ["", "nan", "None", "NaN"])
def test_empty_stock_code_removed(bad_value):
    df = _make_df(stock_code=bad_value)
    result = clean_data(df)
    assert len(result) == 0


@pytest.mark.parametrize("bad_value", ["", "nan", "None", "NaN"])
def test_empty_description_removed(bad_value):
    df = _make_df(description=bad_value)
    result = clean_data(df)
    assert len(result) == 1
    assert result.iloc[0]["description"] == "Unknown description"


def test_description_imputed_from_stock_code_mode():
    df = _make_multi(
        {
            "invoice_no": "536365",
            "stock_code": "85123A",
            "description": "WHITE HEART",
            "quantity": 6.0,
            "invoice_date": pd.Timestamp("2010-12-01"),
            "unit_price": 2.55,
            "customer_id": "17850",
            "country": "United Kingdom",
        },
        {
            "invoice_no": "536366",
            "stock_code": "85123A",
            "description": "",
            "quantity": 7.0,
            "invoice_date": pd.Timestamp("2010-12-02"),
            "unit_price": 2.65,
            "customer_id": "17850",
            "country": "United Kingdom",
        },
    )
    result = clean_data(df)
    assert len(result) == 2
    assert (result["description"] == "WHITE HEART").all()


def test_stock_code_imputed_from_description_mode():
    df = _make_multi(
        {
            "invoice_no": "536365",
            "stock_code": "85123A",
            "description": "WHITE HEART",
            "quantity": 6.0,
            "invoice_date": pd.Timestamp("2010-12-01"),
            "unit_price": 2.55,
            "customer_id": "17850",
            "country": "United Kingdom",
        },
        {
            "invoice_no": "536366",
            "stock_code": "",
            "description": "WHITE HEART",
            "quantity": 7.0,
            "invoice_date": pd.Timestamp("2010-12-02"),
            "unit_price": 2.65,
            "customer_id": "17850",
            "country": "United Kingdom",
        },
    )
    result = clean_data(df)
    assert len(result) == 2
    assert (result["stock_code"] == "85123A").all()


# ── Tests: quantity filtering ──────────────────────────────────────────────────

def test_negative_quantity_removed():
    df = _make_df(quantity=-3.0)
    result = clean_data(df)
    assert len(result) == 0


def test_zero_quantity_removed():
    df = _make_df(quantity=0.0)
    result = clean_data(df)
    assert len(result) == 0


def test_positive_quantity_kept():
    df = _make_df(quantity=1.0)
    result = clean_data(df)
    assert len(result) == 1


def test_missing_quantity_imputed_from_stock_mean():
    df = _make_multi(
        {
            "invoice_no": "536365",
            "stock_code": "85123A",
            "description": "WHITE HEART",
            "quantity": 10.0,
            "invoice_date": pd.Timestamp("2010-12-01"),
            "unit_price": 2.55,
            "customer_id": "17850",
            "country": "United Kingdom",
        },
        {
            "invoice_no": "536366",
            "stock_code": "85123A",
            "description": "WHITE HEART",
            "quantity": float("nan"),
            "invoice_date": pd.Timestamp("2010-12-02"),
            "unit_price": 2.55,
            "customer_id": "17850",
            "country": "United Kingdom",
        },
    )
    result = clean_data(df)
    assert len(result) == 2
    assert result["quantity"].tolist() == [10.0, 10.0]


# ── Tests: unit_price filtering ────────────────────────────────────────────────

def test_zero_price_removed():
    df = _make_df(unit_price=0.0)
    result = clean_data(df)
    assert len(result) == 0


def test_negative_price_removed():
    df = _make_df(unit_price=-1.0)
    result = clean_data(df)
    assert len(result) == 0


def test_valid_price_kept():
    df = _make_df(unit_price=0.01)
    result = clean_data(df)
    assert len(result) == 1


def test_nan_price_removed():
    df = _make_df(unit_price=float("nan"))
    result = clean_data(df)
    assert len(result) == 0


def test_missing_price_imputed_from_stock_mean():
    df = _make_multi(
        {
            "invoice_no": "536365",
            "stock_code": "85123A",
            "description": "WHITE HEART",
            "quantity": 6.0,
            "invoice_date": pd.Timestamp("2010-12-01"),
            "unit_price": 2.55,
            "customer_id": "17850",
            "country": "United Kingdom",
        },
        {
            "invoice_no": "536366",
            "stock_code": "85123A",
            "description": "WHITE HEART",
            "quantity": 4.0,
            "invoice_date": pd.Timestamp("2010-12-02"),
            "unit_price": float("nan"),
            "customer_id": "17850",
            "country": "United Kingdom",
        },
    )
    result = clean_data(df)
    assert len(result) == 2
    assert result["unit_price"].tolist() == [2.55, 2.55]


def test_missing_invoice_number_replaced_with_placeholder():
    df = _make_multi(
        {
            "invoice_no": "",
            "stock_code": "85123A",
            "description": "WHITE HEART",
            "quantity": 6.0,
            "invoice_date": pd.Timestamp("2010-12-01"),
            "unit_price": 2.55,
            "customer_id": "17850",
            "country": "United Kingdom",
        },
        {
            "invoice_no": "536366",
            "stock_code": "85123A",
            "description": "WHITE HEART",
            "quantity": 4.0,
            "invoice_date": pd.Timestamp("2010-12-02"),
            "unit_price": 2.55,
            "customer_id": "17850",
            "country": "United Kingdom",
        },
    )
    result = clean_data(df)
    assert len(result) == 2
    assert result.loc[0, "invoice_no"].startswith("MISSING_INV_")


# ── Tests: output integrity ────────────────────────────────────────────────────

def test_index_is_reset_after_cleaning():
    df = _make_multi(
        {"invoice_no": "C1", "stock_code": "A", "description": "X",
         "quantity": 5.0, "invoice_date": pd.Timestamp("2010-12-01"),
         "unit_price": 1.0, "customer_id": "1", "country": "UK"},
        {"invoice_no": "536365", "stock_code": "B", "description": "Y",
         "quantity": 2.0, "invoice_date": pd.Timestamp("2010-12-01"),
         "unit_price": 2.0, "customer_id": "2", "country": "UK"},
    )
    result = clean_data(df)
    assert result.index.tolist() == [0]


def test_all_valid_rows_kept():
    rows = [
        {"invoice_no": f"53636{i}", "stock_code": f"CODE{i}", "description": "Item",
         "quantity": float(i + 1), "invoice_date": pd.Timestamp("2010-12-01"),
         "unit_price": 1.5, "customer_id": "123", "country": "UK"}
        for i in range(5)
    ]
    df = _make_multi(*rows)
    result = clean_data(df)
    assert len(result) == 5
