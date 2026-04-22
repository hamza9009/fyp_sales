"""
Tests for etl.extract — load_raw_data.

All tests use temporary CSV files so the real dataset is not required.
"""

import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from etl.extract import load_raw_data


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_csv(tmp_path: Path, content: str) -> Path:
    """Write CSV content to a temp file and return its path."""
    p = tmp_path / "test_data.csv"
    p.write_text(textwrap.dedent(content).strip())
    return p


VALID_CSV = """
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,2010-12-01 08:26:00,2.55,17850,United Kingdom
536366,22633,HAND WARMER UNION JACK,6,2010-12-01 08:28:00,1.85,17850,United Kingdom
"""


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_load_csv_returns_dataframe(tmp_path):
    p = _write_csv(tmp_path, VALID_CSV)
    df = load_raw_data(p)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_columns_are_standardised(tmp_path):
    p = _write_csv(tmp_path, VALID_CSV)
    df = load_raw_data(p)
    expected = {"invoice_no", "stock_code", "description", "quantity",
                "invoice_date", "unit_price", "customer_id", "country"}
    assert expected.issubset(set(df.columns))


def test_invoice_date_is_datetime(tmp_path):
    p = _write_csv(tmp_path, VALID_CSV)
    df = load_raw_data(p)
    assert pd.api.types.is_datetime64_any_dtype(df["invoice_date"])


def test_quantity_and_price_are_numeric(tmp_path):
    p = _write_csv(tmp_path, VALID_CSV)
    df = load_raw_data(p)
    assert pd.api.types.is_numeric_dtype(df["quantity"])
    assert pd.api.types.is_numeric_dtype(df["unit_price"])


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_raw_data("/nonexistent/path/data.csv")


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "data.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_raw_data(p)


def test_missing_required_column_raises(tmp_path):
    # Omit 'UnitPrice' column — should trigger ValueError
    bad_csv = "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,CustomerID,Country\n"
    bad_csv += "536365,85123A,Item,6,2010-12-01,17850,UK\n"
    p = _write_csv(tmp_path, bad_csv)
    with pytest.raises(ValueError, match="Mandatory columns missing"):
        load_raw_data(p)


def test_unparseable_dates_are_dropped(tmp_path):
    csv = (
        "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
        "536365,85123A,Item,6,2010-12-01 08:26:00,2.55,17850,UK\n"
        "536366,22633,Item2,6,NOT_A_DATE,1.85,17850,UK\n"
    )
    p = _write_csv(tmp_path, csv)
    df = load_raw_data(p)
    # Only the valid row should survive
    assert len(df) == 1
