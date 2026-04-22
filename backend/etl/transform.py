"""
ETL Transform step.

Responsibilities:
- Add ``revenue`` column (quantity × unit_price)
- Aggregate transaction rows to one row per (stock_code, date) — the schema
  expected by the ``sales_daily`` database table
- Extract the unique product catalogue (one row per stock_code)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``revenue = quantity × unit_price`` and append it as a new column.

    Args:
        df: Cleaned DataFrame containing numeric ``quantity`` and ``unit_price``
            columns.

    Returns:
        Copy of ``df`` with an additional ``revenue`` (float64) column.
    """
    df = df.copy()
    df["revenue"] = df["quantity"] * df["unit_price"]
    logger.debug("Revenue column added")
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transaction rows to one row per (stock_code, date).

    The result matches the ``sales_daily`` table schema:

    ============= ====================================================
    Column        Description
    ============= ====================================================
    stock_code    Product identifier
    sale_date     Calendar date (Python ``date`` object)
    total_quantity Sum of all units sold on that day for that product
    total_revenue  Sum of revenue on that day for that product
    num_transactions Number of unique invoices touching that product
    ============= ====================================================

    Args:
        df: Cleaned, revenue-enriched DataFrame. Requires ``stock_code``,
            ``invoice_date`` (datetime), ``quantity``, ``revenue``,
            and ``invoice_no`` columns.

    Returns:
        Aggregated DataFrame sorted by (stock_code, sale_date).
    """
    df = df.copy()
    # Floor invoice_date to midnight so grouping is strictly date-based
    df["sale_date"] = df["invoice_date"].dt.normalize()

    agg = (
        df.groupby(["stock_code", "sale_date"], as_index=False)
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            num_transactions=("invoice_no", "nunique"),
        )
    )

    agg["total_quantity"] = agg["total_quantity"].astype(int)
    agg["total_revenue"] = agg["total_revenue"].round(4)
    agg["num_transactions"] = agg["num_transactions"].astype(int)
    # Convert Timestamp → plain date so SQLAlchemy/Parquet handles it cleanly
    agg["sale_date"] = pd.to_datetime(agg["sale_date"]).dt.date

    agg = agg.sort_values(["stock_code", "sale_date"]).reset_index(drop=True)

    logger.info(
        "Aggregation complete: %d daily records across %d products",
        len(agg),
        agg["stock_code"].nunique(),
    )
    return agg


def extract_products(df: pd.DataFrame) -> pd.DataFrame:
    """Build the unique product catalogue from the cleaned transaction data.

    For each ``stock_code``:
    - **description**: most frequently occurring description (mode)
    - **unit_price**: median across all transactions (robust to outliers)
    - **country**: most frequently occurring country

    Args:
        df: Cleaned DataFrame with ``stock_code``, ``description``,
            ``unit_price``, and ``country`` columns.

    Returns:
        DataFrame with one row per ``stock_code`` and columns:
        ``stock_code``, ``description``, ``unit_price``, ``country``.
    """

    def _mode_first(series: pd.Series) -> str:
        modes = series.mode()
        return modes.iloc[0] if not modes.empty else ""

    products = (
        df.groupby("stock_code", as_index=False)
        .agg(
            description=("description", _mode_first),
            unit_price=("unit_price", "median"),
            country=("country", _mode_first),
        )
    )

    products["unit_price"] = products["unit_price"].round(4)
    logger.info("Extracted %d unique products", len(products))
    return products
