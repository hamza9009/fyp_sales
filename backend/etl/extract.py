"""
ETL Extract step.

Responsibilities:
- Inspect column headers from raw file and suggest a mapping
- Load raw Excel or CSV from disk
- Standardise column names to snake_case via user mapping or auto-match
- Parse ``invoice_date`` to datetime (drop unparseable rows with a warning)
- Cast ``quantity`` and ``unit_price`` to numeric
- Validate that all required columns are present
"""

import io
import logging
import re
from pathlib import Path

import pandas as pd

from etl.config import MANDATORY_COLUMNS, OPTIONAL_COLUMN_DEFAULTS, RAW_COLUMN_MAP

logger = logging.getLogger(__name__)

# ── Required field metadata + alias lists for auto-suggestion ──────────────────

REQUIRED_FIELD_META: dict[str, dict] = {
    "invoice_no": {
        "label": "InvoiceNo",
        "desc": "Unique transaction ID",
        "required": True,
        "aliases": ["invoiceno", "invoice_no", "invoice", "transid", "transactionid",
                    "billno", "orderid", "orderno", "receiptno"],
    },
    "stock_code": {
        "label": "StockCode",
        "desc": "Product / SKU code",
        "required": True,
        "aliases": ["stockcode", "stock_code", "sku", "productcode", "itemcode",
                    "prodcode", "article", "pid", "productid"],
    },
    "description": {
        "label": "Description",
        "desc": "Product name",
        "required": True,
        "aliases": ["description", "desc", "productname", "itemname", "name",
                    "product", "itemdescription", "prodname", "title"],
    },
    "quantity": {
        "label": "Quantity",
        "desc": "Units sold (negative = return)",
        "required": True,
        "aliases": ["quantity", "qty", "units", "amount", "unitssold", "count",
                    "quantitysold", "vol", "volume"],
    },
    "invoice_date": {
        "label": "InvoiceDate",
        "desc": "Date and time of transaction",
        "required": True,
        "aliases": ["invoicedate", "invoice_date", "date", "transactiondate",
                    "orderdate", "saledate", "datetime", "purchasedate", "timestamp"],
    },
    "unit_price": {
        "label": "UnitPrice",
        "desc": "Price per unit in GBP",
        "required": True,
        "aliases": ["unitprice", "unit_price", "price", "cost", "priceperunit",
                    "sellingprice", "saleprice", "rate"],
    },
    "customer_id": {
        "label": "CustomerID",
        "desc": "Unique customer identifier (optional — 'UNKNOWN' used if absent)",
        "required": False,
        "aliases": ["customerid", "customer_id", "custid", "clientid", "buyerid",
                    "userid", "memberid", "accountid"],
    },
    "country": {
        "label": "Country",
        "desc": "Customer country (optional — 'Unknown' used if absent)",
        "required": False,
        "aliases": ["country", "nation", "location", "region", "territory",
                    "countryname", "market"],
    },
}


def _normalize(s: str) -> str:
    """Strip non-alphanumeric chars and lowercase for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def inspect_columns(content: bytes, suffix: str) -> dict:
    """Read column headers from file bytes and suggest a mapping.

    Only the first row (headers) is read — no data rows are loaded.

    Args:
        content: Raw file bytes.
        suffix:  File extension including dot (e.g. ``".csv"``).

    Returns:
        Dict with three keys:

        - ``columns`` — list of actual column names found in the file
        - ``suggested_mapping`` — ``{internal_snake_case → user_column}`` for
          every required column that was auto-matched
        - ``required_fields`` — list of ``{internal_name, label, desc}`` dicts
          describing what each required column represents
    """
    suffix = suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(content), nrows=0, dtype=str)
        elif suffix == ".csv":
            df = pd.read_csv(
                io.BytesIO(content), nrows=0, dtype=str,
                encoding="utf-8", low_memory=False,
            )
        else:
            raise ValueError(f"Unsupported file format: {suffix!r}")
    except UnicodeDecodeError:
        df = pd.read_csv(
            io.BytesIO(content), nrows=0, dtype=str,
            encoding="latin-1", low_memory=False,
        )

    actual_cols = list(df.columns)
    norm_to_orig = {_normalize(col): col for col in actual_cols}

    suggested: dict[str, str] = {}
    for internal_name, meta in REQUIRED_FIELD_META.items():
        for alias in meta["aliases"]:
            if _normalize(alias) in norm_to_orig:
                suggested[internal_name] = norm_to_orig[_normalize(alias)]
                break

    return {
        "columns": actual_cols,
        "suggested_mapping": suggested,
        "required_fields": [
            {
                "internal_name": k,
                "label": v["label"],
                "desc": v["desc"],
                "required": v["required"],
            }
            for k, v in REQUIRED_FIELD_META.items()
        ],
    }


def load_raw_data(
    path: str | Path,
    column_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load the UCI Online Retail dataset from an Excel or CSV file.

    All columns are initially read as strings so that no implicit casting
    silently corrupts values before our controlled numeric conversions.

    Args:
        path: Absolute or relative path to the raw data file.
              Supported extensions: ``.xlsx``, ``.xls``, ``.csv``.
        column_mapping: Optional explicit mapping ``{internal_snake_case → user_column}``.
              When provided the user's column names are renamed to the standard
              snake_case names before validation.  When ``None``, the automatic
              case-insensitive alias matching is used instead.

    Returns:
        DataFrame with:
        - Standardised snake_case column names
        - ``invoice_date`` parsed as ``datetime64[ns]``
        - ``quantity`` and ``unit_price`` as ``float64``
        - All other columns as ``object`` (string)

    Raises:
        FileNotFoundError: If ``path`` does not exist on disk.
        ValueError: If any required column is missing after renaming,
                    or if the file extension is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    logger.info("Loading raw data from %s", path)

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix!r}. Use .xlsx, .xls, or .csv."
        )

    logger.info("Raw file loaded: %d rows, %d columns", len(df), len(df.columns))

    # ── Rename columns to standard snake_case ────────────────────────────────
    if column_mapping:
        # Explicit user mapping: {internal_name → user_col} → invert for pandas
        rename_map = {user_col: internal for internal, user_col in column_mapping.items()}
        df = df.rename(columns=rename_map)
        logger.info("Applied user column mapping: %s", column_mapping)
    else:
        # Auto case-insensitive match against known aliases
        lower_map = {k.lower(): v for k, v in RAW_COLUMN_MAP.items()}
        rename_map = {col: lower_map[col.lower()] for col in df.columns if col.lower() in lower_map}
        df = df.rename(columns=rename_map)

    # ── Validate mandatory columns ────────────────────────────────────────────
    missing_mandatory = [c for c in MANDATORY_COLUMNS if c not in df.columns]
    if missing_mandatory:
        raise ValueError(
            f"Mandatory columns missing after mapping: {missing_mandatory}. "
            f"These fields must be mapped: InvoiceNo, StockCode, Description, "
            f"Quantity, InvoiceDate, UnitPrice. "
            f"Found after renaming: {list(df.columns)}"
        )

    # ── Inject placeholders for optional missing columns ──────────────────────
    for col, default in OPTIONAL_COLUMN_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            logger.warning(
                "Optional column '%s' not present — injecting placeholder '%s'",
                col, default,
            )

    # ── Parse invoice_date ────────────────────────────────────────────────────
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    nat_count = df["invoice_date"].isna().sum()
    if nat_count > 0:
        logger.warning(
            "Dropping %d rows with unparseable invoice_date", nat_count
        )
        df = df.dropna(subset=["invoice_date"])

    # ── Cast numeric columns ──────────────────────────────────────────────────
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    df = df.reset_index(drop=True)
    logger.info("Extract complete: %d rows ready for cleaning", len(df))
    return df
