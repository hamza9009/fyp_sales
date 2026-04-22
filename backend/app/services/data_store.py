"""
Data store service — Phase 4.

Loads the Phase 2 Parquet outputs once at startup and caches them in memory
as DataFrames.  All query helpers are synchronous and in-process (no DB).

Parquet files consumed:
  - data/processed/features.parquet   (ML-ready feature dataset, 191K rows)
  - data/processed/sales_daily.parquet (aggregated daily sales, 276K rows)
  - data/processed/products.parquet    (product catalogue, 3922 rows)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

_features_df: pd.DataFrame | None = None
_sales_df: pd.DataFrame | None = None
_products_df: pd.DataFrame | None = None


def load_data(processed_dir: str | Path | None = None) -> None:
    """Load all three Parquet datasets into the in-process cache.

    Called once from the FastAPI lifespan startup hook.

    Args:
        processed_dir: Override path to the processed data directory.

    Raises:
        FileNotFoundError: If any required Parquet file is missing.
    """
    global _features_df, _sales_df, _products_df

    base = Path(processed_dir) if processed_dir else _PROCESSED_DIR

    for name in ("features.parquet", "sales_daily.parquet", "products.parquet"):
        if not (base / name).exists():
            raise FileNotFoundError(
                f"Required Parquet file not found: {base / name}. "
                "Run 'python -m etl.pipeline' first."
            )

    logger.info("Loading Parquet datasets from %s", base)

    _features_df = pd.read_parquet(base / "features.parquet")
    _features_df["sale_date"] = pd.to_datetime(_features_df["sale_date"])

    _sales_df = pd.read_parquet(base / "sales_daily.parquet")
    _sales_df["sale_date"] = pd.to_datetime(_sales_df["sale_date"])

    _products_df = pd.read_parquet(base / "products.parquet")

    logger.info(
        "Data loaded — features: %d rows | sales: %d rows | products: %d rows",
        len(_features_df), len(_sales_df), len(_products_df),
    )


# ── Accessor helpers ───────────────────────────────────────────────────────────

def _require(df: pd.DataFrame | None, name: str) -> pd.DataFrame:
    if df is None:
        raise RuntimeError(f"{name} not loaded. Call load_data() at startup.")
    return df


def get_features() -> pd.DataFrame:
    """Return the full ML feature DataFrame."""
    return _require(_features_df, "features")


def get_sales() -> pd.DataFrame:
    """Return the full daily sales DataFrame."""
    return _require(_sales_df, "sales")


def get_products() -> pd.DataFrame:
    """Return the product catalogue DataFrame."""
    return _require(_products_df, "products")


def get_latest_feature_row(stock_code: str) -> pd.Series | None:
    """Return the most recent feature row for ``stock_code``, or None if not found."""
    df = get_features()
    rows = df[df["stock_code"] == stock_code]
    if rows.empty:
        return None
    return rows.sort_values("sale_date").iloc[-1]


def get_product_sales_history(stock_code: str, last_n: int | None = 28) -> pd.DataFrame:
    """Return the last ``last_n`` daily sales rows for ``stock_code``.

    Returns an empty DataFrame if the product is not in the dataset.
    """
    df = get_sales()
    rows = df[df["stock_code"] == stock_code].sort_values("sale_date")
    if last_n is not None:
        rows = rows.tail(last_n)
    return rows.reset_index(drop=True)


def get_product_info(stock_code: str) -> dict | None:
    """Return product metadata (description, unit_price) or None."""
    df = get_products()
    rows = df[df["stock_code"] == stock_code]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "stock_code": str(row["stock_code"]),
        "description": str(row["description"]) if pd.notna(row.get("description")) else None,
        "unit_price": float(row["unit_price"]) if pd.notna(row.get("unit_price")) else None,
    }


def get_unit_price(stock_code: str) -> float:
    """Return the unit price for ``stock_code``, or the global median if unknown."""
    df = get_products()
    rows = df[df["stock_code"] == stock_code]
    if rows.empty or not pd.notna(rows.iloc[0].get("unit_price")):
        return float(df["unit_price"].median())
    return float(rows.iloc[0]["unit_price"])


def product_exists(stock_code: str) -> bool:
    """Return True if ``stock_code`` is present in the feature dataset."""
    return stock_code in get_features()["stock_code"].values


def resolve_stock_code(identifier: str) -> str | None:
    """Resolve a user-supplied SKU or product description to a canonical stock code.

    Resolution order:
    1. Exact stock code match
    2. Exact description match (case-insensitive)
    3. Partial stock code / description match, preferring prefix matches
    """
    query = identifier.strip()
    if not query:
        return None

    df = get_products().copy()
    stock = df["stock_code"].fillna("").astype(str)
    desc = df["description"].fillna("").astype(str)
    query_upper = query.upper()
    query_casefold = query.casefold()

    exact_stock = df[stock.str.upper() == query_upper]
    if not exact_stock.empty:
        return str(exact_stock.sort_values("stock_code").iloc[0]["stock_code"])

    exact_desc = df[desc.str.casefold() == query_casefold]
    if not exact_desc.empty:
        return str(exact_desc.sort_values("stock_code").iloc[0]["stock_code"])

    partial_mask = (
        stock.str.upper().str.contains(query_upper, na=False, regex=False)
        | desc.str.casefold().str.contains(query_casefold, na=False, regex=False)
    )
    partial = df.loc[partial_mask].copy()
    if partial.empty:
        return None

    partial_stock = partial["stock_code"].fillna("").astype(str)
    partial_desc = partial["description"].fillna("").astype(str)
    partial["_stock_prefix"] = partial_stock.str.upper().str.startswith(query_upper)
    partial["_desc_prefix"] = partial_desc.str.casefold().str.startswith(query_casefold)
    partial = partial.sort_values(
        by=["_stock_prefix", "_desc_prefix", "stock_code"],
        ascending=[False, False, True],
    )
    return str(partial.iloc[0]["stock_code"])


def resolve_product_identifier(identifier: str) -> dict | None:
    """Resolve a SKU or product description to the stored product metadata."""
    stock_code = resolve_stock_code(identifier)
    if stock_code is None:
        return None
    return get_product_info(stock_code)


def search_products(query: str, limit: int = 20) -> list[dict]:
    """Return products whose stock_code or description matches ``query``.

    Case-insensitive substring match.  Returns at most ``limit`` results.
    """
    df = get_products()
    q = query.strip().upper()
    mask = (
        df["stock_code"].str.upper().str.contains(q, na=False)
        | df["description"].str.upper().str.contains(q, na=False)
    )
    hits = df[mask].head(limit)
    return [
        {
            "stock_code":  str(r["stock_code"]),
            "description": str(r["description"]) if pd.notna(r.get("description")) else None,
            "unit_price":  float(r["unit_price"]) if pd.notna(r.get("unit_price")) else None,
        }
        for _, r in hits.iterrows()
    ]
