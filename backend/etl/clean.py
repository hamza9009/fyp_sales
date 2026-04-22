"""
ETL Clean step.

Cleaning rules applied in order:
1. Remove cancellation invoices — ``invoice_no`` starts with ``'C'``
2. Impute recoverable missing values:
   - ``description`` from per-``stock_code`` mode, then global mode
   - ``stock_code`` from per-``description`` mode where possible
   - ``quantity`` from per-``stock_code`` mean, then global mean
   - ``unit_price`` from per-``stock_code`` mean, then global mean
   - ``country`` / ``customer_id`` from global mode
   - ``invoice_no`` from synthetic placeholders to preserve row identity
3. Drop rows where essential keys (``stock_code`` / ``invoice_date``) still cannot be recovered
4. Remove returns and zero-quantity rows — ``quantity < MIN_QUANTITY``
5. Remove rows where ``unit_price`` is still invalid after imputation
6. Strip whitespace from remaining string columns

Every imputation and removal is counted and logged explicitly so the pipeline is fully auditable.
"""

import logging

import numpy as np
import pandas as pd

from etl.config import MIN_QUANTITY, MIN_UNIT_PRICE

logger = logging.getLogger(__name__)

# Strings that represent missing values when the file was read as dtype=str
_NULL_STRINGS: frozenset[str] = frozenset({"", "nan", "NaN", "None", "none", "null", "NULL"})


def _normalise_categorical(series: pd.Series) -> pd.Series:
    """Return a trimmed string series with null-like literals converted to ``pd.NA``."""
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.isin(_NULL_STRINGS), pd.NA)


def _mode(series: pd.Series) -> str | None:
    """Return the first non-null mode value, or ``None`` when unavailable."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    modes = non_null.mode(dropna=True)
    if modes.empty:
        return None
    return str(modes.iloc[0])


def _fill_with_group_mode(df: pd.DataFrame, target: str, group_key: str) -> int:
    """Fill missing categorical values using the mode within each group."""
    source = df[df[group_key].notna() & df[target].notna()]
    if source.empty:
        return 0

    mapping = source.groupby(group_key)[target].agg(_mode)
    missing = df[target].isna() & df[group_key].notna()
    before = int(df[target].isna().sum())
    df.loc[missing, target] = df.loc[missing, group_key].map(mapping)
    return before - int(df[target].isna().sum())


def _fill_with_global_mode(df: pd.DataFrame, target: str) -> int:
    """Fill missing categorical values using the dataset-level mode."""
    fill_value = _mode(df[target])
    if fill_value is None:
        return 0

    before = int(df[target].isna().sum())
    df.loc[df[target].isna(), target] = fill_value
    return before - int(df[target].isna().sum())


def _fill_with_constant(df: pd.DataFrame, target: str, value: str) -> int:
    """Fill missing categorical values with a fixed fallback placeholder."""
    before = int(df[target].isna().sum())
    df.loc[df[target].isna(), target] = value
    return before - int(df[target].isna().sum())


def _fill_with_group_mean(
    df: pd.DataFrame,
    target: str,
    group_key: str,
    valid_mask: pd.Series,
    *,
    round_to_int: bool = False,
) -> int:
    """Fill missing numeric values using the mean inside each group."""
    source = df[valid_mask & df[group_key].notna()]
    if source.empty:
        return 0

    mapping = source.groupby(group_key)[target].mean()
    missing = df[target].isna() & df[group_key].notna()
    before = int(df[target].isna().sum())
    filled = df.loc[missing, group_key].map(mapping)
    if round_to_int:
        filled = filled.round()
    df.loc[missing, target] = filled
    return before - int(df[target].isna().sum())


def _fill_with_global_mean(
    df: pd.DataFrame,
    target: str,
    valid_mask: pd.Series,
    *,
    round_to_int: bool = False,
) -> int:
    """Fill missing numeric values using the dataset-level mean."""
    valid_values = df.loc[valid_mask, target]
    if valid_values.empty:
        return 0

    fill_value = float(valid_values.mean())
    if round_to_int:
        fill_value = round(fill_value)
    before = int(df[target].isna().sum())
    df.loc[df[target].isna(), target] = fill_value
    return before - int(df[target].isna().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all data-quality rules to the raw DataFrame.

    Args:
        df: DataFrame produced by :func:`etl.extract.load_raw_data`.
            Must contain columns: ``invoice_no``, ``stock_code``,
            ``description``, ``quantity``, ``unit_price``.

    Returns:
        Cleaned DataFrame with the index reset. Missing values are imputed
        where that can be done without corrupting product/date identity.
    """
    initial = len(df)
    logger.info("Clean step started: %d input rows", initial)

    df = df.copy()

    # Normalise categorical columns once so all later rules see real nulls.
    for col in ("invoice_no", "stock_code", "description", "customer_id", "country"):
        if col in df.columns:
            df[col] = _normalise_categorical(df[col])

    # 1. Remove cancellation invoices (invoice_no starts with 'C')
    cancel_mask = df["invoice_no"].fillna("").str.startswith("C")
    n_cancel = int(cancel_mask.sum())
    df = df.loc[~cancel_mask].copy()
    logger.info("  [1] Removed %d cancellation invoices", n_cancel)

    # 2. Impute recoverable missing values
    imputed_description = _fill_with_group_mode(df, "description", "stock_code")
    imputed_description += _fill_with_global_mode(df, "description")
    unresolved_description = _fill_with_constant(df, "description", "Unknown description")

    imputed_stock_code = _fill_with_group_mode(df, "stock_code", "description")

    # Preserve transaction counting without collapsing multiple rows onto a shared mode.
    missing_invoice = df["invoice_no"].isna()
    n_invoice = int(missing_invoice.sum())
    if n_invoice:
        df.loc[missing_invoice, "invoice_no"] = [
            f"MISSING_INV_{i}" for i in df.index[missing_invoice]
        ]

    quantity_valid = df["quantity"].notna() & (df["quantity"] >= MIN_QUANTITY)
    imputed_quantity = _fill_with_group_mean(
        df, "quantity", "stock_code", quantity_valid, round_to_int=True
    )
    quantity_valid = df["quantity"].notna() & (df["quantity"] >= MIN_QUANTITY)
    imputed_quantity += _fill_with_global_mean(
        df, "quantity", quantity_valid, round_to_int=True
    )

    # Treat non-positive prices as missing before mean imputation.
    bad_price_mask = df["unit_price"].isna() | (df["unit_price"] <= MIN_UNIT_PRICE)
    n_bad_price = int(bad_price_mask.sum())
    df.loc[bad_price_mask, "unit_price"] = np.nan

    price_valid = df["unit_price"].notna() & (df["unit_price"] > MIN_UNIT_PRICE)
    imputed_price = _fill_with_group_mean(df, "unit_price", "stock_code", price_valid)
    price_valid = df["unit_price"].notna() & (df["unit_price"] > MIN_UNIT_PRICE)
    imputed_price += _fill_with_global_mean(df, "unit_price", price_valid)

    imputed_country = _fill_with_global_mode(df, "country") if "country" in df.columns else 0
    imputed_customer = (
        _fill_with_global_mode(df, "customer_id") if "customer_id" in df.columns else 0
    )

    logger.info(
        "  [2] Imputed missing values | description=%d stock_code=%d invoice_no=%d "
        "quantity=%d unit_price=%d country=%d customer_id=%d",
        imputed_description + unresolved_description,
        imputed_stock_code,
        n_invoice,
        imputed_quantity,
        imputed_price,
        imputed_country,
        imputed_customer,
    )
    if n_bad_price:
        logger.info("      Treated %d invalid/non-positive unit_price values as missing before imputation", n_bad_price)

    # 3. Drop rows whose essential keys are still unresolved
    essential_mask = df["stock_code"].isna() | df["invoice_date"].isna()
    n_essential = int(essential_mask.sum())
    df = df.loc[~essential_mask].copy()
    logger.info("  [3] Removed %d rows with unrecoverable stock_code/invoice_date", n_essential)

    # 4. Remove returns / zero-quantity rows
    return_mask = df["quantity"].isna() | (df["quantity"] < MIN_QUANTITY)
    n_return = int(return_mask.sum())
    df = df.loc[~return_mask].copy()
    logger.info("  [4] Removed %d return/zero-quantity rows", n_return)

    # 5. Remove unit prices that still could not be recovered
    price_mask = df["unit_price"].isna() | (df["unit_price"] <= MIN_UNIT_PRICE)
    n_price = int(price_mask.sum())
    df = df.loc[~price_mask].copy()
    logger.info("  [5] Removed %d rows with unresolved unit_price", n_price)

    # 6. Strip whitespace from remaining string columns
    for col in ("invoice_no", "country", "customer_id"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    for col in ("stock_code", "description"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    df = df.reset_index(drop=True)
    removed = initial - len(df)
    logger.info(
        "Clean step complete: %d → %d rows (%d removed, %.1f%%)",
        initial,
        len(df),
        removed,
        100.0 * removed / max(initial, 1),
    )
    return df
