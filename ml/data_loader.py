"""
ML data loader — Phase 3.

Reads the Phase 2 Parquet outputs and returns a validated, feature-enriched
DataFrame ready for the train/test splitter, plus a fitted LabelEncoder for
the ``stock_code`` column.

Enrichment applied here (ML-specific, not ETL-level):
  - ``stock_code_encoded`` — ordinal encoding via sklearn LabelEncoder
  - ``stock_code_target_encoded`` — computed later inside model wrappers /
    CV estimators using training-fold targets only (leakage-safe)
  - ``unit_price``         — joined from products.parquet (static per SKU)
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml.config import (
    ENCODER_ARTIFACT,
    FEATURE_COLS,
    FEATURES_PARQUET,
    ID_COLS,
    PRODUCTS_PARQUET,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


def load_features(
    features_path: str | Path | None = None,
    products_path: str | Path | None = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Load, validate, and enrich the ML feature dataset.

    Steps
    -----
    1. Read ``features.parquet`` produced by Phase 2 ETL.
    2. Join ``unit_price`` from ``products.parquet``.
    3. Fit a :class:`sklearn.preprocessing.LabelEncoder` on ``stock_code``
       and add the resulting ``stock_code_encoded`` column.
    4. Leave target encoding to the training wrappers so it is fit only on
       the relevant training fold and never on future / validation targets.

    Args:
        features_path: Override path to ``features.parquet``.
        products_path: Override path to ``products.parquet``.

    Returns:
        Tuple of ``(enriched_df, fitted_encoder)``.
        The encoder is needed at prediction time to encode unseen stock codes.

    Raises:
        FileNotFoundError: If any required Parquet file is missing.
        ValueError: If required columns are absent.
    """
    feat_path = Path(features_path) if features_path else FEATURES_PARQUET
    prod_path = Path(products_path) if products_path else PRODUCTS_PARQUET

    for p in (feat_path, prod_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {p}\n"
                "Run the Phase 2 ETL pipeline first:\n"
                "  python -m etl.pipeline --raw <file> --processed-dir data/processed --skip-db"
            )

    logger.info("Loading features from %s", feat_path)
    df = pd.read_parquet(feat_path)
    df["sale_date"] = pd.to_datetime(df["sale_date"])

    # ── Validate base columns ──────────────────────────────────────────────────
    base_required = {TARGET_COL, *ID_COLS}
    missing = base_required - set(df.columns)
    if missing:
        raise ValueError(f"features.parquet is missing columns: {sorted(missing)}")

    # ── Join unit_price from products catalogue ────────────────────────────────
    products = pd.read_parquet(prod_path)[["stock_code", "unit_price"]]
    df = df.merge(products, on="stock_code", how="left")

    # Fill missing prices with the global median (robustness for rare SKUs)
    median_price = df["unit_price"].median()
    df["unit_price"] = df["unit_price"].fillna(median_price)

    # ── Encode stock_code → integer ────────────────────────────────────────────
    encoder = LabelEncoder()
    df["stock_code_encoded"] = encoder.fit_transform(df["stock_code"])

    # ── Validate final feature set ─────────────────────────────────────────────
    missing_features = set(FEATURE_COLS) - set(df.columns)
    if missing_features:
        raise ValueError(f"Enriched DataFrame is missing features: {sorted(missing_features)}")

    df = (
        df.sort_values(["stock_code", "sale_date"])
        .reset_index(drop=True)
    )

    logger.info(
        "Loaded %d rows × %d cols | %d products | %s → %s | encoder classes: %d",
        len(df), len(df.columns),
        df["stock_code"].nunique(),
        df["sale_date"].min().date(),
        df["sale_date"].max().date(),
        len(encoder.classes_),
    )
    return df, encoder
