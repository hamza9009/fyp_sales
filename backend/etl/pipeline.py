"""
ETL Pipeline orchestrator — Phase 2.

Runs the full sequence:
    Extract → Clean → Transform → Feature Engineering → Load

Can be executed as a module::

    python -m etl.pipeline --raw data/raw/online_retail.xlsx \\
                           --processed-dir data/processed \\
                           --db-url postgresql://user:pass@localhost:5432/fyp_db

Or imported programmatically::

    from etl.pipeline import run_etl_pipeline

    result = run_etl_pipeline(
        raw_path="data/raw/online_retail.xlsx",
        processed_dir="data/processed",
        skip_db=True,
    )
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from etl.clean import clean_data
from etl.config import (
    PROCESSED_FEATURES_FILENAME,
    PROCESSED_SALES_FILENAME,
    PRODUCTS_FILENAME,
)
from etl.extract import load_raw_data
from etl.features import build_feature_dataset
from etl.load import save_parquet, upsert_products, upsert_sales_daily
from etl.transform import add_revenue, aggregate_daily, extract_products

logger = logging.getLogger(__name__)


def run_etl_pipeline(
    raw_path: str | Path,
    processed_dir: str | Path,
    db_url: str | None = None,
    skip_db: bool = False,
    column_mapping: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Execute the complete ETL pipeline.

    Steps
    -----
    1. **Extract** — load raw file, standardise columns, parse dates
    2. **Clean** — remove cancellations, returns, invalid prices/codes
    3. **Transform** — add revenue, aggregate to daily, extract product catalogue
    4. **Feature engineering** — lag, rolling, time features → ML-ready dataset
    5. **Load** — save Parquet files; optionally upsert into PostgreSQL

    Args:
        raw_path: Path to the raw UCI dataset (``*.xlsx`` or ``*.csv``).
        processed_dir: Directory where processed Parquet files are written.
                       Created automatically if it does not exist.
        db_url: SQLAlchemy-format PostgreSQL URL.  Required unless
                ``skip_db=True``.
        skip_db: When ``True``, the PostgreSQL upsert is skipped and only
                 the Parquet files are produced.  Useful for offline runs
                 and unit tests.

    Returns:
        Dictionary with three keys:

        - ``"products"`` → product catalogue DataFrame
        - ``"sales_daily"`` → daily aggregated sales DataFrame
        - ``"features"`` → ML-ready feature DataFrame

    Raises:
        ValueError: If ``db_url`` is ``None`` and ``skip_db`` is ``False``.
        FileNotFoundError: If ``raw_path`` does not exist.
    """
    if not skip_db and not db_url:
        raise ValueError(
            "db_url is required when skip_db=False. "
            "Pass --skip-db or provide --db-url."
        )

    processed_dir = Path(processed_dir)

    # ── 1. Extract ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ETL PHASE 2  |  STEP 1: EXTRACT")
    logger.info("=" * 60)
    raw_df = load_raw_data(raw_path, column_mapping=column_mapping)

    # ── 2. Clean ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ETL PHASE 2  |  STEP 2: CLEAN")
    logger.info("=" * 60)
    clean_df = clean_data(raw_df)

    # ── 3. Transform ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ETL PHASE 2  |  STEP 3: TRANSFORM")
    logger.info("=" * 60)
    clean_df = add_revenue(clean_df)
    products_df = extract_products(clean_df)
    sales_df = aggregate_daily(clean_df)

    # ── 4. Feature Engineering ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ETL PHASE 2  |  STEP 4: FEATURE ENGINEERING")
    logger.info("=" * 60)
    features_df = build_feature_dataset(sales_df)

    # ── 5. Load ─────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ETL PHASE 2  |  STEP 5: LOAD")
    logger.info("=" * 60)

    # Always write Parquet — consumed by Phase 3 (ML training)
    save_parquet(products_df, processed_dir / PRODUCTS_FILENAME)
    save_parquet(sales_df, processed_dir / PROCESSED_SALES_FILENAME)
    save_parquet(features_df, processed_dir / PROCESSED_FEATURES_FILENAME)

    if not skip_db:
        engine = create_engine(db_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        with SessionLocal() as session:
            upsert_products(products_df, session)
            upsert_sales_daily(sales_df, session)

    logger.info("=" * 60)
    logger.info("ETL PIPELINE COMPLETE")
    logger.info(
        "  products=%d  |  sales_daily=%d  |  features=%d",
        len(products_df),
        len(sales_df),
        len(features_df),
    )
    logger.info("=" * 60)

    return {"products": products_df, "sales_daily": sales_df, "features": features_df}


# ── CLI entry point ─────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


if __name__ == "__main__":
    import argparse

    _configure_logging()

    parser = argparse.ArgumentParser(
        prog="python -m etl.pipeline",
        description="Run the Phase 2 ETL pipeline.",
    )
    parser.add_argument(
        "--raw",
        required=True,
        metavar="PATH",
        help="Path to raw data file (.xlsx or .csv)",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        metavar="DIR",
        help="Output directory for processed Parquet files (default: data/processed)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        metavar="URL",
        help="PostgreSQL connection URL — omit together with --skip-db to write Parquet only",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip PostgreSQL upsert; write Parquet files only",
    )

    args = parser.parse_args()

    # If no db_url is provided, implicitly skip DB
    effective_skip_db = args.skip_db or (args.db_url is None)

    try:
        result = run_etl_pipeline(
            raw_path=args.raw,
            processed_dir=args.processed_dir,
            db_url=args.db_url,
            skip_db=effective_skip_db,
        )
        sys.exit(0)
    except Exception as exc:
        logger.exception("ETL pipeline failed: %s", exc)
        sys.exit(1)
