"""
ETL Load step.

Responsibilities:
- Upsert products into the ``products`` PostgreSQL table
- Upsert daily aggregated sales into the ``sales_daily`` table
- Persist processed DataFrames to Parquet files for ML consumption

Both upserts use PostgreSQL's ``INSERT … ON CONFLICT DO UPDATE`` so the
pipeline is fully idempotent — re-running it never creates duplicates.
"""

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models.products import Product
from app.models.sales_daily import SalesDaily

logger = logging.getLogger(__name__)

# Rows per DB transaction batch — keeps memory usage bounded for large datasets
_BATCH_SIZE: int = 1_000


def upsert_products(products_df: pd.DataFrame, session: Session) -> int:
    """Upsert product rows into the ``products`` table.

    Conflict target: ``stock_code`` (unique constraint).
    On conflict: update ``description``, ``unit_price``, ``country``.

    Args:
        products_df: DataFrame with columns: stock_code, description,
                     unit_price, country.
        session: Active SQLAlchemy session (will be committed inside).

    Returns:
        Total number of rows processed.
    """
    records = products_df[["stock_code", "description", "unit_price", "country"]].to_dict(
        orient="records"
    )
    total = 0

    for i in range(0, len(records), _BATCH_SIZE):
        batch = records[i : i + _BATCH_SIZE]
        stmt = pg_insert(Product.__table__).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_code"],
            set_={
                "description": stmt.excluded.description,
                "unit_price": stmt.excluded.unit_price,
                "country": stmt.excluded.country,
            },
        )
        session.execute(stmt)
        total += len(batch)
        logger.debug("Upserted products batch %d–%d", i, i + len(batch))

    session.commit()
    logger.info("Products upserted: %d rows", total)
    return total


def upsert_sales_daily(sales_df: pd.DataFrame, session: Session) -> int:
    """Upsert daily sales rows into the ``sales_daily`` table.

    Conflict target: unique index on (stock_code, sale_date).
    On conflict: update total_quantity, total_revenue, num_transactions.

    Args:
        sales_df: DataFrame with columns: stock_code, sale_date,
                  total_quantity, total_revenue, num_transactions.
        session: Active SQLAlchemy session (will be committed inside).

    Returns:
        Total number of rows processed.
    """
    cols = ["stock_code", "sale_date", "total_quantity", "total_revenue", "num_transactions"]
    records = sales_df[cols].to_dict(orient="records")
    total = 0

    for i in range(0, len(records), _BATCH_SIZE):
        batch = records[i : i + _BATCH_SIZE]
        stmt = pg_insert(SalesDaily.__table__).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_code", "sale_date"],
            set_={
                "total_quantity": stmt.excluded.total_quantity,
                "total_revenue": stmt.excluded.total_revenue,
                "num_transactions": stmt.excluded.num_transactions,
            },
        )
        session.execute(stmt)
        total += len(batch)
        logger.debug("Upserted sales_daily batch %d–%d", i, i + len(batch))

    session.commit()
    logger.info("sales_daily upserted: %d rows", total)
    return total


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a DataFrame to a Parquet file.

    The parent directory is created if it does not already exist.

    Args:
        df: DataFrame to persist.
        path: Target file path (must end in ``.parquet``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %d rows → %s", len(df), path)
