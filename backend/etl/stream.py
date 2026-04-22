"""
ETL Stream Simulation.

Simulates near-real-time ingestion by replaying historical processed data
in configurable time-ordered batches.

This is a deterministic chronological replay mechanism — NOT a real streaming
system (no Kafka, no message broker).  Its purpose is to demonstrate and test
how the system handles incremental data arrival, and to support the
"simulate streaming ingestion" requirement of Phase 2.

Typical usage::

    from etl.stream import StreamSimulator, StreamConfig

    config = StreamConfig(batch_size=100, delay_seconds=0.5)
    simulator = StreamSimulator(sales_df, config=config)

    for batch_num, batch in enumerate(simulator.stream(), start=1):
        print(f"Batch {batch_num}: {len(batch)} rows")
        # hand batch to downstream consumer (e.g. DB upsert or ML inference)
"""

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Generator

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for :class:`StreamSimulator`.

    Attributes:
        batch_size: Maximum number of rows per yielded batch.
        delay_seconds: Artificial pause between batches in seconds.
                       Set to ``0.0`` (default) for no delay.
        start_date: Replay from this calendar date (inclusive).
                    ``None`` means from the earliest record.
        end_date: Replay up to this calendar date (inclusive).
                  ``None`` means until the latest record.
    """

    batch_size: int = 50
    delay_seconds: float = 0.0
    start_date: date | None = None
    end_date: date | None = None


class StreamSimulator:
    """Replays processed daily sales data in chronological batches.

    The simulator sorts the input DataFrame by ``sale_date`` and then
    yields non-overlapping slices of ``batch_size`` rows.  An optional
    date-range filter allows replaying only a specific window of history.

    Args:
        sales_df: Processed daily sales DataFrame.  Must contain a
                  ``sale_date`` column (``date``, ``datetime``, or
                  ``str`` parseable by pandas).
        config: Stream configuration.  Uses :class:`StreamConfig` defaults
                if not provided.
    """

    def __init__(
        self, sales_df: pd.DataFrame, config: StreamConfig | None = None
    ) -> None:
        self._config = config or StreamConfig()
        self._df = self._prepare(sales_df)
        logger.info(
            "StreamSimulator ready: %d rows, batch_size=%d, delay=%.2fs",
            len(self._df),
            self._config.batch_size,
            self._config.delay_seconds,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort and optionally filter the DataFrame by date range."""
        df = df.copy()
        df["sale_date"] = pd.to_datetime(df["sale_date"])

        if self._config.start_date:
            df = df[df["sale_date"] >= pd.Timestamp(self._config.start_date)]
        if self._config.end_date:
            df = df[df["sale_date"] <= pd.Timestamp(self._config.end_date)]

        return df.sort_values("sale_date").reset_index(drop=True)

    # ── Public interface ───────────────────────────────────────────────────────

    def stream(self) -> Generator[pd.DataFrame, None, None]:
        """Yield batches of rows in chronological order.

        Yields:
            DataFrame slice with the same schema as the input ``sales_df``.
            The final batch may contain fewer than ``batch_size`` rows.
        """
        total = len(self._df)
        batch_size = self._config.batch_size
        delay = self._config.delay_seconds
        n_batches = (total + batch_size - 1) // batch_size  # ceiling division

        for batch_idx, start in enumerate(range(0, total, batch_size)):
            batch = self._df.iloc[start : start + batch_size].copy()
            logger.debug(
                "Batch %d/%d | rows %d–%d | dates %s → %s",
                batch_idx + 1,
                n_batches,
                start,
                min(start + batch_size, total),
                batch["sale_date"].min().date(),
                batch["sale_date"].max().date(),
            )
            yield batch

            if delay > 0 and (start + batch_size) < total:
                time.sleep(delay)

    @property
    def total_rows(self) -> int:
        """Total number of rows in the (filtered) replay window."""
        return len(self._df)

    @property
    def n_batches(self) -> int:
        """Number of batches that will be yielded by :meth:`stream`."""
        return (self.total_rows + self._config.batch_size - 1) // max(
            self._config.batch_size, 1
        )
