"""
Time-based train/test splitter — Phase 3.

Splits the feature dataset by unique calendar dates (NOT randomly) to
prevent temporal data leakage.  All rows whose sale_date falls in the
first ``train_ratio`` fraction of distinct dates go to the training set;
the remaining rows go to the test set.

This mirrors real production behaviour: models are trained on historical
data and evaluated on unseen future dates.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ml.config import TRAIN_RATIO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitResult:
    """Container returned by :func:`time_split`."""

    train: pd.DataFrame
    test: pd.DataFrame
    cutoff_date: datetime
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train_products: int
    n_test_products: int


def time_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    date_col: str = "sale_date",
) -> SplitResult:
    """Split a feature DataFrame into train and test sets by date.

    The split is determined by the distribution of *unique* dates so that
    the cutoff reflects the temporal ordering of the data, not row counts
    (which can be skewed by popular products).

    Args:
        df: Feature DataFrame with a ``sale_date`` column
            (``datetime64[ns]`` or castable string).
        train_ratio: Fraction of unique dates allocated to training.
                     Must be in (0, 1).  Defaults to
                     :data:`ml.config.TRAIN_RATIO` (0.8).
        date_col: Name of the date column.  Default ``"sale_date"``.

    Returns:
        :class:`SplitResult` with ``train`` and ``test`` DataFrames plus
        metadata about the split boundary.

    Raises:
        ValueError: If ``train_ratio`` is not in (0, 1) or if the
                    DataFrame has fewer than 2 unique dates.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1); got {train_ratio}")

    dates = df[date_col].sort_values().unique()
    if len(dates) < 2:
        raise ValueError(
            f"DataFrame has only {len(dates)} unique date(s); "
            "need at least 2 to create a meaningful split."
        )

    cutoff_idx = max(1, int(len(dates) * train_ratio))
    cutoff_date = pd.Timestamp(dates[cutoff_idx])

    train_df = df[df[date_col] < cutoff_date].copy()
    test_df = df[df[date_col] >= cutoff_date].copy()

    result = SplitResult(
        train=train_df,
        test=test_df,
        cutoff_date=cutoff_date,
        train_start=pd.Timestamp(train_df[date_col].min()),
        train_end=pd.Timestamp(train_df[date_col].max()),
        test_start=pd.Timestamp(test_df[date_col].min()),
        test_end=pd.Timestamp(test_df[date_col].max()),
        n_train_products=train_df["stock_code"].nunique(),
        n_test_products=test_df["stock_code"].nunique(),
    )

    logger.info(
        "Time split (ratio=%.2f) | cutoff=%s | "
        "train=%d rows (%d products) | test=%d rows (%d products)",
        train_ratio,
        cutoff_date.date(),
        len(train_df),
        result.n_train_products,
        len(test_df),
        result.n_test_products,
    )
    return result
