"""Tests for time-aware hyperparameter tuning helpers."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from ml.config import FEATURE_COLS
from ml.tuning import DateBasedTimeSeriesSplit, ModelSearchSpec, run_randomized_search


def _make_frame(n_dates: int = 12, rows_per_date: int = 3) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")

    rows = []
    targets = []
    groups = []
    for offset, day in enumerate(dates):
        for _ in range(rows_per_date):
            row = {col: float(rng.integers(1, 20)) for col in FEATURE_COLS}
            row["lag_1"] = float(max(offset, 1))
            rows.append(row)
            targets.append(float(offset + rng.normal(scale=0.1)))
            groups.append(day)

    return pd.DataFrame(rows), pd.Series(targets), pd.Series(groups)


def test_date_based_time_series_split_preserves_date_boundaries():
    X, _, groups = _make_frame()
    splitter = DateBasedTimeSeriesSplit(n_splits=3)

    for train_idx, test_idx in splitter.split(X, groups=groups):
        train_dates = pd.to_datetime(groups.iloc[train_idx])
        test_dates = pd.to_datetime(groups.iloc[test_idx])
        assert train_dates.max() < test_dates.min()
        assert set(train_dates).isdisjoint(set(test_dates))


def test_run_randomized_search_returns_unprefixed_params():
    X, y, groups = _make_frame()
    spec = ModelSearchSpec(
        name="Decision Tree",
        estimator=DecisionTreeRegressor(random_state=0),
        param_distributions={
            "max_depth": [2, None],
            "min_samples_split": [2, 4],
        },
        use_log_transform=False,
    )

    result = run_randomized_search(
        X_train=X,
        y_train=y,
        sale_dates=groups,
        spec=spec,
        n_splits=3,
        n_iter=2,
    )

    assert result.best_cv_rmse >= 0.0
    assert set(result.best_params).issubset({"max_depth", "min_samples_split"})
    assert result.n_splits == 3
