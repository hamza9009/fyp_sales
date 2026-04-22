"""
Time-aware hyperparameter tuning helpers for forecasting models.

The workflow keeps the final test split untouched and performs randomized
cross-validation only on the training portion using date-based folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV, TimeSeriesSplit

from ml.config import CV_SCORING, SEARCH_RANDOM_STATE, USE_LOG_TRANSFORM

logger = logging.getLogger(__name__)


def _to_python_scalar(value: Any) -> Any:
    """Convert NumPy scalars to plain Python types for JSON-safe reporting."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: _to_python_scalar(value) for key, value in params.items()}


class DateBasedTimeSeriesSplit(BaseCrossValidator):
    """TimeSeriesSplit variant that splits on unique dates, not individual rows."""

    def __init__(self, n_splits: int = 5) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2; got {n_splits}")
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("DateBasedTimeSeriesSplit requires `groups` to contain sale dates.")

        dates = pd.to_datetime(pd.Series(groups).reset_index(drop=True))
        unique_dates = pd.Index(sorted(dates.unique()))
        if len(unique_dates) <= self.n_splits:
            raise ValueError(
                f"Need more unique dates ({len(unique_dates)}) than n_splits ({self.n_splits})."
            )

        row_indices = np.arange(len(dates))
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for train_date_idx, test_date_idx in tscv.split(unique_dates):
            train_dates = set(unique_dates[train_date_idx])
            test_dates = set(unique_dates[test_date_idx])

            train_mask = dates.isin(train_dates).to_numpy()
            test_mask = dates.isin(test_dates).to_numpy()

            yield row_indices[train_mask], row_indices[test_mask]


def resolve_cv_splits(n_unique_dates: int, requested_splits: int) -> int:
    """Reduce CV folds when the dataset is too short, but keep at least 2 folds."""
    if requested_splits < 2:
        raise ValueError(f"cv_splits must be at least 2; got {requested_splits}")
    effective = min(requested_splits, max(2, n_unique_dates - 1))
    if effective < requested_splits:
        logger.info(
            "Reducing time-series CV folds from %d to %d because training data has only %d unique dates.",
            requested_splits,
            effective,
            n_unique_dates,
        )
    if n_unique_dates <= 2:
        raise ValueError(
            f"Need at least 3 unique training dates for time-series CV; got {n_unique_dates}."
        )
    return effective


@dataclass(frozen=True)
class ModelSearchSpec:
    """Tuning definition for a single candidate model."""

    name: str
    estimator: RegressorMixin
    param_distributions: dict[str, Any]
    use_log_transform: bool = USE_LOG_TRANSFORM
    param_prefix: str = ""


@dataclass(frozen=True)
class SearchResult:
    """Summary of a randomized hyperparameter search."""

    name: str
    best_params: dict[str, Any]
    best_cv_rmse: float
    n_splits: int
    n_iter: int


def _prefixed_param_distributions(
    param_distributions: dict[str, Any],
    *,
    use_log_transform: bool,
    param_prefix: str,
) -> dict[str, Any]:
    prefix = f"regressor__{param_prefix}" if use_log_transform else param_prefix
    return {f"{prefix}{key}": value for key, value in param_distributions.items()}


def _strip_param_prefixes(params: dict[str, Any], *, param_prefix: str) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        normalized = key.removeprefix("regressor__")
        if param_prefix:
            normalized = normalized.removeprefix(param_prefix)
        cleaned[normalized] = _to_python_scalar(value)
    return cleaned


def run_randomized_search(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sale_dates: pd.Series,
    spec: ModelSearchSpec,
    n_splits: int,
    n_iter: int,
    scoring: str = CV_SCORING,
    random_state: int = SEARCH_RANDOM_STATE,
) -> SearchResult:
    """Run time-aware RandomizedSearchCV on the training partition only."""
    estimator = clone(spec.estimator)
    param_distributions = _prefixed_param_distributions(
        spec.param_distributions,
        use_log_transform=spec.use_log_transform,
        param_prefix=spec.param_prefix,
    )

    if spec.use_log_transform:
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    cv = DateBasedTimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=0,
        n_jobs=1,
        refit=True,
        random_state=random_state,
        error_score="raise",
        return_train_score=False,
    )

    logger.info(
        "[%s] Starting randomized search | trials=%d | folds=%d | scoring=%s",
        spec.name,
        n_iter,
        n_splits,
        scoring,
    )
    search.fit(X_train, y_train, groups=pd.to_datetime(sale_dates))

    best_params = _strip_param_prefixes(
        search.best_params_,
        param_prefix=spec.param_prefix,
    )
    best_cv_rmse = round(float(-search.best_score_), 4)
    logger.info("[%s] Best CV RMSE=%.4f | params=%s", spec.name, best_cv_rmse, best_params)

    return SearchResult(
        name=spec.name,
        best_params=_sanitize_params(best_params),
        best_cv_rmse=best_cv_rmse,
        n_splits=n_splits,
        n_iter=len(search.cv_results_["params"]),
    )
