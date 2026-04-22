"""Tests for ml.evaluate — metric computation and model comparison."""

import numpy as np
import pandas as pd
import pytest

from ml.evaluate import compare_models, compute_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([10.0, 20.0, 30.0])
        metrics = compute_metrics(y, y)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0

    def test_known_values(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        metrics = compute_metrics(y_true, y_pred)
        expected_mae = np.mean(np.abs(y_true - y_pred))
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert abs(metrics["mae"] - round(expected_mae, 4)) < 1e-6
        assert abs(metrics["rmse"] - round(expected_rmse, 4)) < 1e-6

    def test_accepts_series(self):
        y = pd.Series([5.0, 10.0, 15.0])
        metrics = compute_metrics(y, y.to_numpy())
        assert metrics["mae"] == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    def test_metric_keys(self):
        metrics = compute_metrics(np.array([1.0]), np.array([1.0]))
        assert set(metrics.keys()) == {"mae", "rmse"}

    def test_non_negative_metrics(self):
        y_true = np.random.rand(100) * 50
        y_pred = np.random.rand(100) * 50
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0


class TestCompareModels:
    def _results(self) -> dict:
        return {
            "Naive (lag-1)": {"mae": 5.0, "rmse": 7.0},
            "LightGBM": {"mae": 3.0, "rmse": 4.5},
            "XGBoost": {"mae": 2.5, "rmse": 3.8},
        }

    def test_best_model_is_rank_1(self):
        df = compare_models(self._results())
        assert df.iloc[0]["model"] == "XGBoost"
        assert df.iloc[0]["best"] == True

    def test_sorted_by_rmse(self):
        df = compare_models(self._results())
        rmse_values = df["rmse"].tolist()
        assert rmse_values == sorted(rmse_values)

    def test_rank_column(self):
        df = compare_models(self._results())
        assert list(df["rank"]) == list(range(1, len(df) + 1))

    def test_only_one_best(self):
        df = compare_models(self._results())
        assert df["best"].sum() == 1

    def test_all_models_present(self):
        results = self._results()
        df = compare_models(results)
        assert set(df["model"]) == set(results.keys())
