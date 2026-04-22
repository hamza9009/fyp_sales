"""
Tests for the three Phase 3 forecasting models.

Uses a small synthetic dataset so the tests run in milliseconds
without touching disk or the real Parquet files.
"""

import numpy as np
import pandas as pd
import pytest

from ml.config import FEATURE_COLS, TARGET_COL, TARGET_ENCODING_FEATURE_COL
from ml.models.catboost_model import CatBoostForecaster
from ml.models.lightgbm_model import LightGBMForecaster
from ml.models.naive import NaiveForecaster
from ml.models.random_forest import RandomForestForecaster
from ml.models.xgboost_model import XGBoostForecaster


# ── Shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_train, y_train, X_test, y_test) with all FEATURE_COLS present."""
    rng = np.random.default_rng(0)
    n_train, n_test = 300, 60

    def _make_X(n: int) -> pd.DataFrame:
        data = {col: rng.integers(1, 50, size=n).astype(float) for col in FEATURE_COLS}
        # Ensure lag_1 exists with realistic values
        data["lag_1"] = rng.integers(1, 30, size=n).astype(float)
        return pd.DataFrame(data)

    X_tr = _make_X(n_train)
    y_tr = pd.Series(rng.integers(1, 50, size=n_train).astype(float), name=TARGET_COL)
    X_te = _make_X(n_test)
    y_te = pd.Series(rng.integers(1, 50, size=n_test).astype(float), name=TARGET_COL)
    return X_tr, y_tr, X_te, y_te


# ── NaiveForecaster ────────────────────────────────────────────────────────────

class TestNaiveForecaster:
    def test_name(self):
        assert "Naive" in NaiveForecaster().name

    def test_predict_equals_lag1(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        model = NaiveForecaster().fit(X_tr, y_tr)
        preds = model.predict(X_te)
        np.testing.assert_array_equal(preds, X_te["lag_1"].to_numpy())

    def test_predictions_non_negative(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        # Force negative lag_1 to verify clipping
        X_te = X_te.copy()
        X_te["lag_1"] = -5.0
        preds = NaiveForecaster().fit(X_tr, y_tr).predict(X_te)
        assert (preds >= 0).all()

    def test_returns_self_on_fit(self, synthetic_data):
        X_tr, y_tr, _, _ = synthetic_data
        model = NaiveForecaster()
        assert model.fit(X_tr, y_tr) is model

    def test_no_feature_importance(self, synthetic_data):
        X_tr, y_tr, _, _ = synthetic_data
        model = NaiveForecaster().fit(X_tr, y_tr)
        assert model.get_feature_importance() is None

    def test_missing_lag1_raises(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        X_bad = X_te.drop(columns=["lag_1"])
        model = NaiveForecaster().fit(X_tr, y_tr)
        with pytest.raises(ValueError, match="lag_1"):
            model.predict(X_bad)


# ── RandomForestForecaster ─────────────────────────────────────────────────────

class TestRandomForestForecaster:
    def test_name(self):
        assert "Random Forest" in RandomForestForecaster().name

    def test_fit_predict_shape(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        preds = RandomForestForecaster().fit(X_tr, y_tr).predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_predictions_non_negative(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        preds = RandomForestForecaster().fit(X_tr, y_tr).predict(X_te)
        assert (preds >= 0).all()

    def test_feature_importance_keys(self, synthetic_data):
        X_tr, y_tr, _, _ = synthetic_data
        fi = RandomForestForecaster().fit(X_tr, y_tr).get_feature_importance()
        assert fi is not None
        assert set(fi.keys()) == set(FEATURE_COLS)

    def test_predict_before_fit_raises(self, synthetic_data):
        _, _, X_te, _ = synthetic_data
        with pytest.raises(RuntimeError, match="fit()"):
            RandomForestForecaster().predict(X_te)

    def test_custom_params(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        params = {"n_estimators": 10, "random_state": 7, "n_jobs": 1}
        preds = RandomForestForecaster(params=params).fit(X_tr, y_tr).predict(X_te)
        assert preds.shape == (len(X_te),)


# ── XGBoostForecaster ──────────────────────────────────────────────────────────

class TestXGBoostForecaster:
    def test_name(self):
        assert "XGBoost" in XGBoostForecaster().name

    def test_fit_predict_shape(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        preds = XGBoostForecaster().fit(X_tr, y_tr).predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_predictions_non_negative(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        preds = XGBoostForecaster().fit(X_tr, y_tr).predict(X_te)
        assert (preds >= 0).all()

    def test_feature_importance_normalised(self, synthetic_data):
        X_tr, y_tr, _, _ = synthetic_data
        fi = XGBoostForecaster().fit(X_tr, y_tr).get_feature_importance()
        assert fi is not None
        assert TARGET_ENCODING_FEATURE_COL in fi
        assert abs(sum(fi.values()) - 1.0) < 1e-6

    def test_predict_before_fit_raises(self, synthetic_data):
        _, _, X_te, _ = synthetic_data
        with pytest.raises(RuntimeError, match="fit()"):
            XGBoostForecaster().predict(X_te)

    def test_early_stopping_does_not_crash(self, synthetic_data):
        """Ensure early stopping with a small n_estimators still completes."""
        X_tr, y_tr, X_te, _ = synthetic_data
        params = {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 0,
            "n_jobs": 1,
            "early_stopping_rounds": 5,
        }
        preds = XGBoostForecaster(params=params).fit(X_tr, y_tr).predict(X_te)
        assert preds.shape == (len(X_te),)


# ── LightGBMForecaster ─────────────────────────────────────────────────────────

class TestLightGBMForecaster:
    def test_name(self):
        assert "LightGBM" in LightGBMForecaster().name

    def test_fit_predict_shape(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        params = {
            "n_estimators": 20,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "random_state": 0,
            "n_jobs": 1,
            "verbose": -1,
            "early_stopping_rounds": 5,
        }
        preds = LightGBMForecaster(params=params).fit(X_tr, y_tr).predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_feature_importance_keys(self, synthetic_data):
        X_tr, y_tr, _, _ = synthetic_data
        params = {
            "n_estimators": 20,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "random_state": 0,
            "n_jobs": 1,
            "verbose": -1,
            "early_stopping_rounds": 5,
        }
        fi = LightGBMForecaster(params=params).fit(X_tr, y_tr).get_feature_importance()
        assert set(fi.keys()) == set(FEATURE_COLS) | {TARGET_ENCODING_FEATURE_COL}

    def test_predict_before_fit_raises(self, synthetic_data):
        _, _, X_te, _ = synthetic_data
        with pytest.raises(RuntimeError, match="fit()"):
            LightGBMForecaster().predict(X_te)


# ── CatBoostForecaster ─────────────────────────────────────────────────────────

class TestCatBoostForecaster:
    def test_name(self):
        assert "CatBoost" in CatBoostForecaster().name

    def test_fit_predict_shape(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        params = {
            "iterations": 20,
            "depth": 4,
            "learning_rate": 0.1,
            "random_seed": 0,
            "thread_count": 1,
            "verbose": False,
            "allow_writing_files": False,
            "early_stopping_rounds": 5,
        }
        preds = CatBoostForecaster(params=params).fit(X_tr, y_tr).predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_feature_importance_keys(self, synthetic_data):
        X_tr, y_tr, _, _ = synthetic_data
        params = {
            "iterations": 20,
            "depth": 4,
            "learning_rate": 0.1,
            "random_seed": 0,
            "thread_count": 1,
            "verbose": False,
            "allow_writing_files": False,
            "early_stopping_rounds": 5,
        }
        fi = CatBoostForecaster(params=params).fit(X_tr, y_tr).get_feature_importance()
        assert set(fi.keys()) == set(FEATURE_COLS)

    def test_predict_before_fit_raises(self, synthetic_data):
        _, _, X_te, _ = synthetic_data
        with pytest.raises(RuntimeError, match="fit()"):
            CatBoostForecaster().predict(X_te)
