"""Tests for ml.trainer pipeline configuration."""

from ml.trainer import _build_search_specs


def test_trainer_search_specs_only_use_xgboost_and_lightgbm():
    names = [spec.name for spec, _ in _build_search_specs()]
    assert names == ["XGBoost", "LightGBM"]
