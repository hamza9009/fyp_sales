"""
Phase 4 — API endpoint tests.

All tests mock the service layer so they run without a database,
without ML artifacts on disk, and without Parquet files.
This lets CI pass anywhere; integration against real data is done
via the running server (Postman / manual curl).
"""

from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)

# ── Shared mock payloads ────────────────────────────────────────────────────────

_FORECAST_PAYLOAD = [
    {"forecast_date": date(2011, 12, 10), "predicted_quantity": 14.5, "predicted_revenue": 21.75},
    {"forecast_date": date(2011, 12, 11), "predicted_quantity": 12.0, "predicted_revenue": 18.0},
]

_INVENTORY_PAYLOAD = {
    "stock_code": "85123A",
    "description": "WHITE HANGING HEART T-LIGHT HOLDER",
    "as_of_date": date(2011, 12, 9),
    "avg_daily_demand": 18.3,
    "predicted_next_demand": 20.1,
    "initial_stock_level": 549.0,
    "target_stock_level": 640.5,
    "simulated_stock_level": 549.0,
    "reorder_point": 128.1,
    "days_of_stock_remaining": 30,
    "pending_restock_quantity": 72.0,
    "next_restock_date": date(2011, 12, 13),
    "last_restock_date": date(2011, 12, 5),
    "stockout_days_last_30": 1,
    "projected_stockout_days": 0,
    "service_level_last_30": 0.987,
    "stockout_risk": 0.05,
    "alert_level": "low",
    "reorder_suggested": False,
}

_DASHBOARD_PAYLOAD = {
    "total_products": 2435,
    "total_revenue": 9747747.93,
    "total_quantity": 5176450,
    "date_range_start": date(2011, 1, 3),
    "date_range_end": date(2011, 12, 9),
    "total_days": 341,
    "last_30_days_trend": [
        {"sale_date": date(2011, 11, 10), "total_quantity": 12000, "total_revenue": 20000.0}
    ],
    "top_products": [
        {
            "stock_code": "85123A",
            "description": "WHITE HANGING HEART T-LIGHT HOLDER",
            "total_revenue": 100000.0,
            "total_quantity": 50000,
            "num_days_active": 200,
        }
    ],
    "best_model": {
        "model_name": "Random Forest",
        "mae": 24.31,
        "rmse": 61.11,
        "train_rows": 129005,
        "test_rows": 62238,
        "cutoff_date": "2011-10-06",
    },
}

_METRICS_PAYLOAD = {
    "best_model": "Random Forest",
    "split": {
        "cutoff_date": "2011-10-06",
        "train_rows": 129005,
        "test_rows": 62238,
        "train_products": 2092,
        "test_products": 2300,
    },
    "models": [
        {"model_name": "Random Forest", "mae": 24.31, "rmse": 61.11,
         "train_time_sec": 8.54, "rank": 1, "is_best": True},
        {"model_name": "XGBoost", "mae": 24.51, "rmse": 62.28,
         "train_time_sec": 0.31, "rank": 2, "is_best": False},
        {"model_name": "Naive (lag-1)", "mae": 24.21, "rmse": 81.44,
         "train_time_sec": 0.0, "rank": 3, "is_best": False},
    ],
}


# ── /forecast/{stock_code} ─────────────────────────────────────────────────────

class TestForecastEndpoint:
    _stock = "85123A"

    def test_returns_200_with_valid_stock(self):
        with (
            patch("app.routers.forecast.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "Test", "unit_price": 1.5}),
            patch("app.routers.forecast.forecast_service.generate_forecast", return_value=_FORECAST_PAYLOAD),
            patch("app.routers.forecast.model_loader.get_model_name", return_value="Random Forest"),
        ):
            resp = client.get(f"/api/v1/forecast/{self._stock}")
        assert resp.status_code == 200

    def test_response_schema(self):
        with (
            patch("app.routers.forecast.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "Test", "unit_price": 1.5}),
            patch("app.routers.forecast.forecast_service.generate_forecast", return_value=_FORECAST_PAYLOAD),
            patch("app.routers.forecast.model_loader.get_model_name", return_value="Random Forest"),
        ):
            data = client.get(f"/api/v1/forecast/{self._stock}").json()
        assert data["stock_code"] == self._stock
        assert "forecast" in data
        assert isinstance(data["forecast"], list)
        assert len(data["forecast"]) == 2
        assert "predicted_quantity" in data["forecast"][0]

    def test_horizon_query_param(self):
        with (
            patch("app.routers.forecast.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "Test", "unit_price": 1.5}),
            patch("app.routers.forecast.forecast_service.generate_forecast", return_value=_FORECAST_PAYLOAD[:1]) as mock_gen,
            patch("app.routers.forecast.model_loader.get_model_name", return_value="RF"),
        ):
            client.get(f"/api/v1/forecast/{self._stock}?horizon=3")
            mock_gen.assert_called_once_with(self._stock, horizon=3)

    def test_unknown_stock_returns_404(self):
        with patch("app.routers.forecast.data_store.resolve_product_identifier", return_value=None):
            resp = client.get("/api/v1/forecast/UNKNOWN_CODE")
        assert resp.status_code == 404

    def test_horizon_too_large_returns_422(self):
        with patch("app.routers.forecast.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "Test", "unit_price": 1.5}):
            resp = client.get(f"/api/v1/forecast/{self._stock}?horizon=99")
        assert resp.status_code == 422

    def test_horizon_zero_returns_422(self):
        with patch("app.routers.forecast.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "Test", "unit_price": 1.5}):
            resp = client.get(f"/api/v1/forecast/{self._stock}?horizon=0")
        assert resp.status_code == 422

    def test_predicted_quantity_non_negative(self):
        with (
            patch("app.routers.forecast.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "Test", "unit_price": 1.5}),
            patch("app.routers.forecast.forecast_service.generate_forecast", return_value=_FORECAST_PAYLOAD),
            patch("app.routers.forecast.model_loader.get_model_name", return_value="RF"),
        ):
            data = client.get(f"/api/v1/forecast/{self._stock}").json()
        for point in data["forecast"]:
            assert point["predicted_quantity"] >= 0

    def test_product_name_is_accepted(self):
        with (
            patch("app.routers.forecast.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "WHITE HEART", "unit_price": 1.5}),
            patch("app.routers.forecast.forecast_service.generate_forecast", return_value=_FORECAST_PAYLOAD),
            patch("app.routers.forecast.model_loader.get_model_name", return_value="RF"),
        ):
            data = client.get("/api/v1/forecast/WHITE%20HEART").json()
        assert data["stock_code"] == self._stock


# ── /inventory/{stock_code} ────────────────────────────────────────────────────

class TestInventoryEndpoint:
    _stock = "85123A"

    def test_returns_200_with_valid_stock(self):
        with (
            patch("app.routers.inventory.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "Test", "unit_price": 1.5}),
            patch("app.routers.inventory.inventory_service.compute_inventory_signal", return_value=_INVENTORY_PAYLOAD),
        ):
            resp = client.get(f"/api/v1/inventory/{self._stock}")
        assert resp.status_code == 200

    def test_response_schema(self):
        with (
            patch("app.routers.inventory.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "Test", "unit_price": 1.5}),
            patch("app.routers.inventory.inventory_service.compute_inventory_signal", return_value=_INVENTORY_PAYLOAD),
        ):
            data = client.get(f"/api/v1/inventory/{self._stock}").json()
        required = {
            "stock_code", "as_of_date", "avg_daily_demand",
            "predicted_next_demand", "simulated_stock_level",
            "reorder_point", "days_of_stock_remaining",
            "stockout_risk", "alert_level", "reorder_suggested",
            "initial_stock_level", "target_stock_level", "pending_restock_quantity",
            "stockout_days_last_30", "projected_stockout_days", "service_level_last_30",
        }
        assert required.issubset(data.keys())

    def test_stockout_risk_bounded(self):
        with (
            patch("app.routers.inventory.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "Test", "unit_price": 1.5}),
            patch("app.routers.inventory.inventory_service.compute_inventory_signal", return_value=_INVENTORY_PAYLOAD),
        ):
            data = client.get(f"/api/v1/inventory/{self._stock}").json()
        assert 0.0 <= data["stockout_risk"] <= 1.0

    def test_alert_level_valid(self):
        with (
            patch("app.routers.inventory.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "Test", "unit_price": 1.5}),
            patch("app.routers.inventory.inventory_service.compute_inventory_signal", return_value=_INVENTORY_PAYLOAD),
        ):
            data = client.get(f"/api/v1/inventory/{self._stock}").json()
        assert data["alert_level"] in {"low", "medium", "high", "critical"}

    def test_unknown_stock_returns_404(self):
        with patch("app.routers.inventory.data_store.resolve_product_identifier", return_value=None):
            resp = client.get("/api/v1/inventory/UNKNOWN_CODE")
        assert resp.status_code == 404

    def test_product_name_is_accepted(self):
        with (
            patch("app.routers.inventory.data_store.resolve_product_identifier", return_value={"stock_code": self._stock, "description": "WHITE HEART", "unit_price": 1.5}),
            patch("app.routers.inventory.inventory_service.compute_inventory_signal", return_value=_INVENTORY_PAYLOAD),
        ):
            data = client.get("/api/v1/inventory/WHITE%20HEART").json()
        assert data["stock_code"] == self._stock


# ── /dashboard/summary ─────────────────────────────────────────────────────────

class TestDashboardEndpoint:
    def test_returns_200(self):
        with patch("app.routers.dashboard.dashboard_service.get_dashboard_summary", return_value=_DASHBOARD_PAYLOAD):
            resp = client.get("/api/v1/dashboard/summary")
        assert resp.status_code == 200

    def test_response_schema(self):
        with patch("app.routers.dashboard.dashboard_service.get_dashboard_summary", return_value=_DASHBOARD_PAYLOAD):
            data = client.get("/api/v1/dashboard/summary").json()
        required = {
            "total_products", "total_revenue", "total_quantity",
            "date_range_start", "date_range_end", "total_days",
            "last_30_days_trend", "top_products", "best_model",
        }
        assert required.issubset(data.keys())

    def test_top_products_present(self):
        with patch("app.routers.dashboard.dashboard_service.get_dashboard_summary", return_value=_DASHBOARD_PAYLOAD):
            data = client.get("/api/v1/dashboard/summary").json()
        assert isinstance(data["top_products"], list)
        assert len(data["top_products"]) > 0

    def test_trend_present(self):
        with patch("app.routers.dashboard.dashboard_service.get_dashboard_summary", return_value=_DASHBOARD_PAYLOAD):
            data = client.get("/api/v1/dashboard/summary").json()
        assert isinstance(data["last_30_days_trend"], list)

    def test_best_model_keys(self):
        with patch("app.routers.dashboard.dashboard_service.get_dashboard_summary", return_value=_DASHBOARD_PAYLOAD):
            data = client.get("/api/v1/dashboard/summary").json()
        bm = data["best_model"]
        assert {"model_name", "mae", "rmse"}.issubset(bm.keys())


# ── /models/metrics ────────────────────────────────────────────────────────────

class TestModelsMetricsEndpoint:
    def test_returns_200(self):
        with patch("app.routers.models_router.metrics_service.get_model_metrics", return_value=_METRICS_PAYLOAD):
            resp = client.get("/api/v1/models/metrics")
        assert resp.status_code == 200

    def test_response_schema(self):
        with patch("app.routers.models_router.metrics_service.get_model_metrics", return_value=_METRICS_PAYLOAD):
            data = client.get("/api/v1/models/metrics").json()
        assert "best_model" in data
        assert "models" in data
        assert "split" in data

    def test_models_are_ranked(self):
        with patch("app.routers.models_router.metrics_service.get_model_metrics", return_value=_METRICS_PAYLOAD):
            data = client.get("/api/v1/models/metrics").json()
        ranks = [m["rank"] for m in data["models"]]
        assert ranks == sorted(ranks)

    def test_exactly_one_best_model(self):
        with patch("app.routers.models_router.metrics_service.get_model_metrics", return_value=_METRICS_PAYLOAD):
            data = client.get("/api/v1/models/metrics").json()
        best_count = sum(1 for m in data["models"] if m["is_best"])
        assert best_count == 1

    def test_split_fields(self):
        with patch("app.routers.models_router.metrics_service.get_model_metrics", return_value=_METRICS_PAYLOAD):
            data = client.get("/api/v1/models/metrics").json()
        split = data["split"]
        assert {"cutoff_date", "train_rows", "test_rows"}.issubset(split.keys())

    def test_missing_artifact_returns_503(self):
        with patch(
            "app.routers.models_router.metrics_service.get_model_metrics",
            side_effect=FileNotFoundError("metrics_report.json not found"),
        ):
            resp = client.get("/api/v1/models/metrics")
        assert resp.status_code == 503
