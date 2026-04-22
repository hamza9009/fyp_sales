"""Tests for the richer inventory stock replay service."""

from datetime import date, timedelta

import pandas as pd

from app.services import inventory_service


def _history_df(days: int = 60, qty: float = 10.0) -> pd.DataFrame:
    start = date(2011, 10, 1)
    return pd.DataFrame(
        {
            "stock_code": ["85123A"] * days,
            "sale_date": [pd.Timestamp(start + timedelta(days=i)) for i in range(days)],
            "total_quantity": [qty] * days,
            "total_revenue": [qty * 2.5] * days,
            "num_transactions": [1] * days,
        }
    )


def test_compute_inventory_signal_exposes_simulation_fields(monkeypatch):
    history = _history_df()

    monkeypatch.setattr(
        "app.services.inventory_service.data_store.get_product_sales_history",
        lambda stock_code, last_n=None: history,
    )
    monkeypatch.setattr(
        "app.services.inventory_service.data_store.get_product_info",
        lambda stock_code: {"stock_code": stock_code, "description": "WHITE HEART", "unit_price": 2.5},
    )
    monkeypatch.setattr(
        "app.services.inventory_service.forecast_service.generate_forecast",
        lambda stock_code, horizon=14: [
            {
                "forecast_date": history["sale_date"].max().date() + timedelta(days=i + 1),
                "predicted_quantity": 11.0,
                "predicted_revenue": 27.5,
            }
            for i in range(horizon)
        ],
    )

    signal = inventory_service.compute_inventory_signal("85123A")

    required = {
        "initial_stock_level",
        "target_stock_level",
        "pending_restock_quantity",
        "stockout_days_last_30",
        "projected_stockout_days",
        "service_level_last_30",
    }
    assert required.issubset(signal.keys())
    assert 0.0 <= signal["stockout_risk"] <= 1.0
    assert 0.0 <= signal["service_level_last_30"] <= 1.0
    assert signal["target_stock_level"] >= signal["reorder_point"]


def test_compute_inventory_signal_handles_zero_demand_history(monkeypatch):
    history = _history_df(qty=0.0)

    monkeypatch.setattr(
        "app.services.inventory_service.data_store.get_product_sales_history",
        lambda stock_code, last_n=None: history,
    )
    monkeypatch.setattr(
        "app.services.inventory_service.data_store.get_product_info",
        lambda stock_code: {"stock_code": stock_code, "description": "WHITE HEART", "unit_price": 2.5},
    )
    monkeypatch.setattr(
        "app.services.inventory_service.forecast_service.generate_forecast",
        lambda stock_code, horizon=14: [],
    )

    signal = inventory_service.compute_inventory_signal("85123A")

    assert signal["avg_daily_demand"] == 0.0
    assert signal["stockout_risk"] == 0.0
    assert signal["service_level_last_30"] == 1.0
