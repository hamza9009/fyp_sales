"""Tests for prediction history persistence and schemas."""

from unittest.mock import MagicMock

from app.models.prediction_history import PredictionHistory
from app.services import persistence_service


def test_persist_prediction_history_skips_without_client_id():
    db = MagicMock()

    persistence_service.persist_prediction_history(
        db,
        client_id=None,
        endpoint="forecast",
        query_text="85123A",
        resolved_stock_code="85123A",
        model_name="LightGBM",
        horizon_days=7,
        request_payload={"horizon_days": 7},
        response_payload={"stock_code": "85123A"},
    )

    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_persist_prediction_history_saves_row():
    db = MagicMock()

    persistence_service.persist_prediction_history(
        db,
        client_id="client-123",
        endpoint="forecast",
        query_text="white heart",
        resolved_stock_code="85123A",
        model_name="LightGBM",
        horizon_days=7,
        request_payload={"horizon_days": 7},
        response_payload={"stock_code": "85123A", "forecast": []},
    )

    db.add.assert_called_once()
    saved_row = db.add.call_args.args[0]
    assert isinstance(saved_row, PredictionHistory)
    assert saved_row.client_id == "client-123"
    assert saved_row.endpoint == "forecast"
    assert saved_row.query_text == "white heart"
    assert saved_row.resolved_stock_code == "85123A"
    assert saved_row.model_name == "LightGBM"
    assert saved_row.horizon_days == 7
    db.commit.assert_called_once()
