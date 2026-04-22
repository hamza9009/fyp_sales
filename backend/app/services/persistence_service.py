"""Persistence helpers for storing runtime outputs in PostgreSQL."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.forecasts import Forecast
from app.models.inventory_signals import AlertLevel, InventorySignal
from app.models.model_runs import ModelRun, ModelStatus
from app.models.prediction_history import PredictionHistory

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ARTIFACTS_DIR = _PROJECT_ROOT / "ml" / "artifacts"
_METRICS_JSON = _ARTIFACTS_DIR / "metrics_report.json"
_BEST_MODEL_PATH = _ARTIFACTS_DIR / "best_model.joblib"


def _load_metrics_report() -> dict:
    if not _METRICS_JSON.exists():
        return {}
    with _METRICS_JSON.open() as fh:
        return json.load(fh)


def _ensure_model_run(db: Session, model_name: str) -> ModelRun:
    """Return a persistent ModelRun row for the current best model artifact."""
    report = _load_metrics_report()
    metrics = report.get("models", {}).get(model_name, {})
    artifact_path = str(_BEST_MODEL_PATH) if _BEST_MODEL_PATH.exists() else None
    now = datetime.now(timezone.utc)

    model_run = db.execute(
        select(ModelRun)
        .where(ModelRun.model_name == model_name, ModelRun.stock_code.is_(None))
        .order_by(ModelRun.id.desc())
    ).scalars().first()

    if model_run is None:
        model_run = ModelRun(
            model_name=model_name,
            stock_code=None,
            mae=float(metrics.get("mae", 0.0)) if metrics else None,
            rmse=float(metrics.get("rmse", 0.0)) if metrics else None,
            artifact_path=artifact_path,
            status=ModelStatus.COMPLETED,
            completed_at=now,
        )
        db.add(model_run)
    else:
        model_run.mae = float(metrics.get("mae", model_run.mae or 0.0)) if metrics else model_run.mae
        model_run.rmse = float(metrics.get("rmse", model_run.rmse or 0.0)) if metrics else model_run.rmse
        model_run.artifact_path = artifact_path or model_run.artifact_path
        model_run.status = ModelStatus.COMPLETED
        model_run.completed_at = now

    db.flush()
    return model_run


def persist_forecast_response(
    db: Session,
    *,
    stock_code: str,
    model_name: str,
    forecast_points: list[dict],
) -> None:
    """Replace stored forecast rows for a product with the latest response."""
    if not forecast_points:
        return

    model_run = _ensure_model_run(db, model_name)
    forecast_dates = [point["forecast_date"] for point in forecast_points]

    db.execute(
        delete(Forecast).where(
            Forecast.stock_code == stock_code,
            Forecast.forecast_date.in_(forecast_dates),
        )
    )

    for point in forecast_points:
        db.add(
            Forecast(
                stock_code=stock_code,
                model_run_id=model_run.id,
                forecast_date=point["forecast_date"],
                predicted_quantity=float(point["predicted_quantity"]),
                predicted_revenue=(
                    float(point["predicted_revenue"])
                    if point.get("predicted_revenue") is not None
                    else None
                ),
            )
        )

    db.commit()
    logger.info(
        "Persisted %d forecast rows for %s using model_run=%s",
        len(forecast_points),
        stock_code,
        model_run.id,
    )


def persist_inventory_signal(db: Session, signal: dict) -> None:
    """Replace the stored inventory signal for a product/date."""
    signal_date = signal["as_of_date"]
    stock_code = signal["stock_code"]

    db.execute(
        delete(InventorySignal).where(
            InventorySignal.stock_code == stock_code,
            InventorySignal.signal_date == signal_date,
        )
    )
    db.add(
        InventorySignal(
            stock_code=stock_code,
            signal_date=signal_date,
            predicted_demand=float(signal["predicted_next_demand"]),
            avg_daily_demand=float(signal["avg_daily_demand"]),
            initial_stock_level=float(signal["initial_stock_level"]),
            target_stock_level=float(signal["target_stock_level"]),
            simulated_stock_level=float(signal["simulated_stock_level"]),
            reorder_point=float(signal["reorder_point"]),
            days_of_stock_remaining=int(signal["days_of_stock_remaining"]),
            pending_restock_quantity=float(signal["pending_restock_quantity"]),
            stockout_days_last_30=int(signal["stockout_days_last_30"]),
            projected_stockout_days=int(signal["projected_stockout_days"]),
            service_level_last_30=float(signal["service_level_last_30"]),
            next_restock_date=signal.get("next_restock_date"),
            last_restock_date=signal.get("last_restock_date"),
            stockout_risk=float(signal["stockout_risk"]),
            alert_level=AlertLevel(signal["alert_level"]),
        )
    )
    db.commit()
    logger.info("Persisted inventory signal for %s on %s", stock_code, signal_date)


def persist_prediction_history(
    db: Session,
    *,
    client_id: str | None,
    endpoint: str,
    query_text: str,
    resolved_stock_code: str,
    model_name: str | None,
    horizon_days: int | None,
    request_payload: dict | None,
    response_payload: dict,
) -> None:
    """Persist a successful forecast or inventory response for one client."""
    if not client_id:
        return

    db.add(
        PredictionHistory(
            client_id=client_id,
            endpoint=endpoint,
            query_text=query_text.strip() or resolved_stock_code,
            resolved_stock_code=resolved_stock_code,
            model_name=model_name,
            horizon_days=horizon_days,
            request_payload=request_payload,
            response_payload=response_payload,
        )
    )
    db.commit()
    logger.info(
        "Persisted %s history for %s (client=%s)",
        endpoint,
        resolved_stock_code,
        client_id,
    )
