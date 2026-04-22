"""
Forecast router — Phase 4.

GET /forecast/{stock_code}
  Returns a multi-step daily demand forecast for the given product.
"""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.forecast import ForecastPoint, ForecastResponse
from app.services import data_store, forecast_service, model_loader, persistence_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast", tags=["Forecast"])


@router.get(
    "/{product_query}",
    response_model=ForecastResponse,
    summary="Demand forecast for a product",
    description=(
        "Returns a multi-step daily demand forecast for the given StockCode "
        "or product description. "
        "Uses the Phase 3 best model with iterative lag-feature rollover. "
        "Horizon defaults to 7 days (max 30)."
    ),
)
def get_forecast(
    product_query: str,
    horizon: int = Query(default=7, ge=1, le=30, description="Forecast horizon in days (1–30)."),
    client_id: str | None = Header(default=None, alias="X-Client-Id"),
    db: Session = Depends(get_db),
) -> ForecastResponse:
    """Generate a demand forecast for a stock code or product description.

    Args:
        product_query: Product StockCode or description (path parameter).
        horizon: Number of days to predict (query parameter, 1–30).

    Returns:
        :class:`~app.schemas.forecast.ForecastResponse` with per-day predictions.

    Raises:
        404 if the product is not found.
        503 if the model is not loaded.
    """
    product_info = data_store.resolve_product_identifier(product_query)
    if product_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_query}' not found in the dataset.",
        )
    stock_code = product_info["stock_code"]

    try:
        raw_forecast = forecast_service.generate_forecast(stock_code, horizon=horizon)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    description = product_info.get("description")
    model_name = model_loader.get_model_name()

    forecast_points = [
        ForecastPoint(
            forecast_date=point["forecast_date"],
            predicted_quantity=point["predicted_quantity"],
            predicted_revenue=point.get("predicted_revenue"),
        )
        for point in raw_forecast
    ]

    response = ForecastResponse(
        stock_code=stock_code,
        description=description,
        model_name=model_name,
        horizon_days=horizon,
        forecast=forecast_points,
    )

    try:
        persistence_service.persist_forecast_response(
            db,
            stock_code=stock_code,
            model_name=model_name,
            forecast_points=raw_forecast,
        )
    except Exception as exc:  # pragma: no cover - non-critical persistence path
        db.rollback()
        logger.warning("Forecast persistence failed for %s: %s", stock_code, exc)

    try:
        persistence_service.persist_prediction_history(
            db,
            client_id=client_id,
            endpoint="forecast",
            query_text=product_query,
            resolved_stock_code=stock_code,
            model_name=model_name,
            horizon_days=horizon,
            request_payload={"horizon_days": horizon},
            response_payload=response.model_dump(mode="json"),
        )
    except Exception as exc:  # pragma: no cover - non-critical persistence path
        db.rollback()
        logger.warning("Forecast history persistence failed for %s: %s", stock_code, exc)

    return response
