"""Pydantic response schemas for the forecast endpoints."""

from datetime import date

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    """Predicted demand for a single future date."""

    forecast_date: date = Field(..., description="The date being predicted.")
    predicted_quantity: float = Field(..., ge=0, description="Predicted units sold.")
    predicted_revenue: float | None = Field(
        None, ge=0, description="Predicted revenue (quantity × unit_price). None if price unknown."
    )

    model_config = {"json_schema_extra": {"example": {
        "forecast_date": "2011-12-10",
        "predicted_quantity": 14.5,
        "predicted_revenue": 21.75,
    }}}


class ForecastResponse(BaseModel):
    """Full forecast response for a given product."""

    stock_code: str = Field(..., description="Product identifier (StockCode).")
    description: str | None = Field(None, description="Product description.")
    model_name: str = Field(..., description="Name of the model used for predictions.")
    horizon_days: int = Field(..., description="Number of days forecasted.")
    forecast: list[ForecastPoint] = Field(..., description="Per-day predictions.")

    model_config = {"json_schema_extra": {"example": {
        "stock_code": "85123A",
        "description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "model_name": "LightGBM",
        "horizon_days": 7,
        "forecast": [
            {"forecast_date": "2011-12-10", "predicted_quantity": 14.5, "predicted_revenue": 21.75}
        ],
    }}}
