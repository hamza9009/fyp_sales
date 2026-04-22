"""Pydantic response schemas for the inventory intelligence endpoints."""

from datetime import date

from pydantic import BaseModel, Field


class InventoryResponse(BaseModel):
    """Inventory intelligence signal for a product."""

    stock_code: str = Field(..., description="Product identifier.")
    description: str | None = Field(None, description="Product description.")
    as_of_date: date = Field(..., description="Latest date used to compute signals.")

    # Demand statistics
    avg_daily_demand: float = Field(..., ge=0, description="Average daily units sold (last 28 days).")
    predicted_next_demand: float = Field(..., ge=0, description="Model prediction for next day demand.")

    # Simulated inventory (UCI dataset has no stock-on-hand column)
    initial_stock_level: float = Field(..., ge=0, description="Initial simulated stock used to start the replay.")
    target_stock_level: float = Field(..., ge=0, description="Order-up-to target stock level used by the replenishment policy.")
    simulated_stock_level: float = Field(
        ..., ge=0,
        description=(
            "Simulated current stock level. "
            "NOTE: UCI data has no stock column — this is estimated from demand history."
        ),
    )
    reorder_point: float = Field(
        ..., ge=0,
        description="Stock level at which a reorder should be triggered (lead_time × avg_demand).",
    )
    days_of_stock_remaining: int = Field(
        ..., ge=0, description="Estimated days until stockout at current demand rate."
    )
    pending_restock_quantity: float = Field(..., ge=0, description="Units already ordered but not yet received.")
    next_restock_date: date | None = Field(None, description="Expected arrival date of the next pending replenishment.")
    last_restock_date: date | None = Field(None, description="Most recent replenishment arrival observed in the simulation.")
    stockout_days_last_30: int = Field(..., ge=0, description="Number of simulated stockout days in the last 30 days.")
    projected_stockout_days: int = Field(..., ge=0, description="Projected stockout days across the forward simulation horizon.")
    service_level_last_30: float = Field(..., ge=0.0, le=1.0, description="Fulfilled demand ratio over the last 30 simulated days.")

    # Risk classification
    stockout_risk: float = Field(..., ge=0.0, le=1.0, description="Stockout risk score 0.0–1.0.")
    alert_level: str = Field(..., description="One of: low | medium | high | critical.")
    reorder_suggested: bool = Field(..., description="True if stock is at or below reorder point.")

    model_config = {"json_schema_extra": {"example": {
        "stock_code": "85123A",
        "description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "as_of_date": "2011-12-09",
        "avg_daily_demand": 18.3,
        "predicted_next_demand": 20.1,
        "initial_stock_level": 549.0,
        "target_stock_level": 640.5,
        "simulated_stock_level": 549.0,
        "reorder_point": 128.1,
        "days_of_stock_remaining": 30,
        "pending_restock_quantity": 0.0,
        "next_restock_date": None,
        "last_restock_date": "2011-12-04",
        "stockout_days_last_30": 1,
        "projected_stockout_days": 0,
        "service_level_last_30": 0.987,
        "stockout_risk": 0.05,
        "alert_level": "low",
        "reorder_suggested": False,
    }}}
