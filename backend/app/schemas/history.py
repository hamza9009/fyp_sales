"""Schemas for prediction history endpoints."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionHistoryItem(BaseModel):
    """One persisted forecast or inventory query."""

    id: int
    endpoint: str = Field(..., description="The endpoint type: forecast or inventory.")
    query_text: str = Field(..., description="Original user query text.")
    resolved_stock_code: str = Field(..., description="Canonical stock code used internally.")
    model_name: str | None = Field(None, description="Model name used for the prediction.")
    horizon_days: int | None = Field(None, description="Forecast horizon when applicable.")
    request_payload: dict[str, Any] | None = Field(None, description="Stored request metadata.")
    response_payload: dict[str, Any] = Field(..., description="Stored response payload.")
    created_at: datetime = Field(..., description="When the prediction was persisted.")

    model_config = ConfigDict(from_attributes=True)


class PredictionHistoryResponse(BaseModel):
    """Prediction history for a single anonymous client."""

    client_id: str | None = Field(None, description="Anonymous client identifier.")
    total: int = Field(..., ge=0, description="Number of items returned.")
    items: list[PredictionHistoryItem] = Field(..., description="Newest-first history rows.")
