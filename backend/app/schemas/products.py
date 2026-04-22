"""Pydantic schemas for the products search endpoint — Phase 4 (optimised)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ProductResult(BaseModel):
    stock_code: str
    description: Optional[str] = None
    unit_price: Optional[float] = None


class ProductSearchResponse(BaseModel):
    query: str
    results: list[ProductResult]
    total: int = Field(..., description="Number of results returned")
