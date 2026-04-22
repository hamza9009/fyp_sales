"""Pydantic response schemas for the dashboard endpoints."""

from datetime import date

from pydantic import BaseModel, Field


class DailySalesTrend(BaseModel):
    """Aggregated sales figures for a single day (all products)."""

    sale_date: date
    total_quantity: int
    total_revenue: float


class TopProduct(BaseModel):
    """A top-performing product by total revenue."""

    stock_code: str
    description: str | None
    total_revenue: float
    total_quantity: int
    num_days_active: int


class ModelSummary(BaseModel):
    """Snapshot of the best model's performance metrics."""

    model_name: str
    mae: float
    rmse: float
    train_rows: int
    test_rows: int
    cutoff_date: str


class DashboardSummaryResponse(BaseModel):
    """Full dashboard summary response."""

    # Dataset overview
    total_products: int = Field(..., description="Number of unique products in the dataset.")
    total_revenue: float = Field(..., description="Sum of all revenue across the entire dataset.")
    total_quantity: int = Field(..., description="Total units sold across the entire dataset.")
    date_range_start: date = Field(..., description="Earliest sale date in the dataset.")
    date_range_end: date = Field(..., description="Latest sale date in the dataset.")
    total_days: int = Field(..., description="Number of calendar days in the dataset.")

    # Trends
    last_30_days_trend: list[DailySalesTrend] = Field(
        ..., description="Daily revenue + quantity totals for the last 30 days of data."
    )

    # Top products
    top_products: list[TopProduct] = Field(
        ..., description="Top 10 products by total revenue."
    )

    # Model performance
    best_model: ModelSummary = Field(..., description="Best model metrics from Phase 3.")
