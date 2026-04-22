"""Pydantic schemas for the analysis endpoint."""

from __future__ import annotations

from pydantic import BaseModel


class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int


class LabelValue(BaseModel):
    label: str
    value: float


class ScatterPoint(BaseModel):
    x: float
    y: float
    label: str | None = None


class UnivariateAnalysisResponse(BaseModel):
    quantity_histogram: list[HistogramBin]
    revenue_histogram: list[HistogramBin]
    price_histogram: list[HistogramBin]
    sales_by_day_of_week: list[LabelValue]   # avg daily quantity per weekday
    sales_by_month: list[LabelValue]         # avg daily quantity per month
    products_by_activity: list[LabelValue]   # distribution of active days count


class BivariateAnalysisResponse(BaseModel):
    revenue_by_month: list[LabelValue]            # total revenue per month (across all data)
    quantity_by_day_of_week: list[LabelValue]     # avg quantity per weekday
    revenue_by_quarter: list[LabelValue]          # total revenue per quarter
    price_vs_quantity_buckets: list[ScatterPoint] # avg demand per price bucket
    monthly_quantity_trend: list[LabelValue]      # total quantity by month-year
