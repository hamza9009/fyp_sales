"""Pydantic schemas for the dataset upload / pipeline endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

PipelineStatus = Literal[
    "idle", "etl_running", "ml_running", "reloading", "completed", "failed"
]


class PipelineJobResponse(BaseModel):
    job_id: str
    status: PipelineStatus
    progress: int  # 0-100
    message: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    etl_rows: Optional[int] = None
    etl_products: Optional[int] = None
    best_model: Optional[str] = None
    best_rmse: Optional[float] = None


class RequiredField(BaseModel):
    internal_name: str
    label: str
    desc: str
    required: bool  # True = must be mapped; False = pipeline runs without it


class ColumnInspectResponse(BaseModel):
    """Response from POST /pipeline/inspect — headers + auto-suggested mapping."""

    columns: list[str]
    suggested_mapping: dict[str, str]  # internal_snake_case → user_column
    required_fields: list[RequiredField]
