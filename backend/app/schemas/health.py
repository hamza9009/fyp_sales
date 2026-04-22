"""
Pydantic response schemas for the health endpoint.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Standard health check response."""

    status: str
    version: str
    database: str

    model_config = {"json_schema_extra": {"example": {
        "status": "ok",
        "version": "1.0.0",
        "database": "connected",
    }}}
