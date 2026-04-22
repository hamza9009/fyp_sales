"""
Health check router.
Provides a lightweight endpoint to confirm the API and DB are operational.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.database import check_db_connection
from app.schemas.health import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

settings = get_settings()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Returns API status and database connectivity.",
)
def health_check() -> HealthResponse:
    """
    Confirm the API server is running and the database is reachable.

    Returns:
        HealthResponse with status, version, and DB connectivity.

    Raises:
        HTTPException 503 if the database is unreachable.
    """
    db_ok = check_db_connection()

    if not db_ok:
        logger.error("Health check failed: database unreachable.")
        raise HTTPException(
            status_code=503,
            detail="Database connection failed. Check DATABASE_URL and PostgreSQL status.",
        )

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        database="connected",
    )
