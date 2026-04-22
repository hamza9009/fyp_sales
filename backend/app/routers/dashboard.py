"""
Dashboard router — Phase 4.

GET /dashboard/summary
  Returns a high-level SaaS dashboard payload: revenue trends,
  top products, and model performance snapshot.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.dashboard import DashboardSummaryResponse
from app.services import dashboard_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get(
    "/summary",
    response_model=DashboardSummaryResponse,
    summary="Dashboard overview summary",
    description=(
        "Returns a complete dashboard payload including: total revenue, "
        "top 10 products, last-30-days sales trend, and best model metrics."
    ),
)
def get_dashboard_summary() -> DashboardSummaryResponse:
    """Aggregate and return the dashboard summary.

    Returns:
        :class:`~app.schemas.dashboard.DashboardSummaryResponse`.

    Raises:
        503 if the data store is not loaded.
    """
    try:
        payload = dashboard_service.get_dashboard_summary()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return DashboardSummaryResponse(**payload)
