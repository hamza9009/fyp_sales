"""
Model metrics router — Phase 4.

GET /models/metrics
  Returns MAE, RMSE, and ranking for all Phase 3 trained models.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.models import ModelMetricsResponse
from app.services import metrics_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    summary="Model evaluation metrics",
    description=(
        "Returns MAE, RMSE, training time, and ranking for all "
        "Phase 3 trained models (XGBoost, LightGBM, and their averaging ensemble). "
        "Also returns train/test split metadata."
    ),
)
def get_model_metrics() -> ModelMetricsResponse:
    """Return the Phase 3 model comparison report.

    Returns:
        :class:`~app.schemas.models.ModelMetricsResponse`.

    Raises:
        503 if the metrics report has not been generated.
    """
    try:
        payload = metrics_service.get_model_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ModelMetricsResponse(**payload)
