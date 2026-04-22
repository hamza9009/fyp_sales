"""Prediction history router."""

from fastapi import APIRouter, Depends, Header, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.prediction_history import PredictionHistory
from app.schemas.history import PredictionHistoryResponse

router = APIRouter(prefix="/history", tags=["History"])


@router.get(
    "",
    response_model=PredictionHistoryResponse,
    summary="Recent prediction history for the current client",
    description=(
        "Returns recent forecast and inventory queries persisted for the "
        "current anonymous frontend client."
    ),
)
def get_prediction_history(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of history rows."),
    client_id: str | None = Header(default=None, alias="X-Client-Id"),
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    if not client_id:
        return PredictionHistoryResponse(client_id=None, total=0, items=[])

    items = db.execute(
        select(PredictionHistory)
        .where(PredictionHistory.client_id == client_id)
        .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
        .limit(limit)
    ).scalars().all()

    return PredictionHistoryResponse(client_id=client_id, total=len(items), items=items)
