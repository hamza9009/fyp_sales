"""Prediction request history model."""

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PredictionHistory(Base):
    """Stores successful forecast and inventory queries per client."""

    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    client_id: Mapped[str] = mapped_column(String(120), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(20), nullable=False)
    query_text: Mapped[str] = mapped_column(String(255), nullable=False)
    resolved_stock_code: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    horizon_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    response_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_prediction_history_client_created", "client_id", "created_at"),
        Index("ix_prediction_history_stock_code", "resolved_stock_code"),
    )

    def __repr__(self) -> str:
        return (
            f"<PredictionHistory id={self.id} endpoint={self.endpoint!r} "
            f"stock_code={self.resolved_stock_code!r}>"
        )
