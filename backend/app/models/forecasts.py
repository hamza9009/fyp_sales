"""
Forecast predictions model.
Stores per-product daily demand predictions from any trained model.
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Forecast(Base):
    """
    Stores model forecast output for a specific product and date.
    Linked to a ModelRun so predictions are traceable to the model version.
    """

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(String(50), nullable=False)
    model_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("model_runs.id", ondelete="SET NULL"), nullable=True
    )
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Prediction outputs
    predicted_quantity: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_revenue: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Optional confidence bounds (for models that support it)
    confidence_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_upper: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<Forecast stock_code={self.stock_code!r} "
            f"date={self.forecast_date} qty={self.predicted_quantity:.2f}>"
        )
