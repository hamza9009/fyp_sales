"""
Inventory intelligence signals model.
Stores stockout risk, reorder suggestions, and alert levels per product per day.
Inventory levels are simulated from demand estimates since the UCI dataset
does not contain explicit stock-on-hand data.
"""

from datetime import date, datetime
from enum import Enum as PyEnum

from sqlalchemy import Date, DateTime, Enum, Float, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AlertLevel(str, PyEnum):
    """Risk alert levels for inventory signals."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def _enum_values(enum_cls: type[PyEnum]) -> list[str]:
    """Persist enum values so the ORM matches the PostgreSQL enum labels."""
    return [str(member.value) for member in enum_cls]


class InventorySignal(Base):
    """
    Per-product daily inventory intelligence signal.

    NOTE: Since the UCI dataset has no stock-on-hand column,
    simulated_stock_level is derived from a configurable starting
    stock assumption minus cumulative demand. This is documented
    explicitly to avoid silent assumptions.
    """

    __tablename__ = "inventory_signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(String(50), nullable=False)
    signal_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Demand-based inputs
    predicted_demand: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_daily_demand: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Inventory intelligence (simulated)
    initial_stock_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_stock_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    simulated_stock_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    reorder_point: Mapped[float | None] = mapped_column(Float, nullable=True)
    days_of_stock_remaining: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pending_restock_quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    stockout_days_last_30: Mapped[int | None] = mapped_column(Integer, nullable=True)
    projected_stockout_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    service_level_last_30: Mapped[float | None] = mapped_column(Float, nullable=True)
    next_restock_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    last_restock_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Risk classification
    stockout_risk: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0.0–1.0
    alert_level: Mapped[AlertLevel] = mapped_column(
        Enum(AlertLevel, name="alert_level_enum", values_callable=_enum_values),
        nullable=False,
        default=AlertLevel.LOW,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<InventorySignal stock_code={self.stock_code!r} "
            f"date={self.signal_date} alert={self.alert_level}>"
        )
