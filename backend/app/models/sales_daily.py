"""
Daily aggregated sales model.
One row per (stock_code, date) pair after ETL aggregation.
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Index, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SalesDaily(Base):
    """
    Stores daily aggregated sales metrics per product.
    Negative quantities (returns) are excluded during ETL.
    """

    __tablename__ = "sales_daily"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(String(50), nullable=False)
    sale_date: Mapped[date] = mapped_column(Date, nullable=False)
    total_quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_revenue: Mapped[float] = mapped_column(Numeric(14, 4), nullable=False, default=0.0)
    num_transactions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_sales_daily_stock_code", "stock_code"),
        Index("ix_sales_daily_sale_date", "sale_date"),
        Index("ix_sales_daily_stock_date", "stock_code", "sale_date", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<SalesDaily stock_code={self.stock_code!r} "
            f"date={self.sale_date} qty={self.total_quantity}>"
        )
