"""
Product catalogue model.
One row per unique StockCode in the UCI dataset.
"""

from datetime import datetime

from sqlalchemy import DateTime, Index, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Product(Base):
    """Represents a unique e-commerce product (StockCode)."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    unit_price: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_products_stock_code", "stock_code"),
    )

    def __repr__(self) -> str:
        return f"<Product stock_code={self.stock_code!r} description={self.description!r}>"
