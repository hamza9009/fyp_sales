"""
API request/response logging model.
Captures every inbound API call for latency tracking and audit.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ApiLog(Base):
    """
    Stores one row per API request.
    Used for latency benchmarking and load testing analysis.
    """

    __tablename__ = "api_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)

    # Optional: store request body/params for debugging
    request_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Latency in milliseconds
    response_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Optional error detail
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<ApiLog endpoint={self.endpoint!r} "
            f"status={self.status_code} latency={self.response_time_ms}ms>"
        )
