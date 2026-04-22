"""
Model run tracking model.
Records training metadata, evaluation metrics, and artifact paths for every
model training session. Enables reproducibility and model versioning.
"""

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import DateTime, Enum, Float, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ModelStatus(str, PyEnum):
    """Lifecycle states for a model training run."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _enum_values(enum_cls: type[PyEnum]) -> list[str]:
    """Persist enum values so the ORM matches the PostgreSQL enum labels."""
    return [str(member.value) for member in enum_cls]


class ModelRun(Base):
    """
    Tracks a single model training run including hyperparameters,
    evaluation metrics, and the path to the saved artifact.
    """

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Scope: None = global model (e.g. XGBoost across all SKUs),
    #        set = per-SKU model
    stock_code: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Evaluation metrics (populated after training completes)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Training window
    train_start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    train_end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Feature list stored as JSON array
    feature_names: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Filesystem path to the joblib/pickle artifact
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus, name="model_status_enum", values_callable=_enum_values),
        nullable=False,
        default=ModelStatus.RUNNING,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ModelRun id={self.id} model={self.model_name!r} "
            f"status={self.status} mae={self.mae} rmse={self.rmse}>"
        )
