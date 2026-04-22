"""Add prediction history table

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-22 00:10:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "prediction_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("client_id", sa.String(length=120), nullable=False),
        sa.Column("endpoint", sa.String(length=20), nullable=False),
        sa.Column("query_text", sa.String(length=255), nullable=False),
        sa.Column("resolved_stock_code", sa.String(length=50), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=True),
        sa.Column("horizon_days", sa.Integer(), nullable=True),
        sa.Column("request_payload", sa.JSON(), nullable=True),
        sa.Column("response_payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_prediction_history_client_created",
        "prediction_history",
        ["client_id", "created_at"],
    )
    op.create_index(
        "ix_prediction_history_stock_code",
        "prediction_history",
        ["resolved_stock_code"],
    )


def downgrade() -> None:
    op.drop_index("ix_prediction_history_stock_code", table_name="prediction_history")
    op.drop_index("ix_prediction_history_client_created", table_name="prediction_history")
    op.drop_table("prediction_history")
