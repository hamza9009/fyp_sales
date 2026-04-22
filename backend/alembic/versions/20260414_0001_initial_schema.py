"""Initial schema — create all tables

Revision ID: 0001
Revises:
Create Date: 2026-04-14 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # ENUM types (must be created before tables that reference them)
    # ------------------------------------------------------------------
    alert_level_enum = postgresql.ENUM(
        "low", "medium", "high", "critical", name="alert_level_enum", create_type=False
    )
    alert_level_enum.create(op.get_bind(), checkfirst=True)

    model_status_enum = postgresql.ENUM(
        "running", "completed", "failed", name="model_status_enum", create_type=False
    )
    model_status_enum.create(op.get_bind(), checkfirst=True)

    # ------------------------------------------------------------------
    # products
    # ------------------------------------------------------------------
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stock_code", sa.String(50), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
        sa.Column("unit_price", sa.Numeric(10, 4), nullable=True),
        sa.Column("country", sa.String(100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stock_code"),
    )
    op.create_index("ix_products_stock_code", "products", ["stock_code"])

    # ------------------------------------------------------------------
    # sales_daily
    # ------------------------------------------------------------------
    op.create_table(
        "sales_daily",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stock_code", sa.String(50), nullable=False),
        sa.Column("sale_date", sa.Date(), nullable=False),
        sa.Column("total_quantity", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_revenue", sa.Numeric(14, 4), nullable=False, server_default="0"),
        sa.Column("num_transactions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sales_daily_stock_code", "sales_daily", ["stock_code"])
    op.create_index("ix_sales_daily_sale_date", "sales_daily", ["sale_date"])
    op.create_index(
        "ix_sales_daily_stock_date",
        "sales_daily",
        ["stock_code", "sale_date"],
        unique=True,
    )

    # ------------------------------------------------------------------
    # model_runs (referenced by forecasts FK)
    # ------------------------------------------------------------------
    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("stock_code", sa.String(50), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("mape", sa.Float(), nullable=True),
        sa.Column("train_start_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("train_end_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("feature_names", sa.JSON(), nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=True),
        sa.Column(
            "status",
            model_status_enum,
            nullable=False,
            server_default="running",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # ------------------------------------------------------------------
    # forecasts
    # ------------------------------------------------------------------
    op.create_table(
        "forecasts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stock_code", sa.String(50), nullable=False),
        sa.Column("model_run_id", sa.Integer(), nullable=True),
        sa.Column("forecast_date", sa.Date(), nullable=False),
        sa.Column("predicted_quantity", sa.Float(), nullable=False),
        sa.Column("predicted_revenue", sa.Float(), nullable=True),
        sa.Column("confidence_lower", sa.Float(), nullable=True),
        sa.Column("confidence_upper", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["model_run_id"], ["model_runs.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # ------------------------------------------------------------------
    # inventory_signals
    # ------------------------------------------------------------------
    op.create_table(
        "inventory_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stock_code", sa.String(50), nullable=False),
        sa.Column("signal_date", sa.Date(), nullable=False),
        sa.Column("predicted_demand", sa.Float(), nullable=True),
        sa.Column("avg_daily_demand", sa.Float(), nullable=True),
        sa.Column("simulated_stock_level", sa.Float(), nullable=True),
        sa.Column("reorder_point", sa.Float(), nullable=True),
        sa.Column("days_of_stock_remaining", sa.Integer(), nullable=True),
        sa.Column("stockout_risk", sa.Float(), nullable=True),
        sa.Column(
            "alert_level",
            alert_level_enum,
            nullable=False,
            server_default="low",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # ------------------------------------------------------------------
    # api_logs
    # ------------------------------------------------------------------
    op.create_table(
        "api_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("endpoint", sa.String(255), nullable=False),
        sa.Column("method", sa.String(10), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("request_payload", sa.JSON(), nullable=True),
        sa.Column("response_time_ms", sa.Float(), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("api_logs")
    op.drop_table("inventory_signals")
    op.drop_table("forecasts")
    op.drop_table("model_runs")
    op.drop_table("sales_daily")
    op.drop_table("products")

    # Drop enums after tables
    postgresql.ENUM(name="alert_level_enum").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="model_status_enum").drop(op.get_bind(), checkfirst=True)
