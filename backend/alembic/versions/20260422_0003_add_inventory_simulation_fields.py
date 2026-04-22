"""Add richer inventory simulation fields

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-22 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("inventory_signals", sa.Column("initial_stock_level", sa.Float(), nullable=True))
    op.add_column("inventory_signals", sa.Column("target_stock_level", sa.Float(), nullable=True))
    op.add_column("inventory_signals", sa.Column("pending_restock_quantity", sa.Float(), nullable=True))
    op.add_column("inventory_signals", sa.Column("stockout_days_last_30", sa.Integer(), nullable=True))
    op.add_column("inventory_signals", sa.Column("projected_stockout_days", sa.Integer(), nullable=True))
    op.add_column("inventory_signals", sa.Column("service_level_last_30", sa.Float(), nullable=True))
    op.add_column("inventory_signals", sa.Column("next_restock_date", sa.Date(), nullable=True))
    op.add_column("inventory_signals", sa.Column("last_restock_date", sa.Date(), nullable=True))


def downgrade() -> None:
    op.drop_column("inventory_signals", "last_restock_date")
    op.drop_column("inventory_signals", "next_restock_date")
    op.drop_column("inventory_signals", "service_level_last_30")
    op.drop_column("inventory_signals", "projected_stockout_days")
    op.drop_column("inventory_signals", "stockout_days_last_30")
    op.drop_column("inventory_signals", "pending_restock_quantity")
    op.drop_column("inventory_signals", "target_stock_level")
    op.drop_column("inventory_signals", "initial_stock_level")
