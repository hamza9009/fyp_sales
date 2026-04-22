"""Remove MAPE from model_runs

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-21 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("model_runs", "mape")


def downgrade() -> None:
    op.add_column("model_runs", sa.Column("mape", sa.Float(), nullable=True))
