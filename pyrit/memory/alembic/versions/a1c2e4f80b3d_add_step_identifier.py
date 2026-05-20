# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
add step_identifier to AttackResultEntries.

Revision ID: a1c2e4f80b3d
Revises: 7a1b2c3d4e5f
Create Date: 2026-05-20 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1c2e4f80b3d"
down_revision: str | None = "7a1b2c3d4e5f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this schema upgrade."""
    # Additive nullable column: scenarios that opt into StrategyGraph populate
    # this with the composite ScenarioStep identifier; legacy and direct-attack
    # rows leave it null. No backfill needed.
    op.add_column(
        "AttackResultEntries",
        sa.Column("step_identifier", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    """Revert this schema upgrade."""
    op.drop_column("AttackResultEntries", "step_identifier")
