"""Add system_prompt_ai column to trees table

Revision ID: 9c3e7f1a2b5d
Revises: 2082445b0769
Create Date: 2026-03-22 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector


# revision identifiers, used by Alembic.
revision = "9c3e7f1a2b5d"
down_revision = "2082445b0769"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    columns = [col["name"] for col in inspector.get_columns("trees")]
    if "system_prompt_ai" not in columns:
        op.add_column(
            "trees",
            sa.Column("system_prompt_ai", sa.Text(), nullable=True),
        )


def downgrade():
    op.drop_column("trees", "system_prompt_ai")
