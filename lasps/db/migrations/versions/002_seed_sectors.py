"""Seed 20 sectors

Revision ID: 002
Revises: 001
Create Date: 2026-02-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SECTORS = [
    (0, "001", "전기전자"),
    (1, "002", "금융업"),
    (2, "003", "서비스업"),
    (3, "004", "의약품"),
    (4, "005", "운수창고"),
    (5, "006", "유통업"),
    (6, "007", "건설업"),
    (7, "008", "철강금속"),
    (8, "009", "기계"),
    (9, "010", "화학"),
    (10, "011", "섬유의복"),
    (11, "012", "음식료품"),
    (12, "013", "비금속광물"),
    (13, "014", "종이목재"),
    (14, "015", "운수장비"),
    (15, "016", "통신업"),
    (16, "017", "전기가스업"),
    (17, "018", "제조업(기타)"),
    (18, "019", "농업임업어업"),
    (19, "020", "광업"),
]


def upgrade() -> None:
    sectors_table = sa.table(
        "sectors",
        sa.column("id", sa.SmallInteger),
        sa.column("code", sa.String),
        sa.column("name", sa.String),
    )
    op.bulk_insert(
        sectors_table,
        [{"id": sid, "code": code, "name": name} for sid, code, name in SECTORS],
    )


def downgrade() -> None:
    op.execute("DELETE FROM sectors")
