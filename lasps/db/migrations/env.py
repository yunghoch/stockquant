from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from lasps.config.settings import settings
from lasps.db.base import Base
import lasps.db.models  # noqa: F401 — 모든 모델 등록

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# settings에서 DB URL 오버라이드
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """오프라인(SQL 생성) 모드."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """온라인(직접 실행) 모드."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # SQLite ALTER TABLE 지원
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
