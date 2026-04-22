"""
Alembic environment configuration.

Reads DATABASE_URL from environment (via app.config) and imports all
SQLAlchemy models so that autogenerate can detect schema changes.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Ensure the backend/ directory is on sys.path so 'app' is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings
from app.database import Base, get_engine  # noqa: F401 — Base must be imported before models

# Import all models so Alembic autogenerate can detect them
import app.models  # noqa: F401

# ---------------------------------------------------------------------------
# Alembic Config object (gives access to alembic.ini)
# ---------------------------------------------------------------------------
config = context.config

# Set the DB URL programmatically from our settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpret the config file for Python logging if present
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata for autogenerate support
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline migration (no live DB connection)
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    Generates SQL script without connecting to the database.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migration (live DB connection)
# ---------------------------------------------------------------------------
def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    Creates a real connection and applies migrations immediately.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
