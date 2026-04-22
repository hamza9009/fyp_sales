"""
Database engine, session factory, and base model for SQLAlchemy.

The engine and session factory are created lazily on first use so that
importing this module never fails due to a missing DB driver or unreachable
database — useful during testing and CI.
"""

import logging
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Lazy engine / session factory
# ---------------------------------------------------------------------------
_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _get_engine() -> Engine:
    """Return the singleton engine, creating it on first call."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,   # validates connections before use
            pool_size=10,
            max_overflow=20,
            echo=settings.DEBUG,  # log SQL only in debug mode
        )

        @event.listens_for(_engine, "connect")
        def on_connect(dbapi_connection, connection_record):  # noqa: ARG001
            logger.debug("New database connection established.")

    return _engine


def _get_session_factory() -> sessionmaker:
    """Return the singleton session factory, creating it on first call."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_get_engine(),
        )
    return _SessionLocal


# Public alias so Alembic env.py can reference it directly
def get_engine() -> Engine:
    return _get_engine()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and ensures
    it is closed after the request completes.
    """
    db: Session = _get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """
    Verify the database is reachable.

    Returns:
        True if connection succeeds, False otherwise.
    """
    try:
        with _get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database connection check failed: %s", exc)
        return False
