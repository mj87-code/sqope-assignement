import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker


def _to_async_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _to_sync_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


# Engines and session factories are created once and reused for the process
# lifetime. Creating/disposing an engine per query (as the original code did)
# rebuilds the connection pool on every call — slow and connection-exhausting
# under concurrency.
_async_engine: AsyncEngine | None = None
_async_factory: async_sessionmaker[AsyncSession] | None = None
_sync_engine: Engine | None = None
_sync_factory: sessionmaker[Session] | None = None


def _get_async_factory() -> async_sessionmaker[AsyncSession]:
    global _async_engine, _async_factory
    if _async_factory is None:
        _async_engine = create_async_engine(
            _to_async_url(_database_url()), pool_pre_ping=True
        )
        _async_factory = async_sessionmaker(_async_engine, expire_on_commit=False)
    return _async_factory


def _get_sync_factory() -> sessionmaker[Session]:
    global _sync_engine, _sync_factory
    if _sync_factory is None:
        _sync_engine = create_engine(_to_sync_url(_database_url()), pool_pre_ping=True)
        _sync_factory = sessionmaker(_sync_engine)
    return _sync_factory


@asynccontextmanager
async def get_async_session() -> AsyncIterator[AsyncSession]:
    factory = _get_async_factory()
    async with factory() as session:
        yield session


@contextmanager
def get_sync_session() -> Iterator[Session]:
    factory = _get_sync_factory()
    with factory() as session:
        yield session


async def dispose_engines() -> None:
    """Dispose pooled engines — call on application shutdown."""
    global _async_engine, _async_factory, _sync_engine, _sync_factory
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_factory = None
    if _sync_engine is not None:
        _sync_engine.dispose()
        _sync_engine = None
        _sync_factory = None
