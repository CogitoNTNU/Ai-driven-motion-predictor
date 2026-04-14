"""asyncpg connection pool creation."""

import asyncpg

from Kaare import config

_shared_pool: asyncpg.Pool | None = None


async def create_pool() -> asyncpg.Pool:
    """Create and return a fresh asyncpg connection pool.

    Returns:
        A connected asyncpg pool ready for use.

    Raises:
        asyncpg.PostgresConnectionError: If the database is unreachable.
    """
    return await asyncpg.create_pool(
        dsn=config.DB_DSN,
        min_size=1,
        max_size=10,
    )


async def get_shared_pool() -> asyncpg.Pool:
    """Return the process-wide singleton pool, creating it on first call.

    Subsequent calls return the same pool without reconnecting.
    """
    global _shared_pool
    if _shared_pool is None:
        _shared_pool = await create_pool()
    return _shared_pool


async def close_shared_pool() -> None:
    """Close and discard the singleton pool (called at shutdown)."""
    global _shared_pool
    if _shared_pool is not None:
        await _shared_pool.close()
        _shared_pool = None
