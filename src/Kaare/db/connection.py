"""asyncpg connection pool creation."""

import asyncpg

from Kaare import config


async def create_pool() -> asyncpg.Pool:
    """Create and return an asyncpg connection pool.

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
