"""Database connection and session management."""

from typing import AsyncGenerator

import asyncpg
from asyncpg import Pool

from app.config import get_settings


class Database:
    """Database connection manager using asyncpg."""

    def __init__(self) -> None:
        self._pool: Pool | None = None

    async def connect(self) -> None:
        """Create database connection pool."""
        settings = get_settings()
        self._pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=5,
            max_size=20,
        )
        print("✅ Database connection pool created")

        # Enable pgvector extension
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("✅ pgvector extension enabled")

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            print("✅ Database connection pool closed")

    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a connection from the pool."""
        if not self._pool:
            raise RuntimeError("Database not connected")
        async with self._pool.acquire() as connection:
            yield connection

    @property
    def pool(self) -> Pool:
        """Get the connection pool."""
        if not self._pool:
            raise RuntimeError("Database not connected")
        return self._pool


# Global database instance
db = Database()


async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """Dependency for getting database connection."""
    async for conn in db.get_connection():
        yield conn
