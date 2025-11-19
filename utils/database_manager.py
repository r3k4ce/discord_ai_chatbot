from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Sequence

import aiosqlite


class DatabaseManager:
    """Thin async wrapper around SQLite for user settings and chat history."""

    def __init__(self, db_path: str | Path, history_limit: int = 20) -> None:
        self._db_path = str(db_path)
        self._history_limit = history_limit
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Open the database connection and create tables if needed."""
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(self._db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL
                -- web_search flag toggles using web search tool
            )
            """
        )
        # Ensure web_search column exists (backwards-compatible ALTER TABLE)
        cursor = await self._conn.execute("PRAGMA table_info(user_settings);")
        cols = await cursor.fetchall()
        await cursor.close()
        col_names = {row[1] for row in cols}
        if "web_search" not in col_names:
            try:
                await self._conn.execute(
                    "ALTER TABLE user_settings ADD COLUMN web_search INTEGER NOT NULL DEFAULT 0"
                )
                await self._conn.commit()
            except Exception:
                # If migration fails for older SQLite versions or other reasons,
                # ignore and continue; absence of column will default to false via code paths.
                pass
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def _connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            await self.initialize()
        assert self._conn is not None
        return self._conn

    async def set_user_model(self, user_id: int, provider: str, model: str) -> None:
        conn = await self._connection()
        async with self._lock:
            await conn.execute(
                """
                INSERT INTO user_settings (user_id, provider, model)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id)
                DO UPDATE SET provider=excluded.provider, model=excluded.model
                """,
                (user_id, provider, model),
            )
            await conn.commit()

    async def get_user_model(self, user_id: int) -> tuple[str, str] | None:
        conn = await self._connection()
        async with self._lock:
            cursor = await conn.execute(
                "SELECT provider, model FROM user_settings WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return None
        return row[0], row[1]

    async def set_user_web_search(self, user_id: int, enabled: bool) -> None:
        conn = await self._connection()
        async with self._lock:
            await conn.execute(
                "UPDATE user_settings SET web_search = ? WHERE user_id = ?",
                (1 if enabled else 0, user_id),
            )
            await conn.commit()

    async def get_user_web_search(self, user_id: int) -> bool:
        conn = await self._connection()
        async with self._lock:
            cursor = await conn.execute(
                "SELECT web_search FROM user_settings WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return False
        return bool(row[0])

    async def append_history(self, user_id: int, role: str, content: str) -> None:
        conn = await self._connection()
        async with self._lock:
            await conn.execute(
                "INSERT INTO conversation_history (user_id, role, content) VALUES (?, ?, ?)",
                (user_id, role, content),
            )
            await conn.execute(
                """
                DELETE FROM conversation_history
                WHERE user_id = ?
                AND id NOT IN (
                    SELECT id FROM conversation_history
                    WHERE user_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (user_id, user_id, self._history_limit),
            )
            await conn.commit()

    async def get_history(self, user_id: int) -> list[dict[str, str]]:
        conn = await self._connection()
        async with self._lock:
            cursor = await conn.execute(
                """
                SELECT role, content FROM conversation_history
                WHERE user_id = ?
                ORDER BY id ASC
                """,
                (user_id,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [
            {"role": role, "content": content}
            for role, content in rows
        ]

    async def clear_history(self, user_id: int) -> None:
        conn = await self._connection()
        async with self._lock:
            await conn.execute(
                "DELETE FROM conversation_history WHERE user_id = ?",
                (user_id,),
            )
            await conn.commit()

    async def list_users(self) -> Sequence[int]:
        conn = await self._connection()
        async with self._lock:
            cursor = await conn.execute("SELECT DISTINCT user_id FROM conversation_history")
            rows = await cursor.fetchall()
            await cursor.close()
        return [row[0] for row in rows]
