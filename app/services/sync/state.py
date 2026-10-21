"""Local sync watermark and per-SKU content hashes."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import resolve_path, settings


class SyncState:
    def __init__(self, path: Path | None = None) -> None:
        self.path = resolve_path(path or settings.sync_state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sync_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS product_sync (
                    sku TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    source_updated_at TEXT,
                    indexed_at TEXT NOT NULL
                );
                """
            )

    def get_watermark(self) -> datetime | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM sync_meta WHERE key = 'watermark'"
            ).fetchone()
        if row is None:
            return None
        return datetime.fromisoformat(row["value"])

    def set_watermark(self, value: datetime) -> None:
        stamp = _isoformat(value)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sync_meta(key, value) VALUES('watermark', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (stamp,),
            )

    def get_hash(self, sku: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content_hash FROM product_sync WHERE sku = ?",
                (sku,),
            ).fetchone()
        return None if row is None else str(row["content_hash"])

    def upsert_product(self, sku: str, content_hash: str, source_updated_at: datetime | None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO product_sync(sku, content_hash, source_updated_at, indexed_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(sku) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    source_updated_at = excluded.source_updated_at,
                    indexed_at = excluded.indexed_at
                """,
                (
                    sku,
                    content_hash,
                    None if source_updated_at is None else _isoformat(source_updated_at),
                    _isoformat(datetime.now(timezone.utc)),
                ),
            )


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()
