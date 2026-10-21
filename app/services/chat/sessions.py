"""Server-side chat history. Redis when available, memory otherwise."""

from __future__ import annotations

import json
import threading
import time
import uuid
from abc import ABC, abstractmethod

from app.core.config import settings
from app.core.logging import event
from app.schemas.chat import ChatTurn

_MAX_TURNS = 40


class SessionStore(ABC):
    @abstractmethod
    def create(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get(self, session_id: str) -> list[ChatTurn]:
        raise NotImplementedError

    @abstractmethod
    def save(self, session_id: str, history: list[ChatTurn]) -> None:
        raise NotImplementedError


class MemorySessionStore(SessionStore):
    def __init__(self, ttl: int | None = None) -> None:
        self.ttl = ttl or settings.chat_session_ttl
        self._lock = threading.Lock()
        self._data: dict[str, tuple[float, list[dict]]] = {}

    def create(self) -> str:
        session_id = uuid.uuid4().hex
        self.save(session_id, [])
        return session_id

    def get(self, session_id: str) -> list[ChatTurn]:
        with self._lock:
            self._purge()
            entry = self._data.get(session_id)
            if entry is None:
                return []
            _, rows = entry
            return [ChatTurn.model_validate(row) for row in rows]

    def save(self, session_id: str, history: list[ChatTurn]) -> None:
        rows = [turn.model_dump() for turn in history[-_MAX_TURNS:]]
        with self._lock:
            self._data[session_id] = (time.time() + self.ttl, rows)

    def _purge(self) -> None:
        now = time.time()
        expired = [key for key, (expires, _) in self._data.items() if expires < now]
        for key in expired:
            del self._data[key]


class RedisSessionStore(SessionStore):
    def __init__(self, client, ttl: int | None = None) -> None:
        self.client = client
        self.ttl = ttl or settings.chat_session_ttl
        self.prefix = "ai-ma:chat:"

    def create(self) -> str:
        session_id = uuid.uuid4().hex
        self.save(session_id, [])
        return session_id

    def get(self, session_id: str) -> list[ChatTurn]:
        raw = self.client.get(self.prefix + session_id)
        if not raw:
            return []
        rows = json.loads(raw)
        return [ChatTurn.model_validate(row) for row in rows]

    def save(self, session_id: str, history: list[ChatTurn]) -> None:
        rows = [turn.model_dump() for turn in history[-_MAX_TURNS:]]
        self.client.setex(self.prefix + session_id, self.ttl, json.dumps(rows, ensure_ascii=False))


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is not None:
        return _store
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=1)
        client.ping()
        _store = RedisSessionStore(client)
        event("chat.sessions", backend="redis")
    except Exception as exc:
        _store = MemorySessionStore()
        event("chat.sessions", backend="memory", reason=exc.__class__.__name__)
    return _store


def set_session_store(store: SessionStore | None) -> None:
    global _store
    _store = store
