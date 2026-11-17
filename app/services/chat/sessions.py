"""服务端对话历史。优先用 Redis，连不上则退回进程内内存。"""

from __future__ import annotations

import json
import threading
import time
import uuid
from abc import ABC, abstractmethod

from app.core.config import settings
from app.core.logging import event
from app.schemas.chat import ChatTurn

# 每个会话只保留最近这么多轮，避免历史无限变长后把模型上下文撑满。
_MAX_TURNS = 40


class SessionStore(ABC):
    """对话历史的存储接口。调用方只依赖这几个方法，不关心背后是 Redis 还是内存。"""

    @abstractmethod
    def create(self) -> str:
        """新建空会话，返回 session_id。"""
        raise NotImplementedError

    @abstractmethod
    def get(self, session_id: str) -> list[ChatTurn]:
        """读取会话历史。会话不存在或已过期时返回空列表。"""
        raise NotImplementedError

    @abstractmethod
    def save(self, session_id: str, history: list[ChatTurn]) -> None:
        """覆盖写入整段历史，并刷新过期时间。"""
        raise NotImplementedError


class MemorySessionStore(SessionStore):
    """进程内存储。Redis 不可用时的退路，重启进程后历史会丢。"""

    def __init__(self, ttl: int | None = None) -> None:
        # 过期秒数，默认跟配置里的 CHAT_SESSION_TTL 一致。
        self.ttl = ttl or settings.chat_session_ttl
        # 对话请求跑在线程池里，读写这份字典需要加锁。
        self._lock = threading.Lock()
        # session_id -> (过期时间戳, 历史记录)
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
        # 只留最近若干轮，并把过期时间往后推。
        rows = [turn.model_dump() for turn in history[-_MAX_TURNS:]]
        with self._lock:
            self._data[session_id] = (time.time() + self.ttl, rows)

    def _purge(self) -> None:
        """顺手清掉已过期的会话。内存实现没有 Redis 那种自动过期。"""
        now = time.time()
        expired = [key for key, (expires, _) in self._data.items() if expires < now]
        for key in expired:
            del self._data[key]


class RedisSessionStore(SessionStore):
    """Redis 存储。多进程、重启后会话还在。"""

    def __init__(self, client, ttl: int | None = None) -> None:
        self.client = client
        self.ttl = ttl or settings.chat_session_ttl
        # 键前缀，避免和别的业务抢同一个 Redis。
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
        # setex 会在 ttl 秒后自动删掉这条会话。
        self.client.setex(self.prefix + session_id, self.ttl, json.dumps(rows, ensure_ascii=False))


# 进程内只建一次存储。第一次对话时再决定用 Redis 还是内存。
_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """返回当前进程使用的会话存储。Redis 能 ping 通就用它，否则退回内存。"""
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
    """替换会话存储。测试时注入内存实现，避免依赖本机 Redis。"""
    global _store
    _store = store
