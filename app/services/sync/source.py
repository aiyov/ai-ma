"""Change sources. MockFileSource stands in for an ATG export until a real feed exists."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import resolve_path, settings
from app.services.catalog import Product, load_products


@dataclass(frozen=True)
class ChangedProduct:
    product: Product
    source_updated_at: datetime


class ChangeSource(ABC):
    """Contract an ATG-facing adapter should implement later."""

    @abstractmethod
    def fetch_changed(self, since: datetime | None) -> list[ChangedProduct]:
        """Return products updated after `since`. None means full catalog."""

    @abstractmethod
    def fetch_all(self) -> list[ChangedProduct]:
        raise NotImplementedError


class MockFileSource(ChangeSource):
    """Local JSON catalog used as a stand-in for ATG.

    Each product's source_updated_at defaults to the file mtime. Optional sidecar
    `.cache/sync_overrides.json` can set per-SKU ISO timestamps to simulate ATG
    edits without rewriting data.json:

        {"site7sku1497780": "2026-10-21T12:00:00+00:00"}
    """

    def __init__(self, path: Path | None = None, overrides_path: Path | None = None) -> None:
        self.path = resolve_path(path or settings.data_path)
        self.overrides_path = overrides_path or resolve_path(Path(".cache/sync_overrides.json"))

    def fetch_all(self) -> list[ChangedProduct]:
        return self.fetch_changed(since=None)

    def fetch_changed(self, since: datetime | None) -> list[ChangedProduct]:
        products = load_products(self.path)
        file_mtime = datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)
        overrides = _load_overrides(self.overrides_path)
        changed: list[ChangedProduct] = []
        for product in products:
            updated = overrides.get(product.sku, file_mtime)
            if since is not None and updated <= since:
                continue
            changed.append(ChangedProduct(product=product, source_updated_at=updated))
        return changed


def _load_overrides(path: Path) -> dict[str, datetime]:
    if not path.is_file():
        return {}
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, datetime] = {}
    for sku, value in raw.items():
        try:
            parsed[str(sku)] = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
    return parsed


def build_source(name: str | None = None) -> ChangeSource:
    kind = (name or settings.sync_source).strip().lower()
    if kind in {"file", "mock", "json"}:
        return MockFileSource()
    raise ValueError(f"unknown sync source: {kind!r}. Implement an ATG adapter or use 'file'.")
