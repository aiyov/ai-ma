"""Incremental catalog sync: watermark filter + content hash skip."""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.logging import event, timed
from app.schemas.product import SyncResponse
from app.services.importer import import_products
from app.services.store import ProductStore
from app.services.sync.hashing import content_hash
from app.services.sync.source import ChangeSource, build_source
from app.services.sync.state import SyncState


def run_sync(
    *,
    full: bool = False,
    source: ChangeSource | None = None,
    state: SyncState | None = None,
    store: ProductStore | None = None,
) -> SyncResponse:
    catalog = source or build_source()
    sync_state = state or SyncState()
    target = store or ProductStore()
    watermark = None if full else sync_state.get_watermark()

    with timed("sync.run", full=full, watermark=watermark) as meta:
        candidates = catalog.fetch_all() if full else catalog.fetch_changed(watermark)
        to_upsert = []
        skipped = 0
        newest = watermark
        for item in candidates:
            digest = content_hash(item.product)
            if sync_state.get_hash(item.product.sku) == digest:
                skipped += 1
                if newest is None or item.source_updated_at > newest:
                    newest = item.source_updated_at
                continue
            to_upsert.append(item)
            if newest is None or item.source_updated_at > newest:
                newest = item.source_updated_at

        meta["scanned"] = len(candidates)
        meta["changed"] = len(to_upsert)
        meta["skipped"] = skipped

        if not to_upsert:
            if newest is not None and not full:
                sync_state.set_watermark(newest)
            event("sync.idle", scanned=len(candidates), skipped=skipped)
            return SyncResponse(
                scanned=len(candidates),
                changed=0,
                upserted=0,
                skipped=skipped,
                chunks=0,
                count=target.count(),
                watermark=None if newest is None else newest.isoformat(),
                full=full,
            )

        result = import_products([item.product for item in to_upsert], target)
        for item in to_upsert:
            sync_state.upsert_product(
                item.product.sku,
                content_hash(item.product),
                item.source_updated_at,
            )
        if newest is not None:
            sync_state.set_watermark(newest)

        event(
            "sync.done",
            scanned=len(candidates),
            changed=len(to_upsert),
            skipped=skipped,
            chunks=result.chunks,
        )
        return SyncResponse(
            scanned=len(candidates),
            changed=len(to_upsert),
            upserted=result.products,
            skipped=skipped,
            chunks=result.chunks,
            count=result.count,
            watermark=newest.isoformat() if newest else None,
            failed_images=result.failed_images,
            full=full,
        )


def touch_override(sku: str, when: datetime | None = None) -> Path:
    """Simulate an ATG edit by bumping a SKU's source_updated_at in the mock sidecar."""
    import json
    from pathlib import Path

    from app.core.config import resolve_path

    path = resolve_path(Path(".cache/sync_overrides.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    stamp = when or datetime.now(timezone.utc)
    payload[sku] = stamp.astimezone(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
