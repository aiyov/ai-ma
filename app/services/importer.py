import logging
from pathlib import Path

import httpx

from app.core.config import resolve_path, settings
from app.services.catalog import Product, load_products
from app.services.store import ProductStore

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "ai-ma-importer/1.0"}


def import_file(path: Path, store: ProductStore | None = None) -> int:
    products = load_products(path)
    image_paths = download_images(products)
    target = store or ProductStore()
    count = target.upsert(products, image_paths)
    logger.info("upserted %s products into %s", count, target.collection)
    return count


def download_images(products: list[Product]) -> dict[str, Path]:
    cache = resolve_path(settings.image_cache)
    cache.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    with httpx.Client(follow_redirects=True, timeout=40, headers=HEADERS) as client:
        for product in products:
            if not product.embed_image_url:
                logger.warning("sku %s has no image", product.sku)
                continue
            destination = cache / f"{product.sku}.jpg"
            try:
                _download(client, product.embed_image_url, destination)
            except httpx.HTTPError as exc:
                logger.warning("image download failed for %s: %s", product.sku, exc)
                continue
            saved[product.sku] = destination
    return saved


def _download(client: httpx.Client, url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(".part")
    with client.stream("GET", url) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type.startswith("text/") or content_type in {"application/json", "application/xml"}:
            raise httpx.HTTPError(f"unexpected content type {content_type}")
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    temporary.replace(destination)


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.data_path
    path = resolve_path(raw)
    store = ProductStore()
    count = import_file(path, store)
    print(f"upserted {count} products, collection count {store.count()}")
    print("\ntext search: grenadilla lip plate")
    for hit in store.search_text("intermediate flute with grenadilla wood lip plate", limit=2):
        print(f"  {hit.score:.4f}  {hit.brand}  {hit.sale_price}  {hit.title}")
    print("\nvisual text search: silver flute")
    for hit in store.search_visual_text("a silver concert flute", limit=2):
        print(f"  {hit.score:.4f}  {hit.brand}  {hit.title}")


if __name__ == "__main__":
    main()
