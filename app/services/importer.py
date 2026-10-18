import logging
from pathlib import Path

import httpx

from app.core.config import resolve_path, settings
from app.schemas.product import ImageFailure, ImportResponse
from app.services.catalog import Product, load_products
from app.services.store import ProductStore

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "ai-ma-importer/1.0"}


def import_file(path: Path, store: ProductStore | None = None) -> ImportResponse:
    products = load_products(path)
    image_paths, failures = download_images(products)
    target = store or ProductStore()
    chunks = target.upsert(products, image_paths)
    logger.info(
        "upserted %s chunks for %s products into %s, image failures %s",
        chunks,
        len(products),
        target.collection,
        len(failures),
    )
    return ImportResponse(
        products=len(products),
        chunks=chunks,
        count=target.count(),
        failed_images=failures,
    )


def download_images(products: list[Product]) -> tuple[dict[str, Path], list[ImageFailure]]:
    cache = resolve_path(settings.image_cache)
    cache.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    failures: list[ImageFailure] = []
    with httpx.Client(follow_redirects=True, timeout=40, headers=HEADERS) as client:
        for product in products:
            if not product.embed_image_url:
                failures.append(ImageFailure(sku=product.sku, reason="missing image url"))
                continue
            destination = cache / f"{product.sku}.jpg"
            try:
                _download(client, product.embed_image_url, destination)
            except httpx.HTTPError as exc:
                logger.warning("image download failed for %s: %s", product.sku, exc)
                failures.append(
                    ImageFailure(sku=product.sku, url=product.embed_image_url, reason=str(exc))
                )
                continue
            saved[product.sku] = destination
    return saved, failures


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
    result = import_file(path, ProductStore())
    print(
        f"upserted {result.chunks} chunks for {result.products} products, "
        f"collection has {result.count}"
    )
    for failure in result.failed_images:
        print(f"image failed  {failure.sku}  {failure.reason}")


if __name__ == "__main__":
    main()
