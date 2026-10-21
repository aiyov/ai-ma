"""Run once or on an interval.

  .venv/bin/python -m app.services.sync
  .venv/bin/python -m app.services.sync --full
  .venv/bin/python -m app.services.sync --schedule 3600
  .venv/bin/python -m app.services.sync --touch site7sku1497780
"""

from __future__ import annotations

import argparse
import logging
import time

from app.core.logging import configure
from app.services.sync.job import run_sync, touch_override


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental product sync into Qdrant")
    parser.add_argument("--full", action="store_true", help="Ignore watermark; still skip unchanged hashes")
    parser.add_argument(
        "--schedule",
        type=int,
        metavar="SECONDS",
        help="Run forever every N seconds (simple timer; use cron in production if preferred)",
    )
    parser.add_argument(
        "--touch",
        metavar="SKU",
        help="Bump mock ATG updated_at for a SKU, then exit (use before a normal sync to demo change)",
    )
    args = parser.parse_args()
    configure()
    logging.getLogger("ai_ma").setLevel(logging.INFO)

    if args.touch:
        path = touch_override(args.touch)
        print(f"touched {args.touch} in {path}")
        return

    def once() -> None:
        result = run_sync(full=args.full)
        print(
            f"scanned={result.scanned} changed={result.changed} upserted={result.upserted} "
            f"skipped={result.skipped} chunks={result.chunks} count={result.count} "
            f"watermark={result.watermark}"
        )
        for failure in result.failed_images:
            print(f"image failed  {failure.sku}  {failure.reason}")

    if not args.schedule:
        once()
        return

    interval = max(args.schedule, 30)
    print(f"scheduling sync every {interval}s")
    while True:
        once()
        time.sleep(interval)


if __name__ == "__main__":
    main()
