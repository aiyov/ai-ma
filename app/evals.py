"""Retrieval checks for the demo catalog. Run: python -m app.evals"""

from dataclasses import dataclass

from app.services.store import ProductStore


@dataclass(frozen=True)
class Case:
    query: str
    brand: str
    contains: str


CASES = [
    Case("grenadilla wood lip plate", "Tomasi", "grenadilla"),
    Case("plush-lined flute case", "Tomasi", "plush"),
    Case("Offset G B-Foot sterling silver headjoint", "Giardinelli", "Offset G"),
    Case("哪支长笛有 grenadilla 唇板", "Tomasi", "grenadilla"),
]


def run(store: ProductStore | None = None) -> list[str]:
    target = store or ProductStore()
    failures: list[str] = []
    for case in CASES:
        hits = target.search_text(case.query, limit=2)
        if not hits:
            failures.append(f"{case.query!r}: no hits")
            continue
        top = hits[0]
        if top.brand != case.brand:
            failures.append(f"{case.query!r}: expected {case.brand}, got {top.brand}")
        if case.contains.casefold() not in top.document.casefold():
            failures.append(f"{case.query!r}: chunk missing {case.contains!r}")
    return failures


def main() -> None:
    failures = run()
    if failures:
        for item in failures:
            print("FAIL", item)
        raise SystemExit(1)
    print(f"ok {len(CASES)} queries")


if __name__ == "__main__":
    main()
