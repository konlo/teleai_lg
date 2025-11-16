"""Utility script to print tables available in the configured Databricks catalog/schema."""
from __future__ import annotations

from core.databricks import list_tables


def main() -> None:
    try:
        tables = list_tables()
    except Exception as exc:  # pragma: no cover - manual script
        raise SystemExit(f"Failed to fetch tables: {exc}") from exc

    if not tables:
        print("No tables found.")
        return

    print("Tables:")
    for name in tables:
        print(f"- {name}")


if __name__ == "__main__":
    main()
