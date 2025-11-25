"""Schema and metadata helpers for Databricks table context."""
from __future__ import annotations

from typing import Dict, List

from core.databricks import _default_catalog_schema, _quote_identifier


def resolve_table_sequence(
    requested: List[str] | None,
    query: str,
    table_metadata: Dict[str, any] | None,
    max_tables: int,
    keywords: tuple[str, ...],
) -> List[str]:
    if not table_metadata:
        return []
    normalized_map = {name.lower(): name for name in table_metadata}

    def _dedup(names: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for name in names:
            actual = normalized_map.get(name.lower())
            if actual and actual not in seen:
                seen.add(actual)
                ordered.append(actual)
        return ordered

    if requested:
        return _dedup(requested)[:max_tables]

    normalized_query = (query or "").lower()
    hits = []
    for lowered, actual in normalized_map.items():
        index = normalized_query.find(lowered)
        if index != -1:
            hits.append((index, actual))
    if hits:
        hits.sort()
        return [name for _, name in hits][:max_tables]
    if requests_all_tables(normalized_query, keywords):
        return list(table_metadata.keys())[:max_tables]
    return []


def requests_all_tables(query: str, keywords: tuple[str, ...]) -> bool:
    normalized = (query or "").lower()
    return any(keyword in normalized for keyword in keywords)


def table_full_name(table_name: str, metadata: Dict[str, any] | None) -> str:
    full_name = (metadata or {}).get("full_name")
    if full_name:
        return full_name
    try:
        catalog, schema = _default_catalog_schema()
        return ".".join(
            [_quote_identifier(part) for part in (catalog, schema, table_name)]
        )
    except Exception:
        return table_name


def format_schema_context(table_metadata: Dict[str, any] | None) -> str:
    """Build a short schema section listing full table identifiers."""
    if not table_metadata:
        return ""
    lines = ["Schema context (use identifiers exactly as provided):"]
    for table_name, metadata in table_metadata.items():
        full_name = (metadata or {}).get("full_name")
        columns = (metadata or {}).get("columns") or []
        if not full_name:
            continue
        column_text = ""
        if columns:
            col_parts = []
            for col in columns:
                name = col.get("name") or ""
                col_type = col.get("type") or ""
                if name:
                    col_parts.append(f"{name} ({col_type})")
            if col_parts:
                column_text = " | columns: " + ", ".join(col_parts)
        lines.append(f"- {table_name}: {full_name}{column_text}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def select_relevant_metadata(
    user_query: str, table_metadata: Dict[str, any] | None, max_tables: int, keywords: tuple[str, ...]
) -> Dict[str, any]:
    """Return only the table metadata referenced in the user query."""
    if not table_metadata:
        return {}
    tables = resolve_table_sequence(None, user_query, table_metadata, max_tables, keywords)
    if not tables:
        return {}
    return {name: table_metadata[name] for name in tables if name in table_metadata}
