"""Databricks helpers reused across scripts and apps."""
from __future__ import annotations

import os
import re
from contextlib import closing
from typing import Any, Dict, List, Sequence, Tuple

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv():
        return None

load_dotenv()


def _get_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _connection_args() -> Tuple[str, str, str]:
    host = _get_required("DATABRICKS_HOST")
    token = _get_required("DATABRICKS_TOKEN")
    http_path = _get_required("DATABRICKS_HTTP_PATH")
    return host, token, http_path


def _default_catalog_schema(
    catalog: str | None = None, schema: str | None = None
) -> Tuple[str, str]:
    db_catalog = catalog or os.environ.get("DATABRICKS_CATALOG", "workspace")
    db_schema = schema or os.environ.get("DATABRICKS_SCHEMA", "default")
    return db_catalog, db_schema


def _quote_identifier(name: str) -> str:
    return f"`{name}`"


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(label: str, value: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"{label} must match {_IDENTIFIER_RE.pattern}, got {value!r}")
    return value


def _execute_query(statement: str) -> Tuple[Sequence[str], List[Sequence[Any]]]:
    from databricks import sql

    host, token, http_path = _connection_args()
    with closing(
        sql.connect(
            server_hostname=host,
            http_path=http_path,
            access_token=token,
        )
    ) as connection, closing(connection.cursor()) as cursor:
        cursor.execute(statement)
        rows = cursor.fetchall()
        columns: Sequence[str] = []
        if cursor.description:
            columns = [description[0] for description in cursor.description]
    return columns, rows


def list_tables(
    catalog: str | None = None,
    schema: str | None = None,
) -> List[str]:
    """Return table names for the given catalog + schema."""
    db_catalog, db_schema = _default_catalog_schema(catalog, schema)
    statement = f"SHOW TABLES IN {db_catalog}.{db_schema}"
    _columns, rows = _execute_query(statement)
    return [row.tableName for row in rows]


def fetch_table_preview(
    table: str,
    *,
    catalog: str | None = None,
    schema: str | None = None,
    limit: int = 50,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Return the SQL preview statement and rows for the requested table."""
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    safe_table = _validate_identifier("table", table)

    db_catalog, db_schema = _default_catalog_schema(catalog, schema)
    statement = (
        f"SELECT * FROM "
        f"{_quote_identifier(db_catalog)}."
        f"{_quote_identifier(db_schema)}."
        f"{_quote_identifier(safe_table)} "
        f"LIMIT {limit}"
    )
    columns, rows = _execute_query(statement)
    data = [dict(zip(columns, row)) for row in rows]
    return statement, data


def fetch_table_metadata(
    table: str,
    *,
    catalog: str | None = None,
    schema: str | None = None,
) -> Dict[str, Any]:
    """Return column metadata for the requested table."""
    safe_table = _validate_identifier("table", table)
    db_catalog, db_schema = _default_catalog_schema(catalog, schema)
    statement = (
        f"DESCRIBE TABLE "
        f"{_quote_identifier(db_catalog)}."
        f"{_quote_identifier(db_schema)}."
        f"{_quote_identifier(safe_table)}"
    )
    columns, rows = _execute_query(statement)
    records: List[Dict[str, str]] = []
    for row in rows:
        row_dict = dict(zip(columns, row))
        col_name = str(row_dict.get("col_name") or "").strip()
        if not col_name or col_name.startswith("#"):
            continue
        records.append(
            {
                "name": col_name,
                "type": str(row_dict.get("data_type") or "").strip(),
                "comment": str(row_dict.get("comment") or "").strip(),
            }
        )
    return {
        "statement": statement,
        "columns": records,
        "catalog": db_catalog,
        "schema": db_schema,
        "full_name": (
            f"{_quote_identifier(db_catalog)}."
            f"{_quote_identifier(db_schema)}."
            f"{_quote_identifier(safe_table)}"
        ),
    }


def run_sql_query(statement: str) -> Tuple[Sequence[str], List[Dict[str, Any]]]:
    """Execute an arbitrary SQL statement and return columns plus row dicts."""
    columns, rows = _execute_query(statement)
    return columns, [dict(zip(columns, row)) for row in rows]
