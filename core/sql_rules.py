"""SQL parsing and validation helpers for Databricks-safe queries."""
from __future__ import annotations

import re
from typing import Any, Dict, List

SQL_DANGER_PATTERN = re.compile(r"(?i)\b(drop|delete|alter|update|insert|merge|truncate)\b")
SQL_SELECT_PATTERN = re.compile(r"(?i)\bselect\b")


def count_select_columns(statement: str) -> int:
    match = re.search(r"(?i)\bselect\b\s+(.*?)\bfrom\b", statement, re.S)
    if not match:
        return 0
    select_clause = match.group(1)
    columns: List[str] = []
    current: List[str] = []
    depth = 0
    for char in select_clause:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            columns.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        columns.append("".join(current).strip())
    return len([col for col in columns if col])


def normalize_table_identifier(token: str) -> str:
    cleaned = token.strip().strip("`").strip("[]").strip('"')
    parts = [part.strip("`[]\"") for part in cleaned.split(".") if part]
    return ".".join(parts).lower()


def extract_table_tokens(statement: str) -> List[str]:
    """Lightweight extractor for table tokens in FROM/JOIN clauses."""
    tokens: List[str] = []
    pattern = re.compile(r"(?i)\b(from|join)\s+([`\"\\[\\]A-Za-z0-9_\\.]+)")
    for match in pattern.finditer(statement):
        raw = match.group(2)
        raw = raw.split()[0].rstrip(",")  # strip alias/punctuation
        tokens.append(raw)
    return tokens


def extract_select_columns_simple(statement: str) -> List[str]:
    """Best-effort extraction of column identifiers in the top-level SELECT list."""
    match = re.search(r"(?is)\bselect\b\s+(.*?)\bfrom\b", statement)
    if not match:
        return []
    select_clause = match.group(1)
    columns: List[str] = []
    current: List[str] = []
    depth = 0
    for char in select_clause:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            columns.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        columns.append("".join(current).strip())

    extracted: List[str] = []
    for col in columns:
        if "(" in col:  # skip expressions/functions to reduce false positives
            continue
        alias_split = re.split(r"\s+as\s+|\s+", col, flags=re.IGNORECASE)
        base = alias_split[0] if alias_split else col
        base = base.strip().strip("`").strip("[]").strip('"')
        if not base or base == "*":
            continue
        if "." in base:
            base = base.split(".")[-1]
        extracted.append(base.lower())
    return extracted


def validate_sql_statement(
    statement: str,
    max_limit: int | None = None,
    *,
    require_where: bool = False,
    table_metadata: Dict[str, Any] | None = None,
) -> str | None:
    normalized = statement.strip()
    if not normalized:
        return "SQL 문장이 비어 있습니다."
    if SQL_DANGER_PATTERN.search(normalized):
        return "DDL/DML 구문(DROP/DELETE 등)은 허용되지 않습니다."
    select_matches = SQL_SELECT_PATTERN.findall(normalized)
    if len(select_matches) > 1:
        return "단일 SELECT 문만 허용됩니다."
    semicolons = normalized.count(";")
    if semicolons > 1 or (semicolons == 1 and not normalized.endswith(";")):
        return "세미콜론으로 구분된 다중 쿼리는 지원되지 않습니다."
    column_count = count_select_columns(normalized)
    group_match = re.search(r"(?i)group\s+by\s+([0-9,\s]+)", normalized)
    if column_count and group_match:
        numbers = [int(token) for token in re.findall(r"\d+", group_match.group(1))]
        if numbers and max(numbers) > column_count:
            return "GROUP BY 인덱스가 SELECT 항목 수를 초과합니다."
    order_match = re.search(r"(?i)order\s+by\s+([0-9,\s]+)", normalized)
    if column_count and order_match:
        numbers = [int(token) for token in re.findall(r"\d+", order_match.group(1))]
        if numbers and max(numbers) > column_count:
            return "ORDER BY 인덱스가 SELECT 항목 수를 초과합니다."
    if require_where and "where 1=1" not in normalized.lower():
        if not re.search(r"where\s+1\s*=\s*1", normalized.lower()):
            return "기본 필터로 WHERE 1=1 절을 포함해 주세요."
    if max_limit:
        limit_match = re.search(r"(?i)limit\s+(\d+)", normalized)
        if not limit_match:
            return f"LIMIT 절을 포함하고 값을 {max_limit} 이하로 설정해 주세요."
        limit_value = int(limit_match.group(1))
        if limit_value > max_limit:
            return f"LIMIT 값은 {max_limit} 이하로 설정해야 합니다."
    if table_metadata:
        table_tokens = extract_table_tokens(normalized)
        if len(set(normalize_table_identifier(t) for t in table_tokens)) > 1:
            return "현재는 단일 테이블만 조회할 수 있습니다. 여러 테이블/UNION/조인은 지원하지 않습니다."
        allowed_tables: set[str] = set()
        columns_by_table: Dict[str, set[str]] = {}
        for tname, meta in table_metadata.items():
            norm_key = normalize_table_identifier(tname)
            allowed_tables.add(norm_key)
            meta = meta or {}
            full_name = meta.get("full_name")
            if full_name:
                allowed_tables.add(normalize_table_identifier(full_name))
            col_set = {
                (col.get("name") or "").strip().lower()
                for col in (meta.get("columns") or [])
                if (col.get("name") or "").strip()
            }
            if col_set:
                columns_by_table[norm_key] = col_set
                if full_name:
                    columns_by_table[normalize_table_identifier(full_name)] = col_set
        if table_tokens:
            for token in table_tokens:
                norm_token = normalize_table_identifier(token)
                if norm_token not in allowed_tables:
                    return f"존재하지 않는 테이블을 참조합니다: {token}"
            if len(table_tokens) == 1:
                norm_token = normalize_table_identifier(table_tokens[0])
                col_set = columns_by_table.get(norm_token)
                if col_set:
                    select_columns = extract_select_columns_simple(normalized)
                    invalid = [col for col in select_columns if col not in col_set]
                    if invalid:
                        bad = ", ".join(sorted(set(invalid)))
                        return f"존재하지 않는 컬럼을 사용했습니다: {bad}"
    return None
