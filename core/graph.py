"""LangGraph nodes and builder for the Databricks assistant."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Literal, Tuple, TypedDict

from langgraph.graph import END, StateGraph

from core.databricks import run_sql_query
from core.llm_io import (
    AgentIntent,
    ChatMessage,
    ClarificationRequest,
    DEFAULT_PROVIDER,
    GeneratedSQL,
    Intent,
    StructuredAnswer,
    VisualizationCodeResponse,
    SUMMARY_SYSTEM_PROMPT,
    _call_llm,
    _call_structured_llm,
    _safe_json_dumps,
    _strip_code_block,
)
from core.schema_utils import (
    format_schema_context,
    resolve_table_sequence,
    select_relevant_metadata,
    table_full_name,
)
from core.sql_rules import validate_sql_statement
from core.trace_utils import (
    append_node_trace as _append_node_trace,
    append_trace_text as _append_trace_text,
    log_node_io as _log_node_io,
    node_path_diagram as _node_path_diagram,
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv():
        return None

load_dotenv()


class AgentState(TypedDict, total=False):
    state_messages: List[ChatMessage]
    state_limit_requested: bool
    state_limit_only: bool
    state_loading_requested: bool
    state_loaded_data: Any
    state_loaded_table: str
    state_loaded_columns: List[str]
    state_active_table_metadata: Dict[str, Any]
    state_query_columns: List[str]
    state_table_sequence: List[str]
    state_table_queue: List[str]
    state_current_table: str
    state_table_outputs: List[Dict[str, Any]]
    state_visualization_blocks: List[str]
    state_table_metadata: Dict[str, Any]
    state_user_query: str
    state_clean_user_query: str
    state_visualization_mode: Literal["raw_detail", "sql_aggregate", "ambiguous"]
    state_visualization_mode_reason: str
    state_require_mode_confirmation: bool
    state_size_guardrail_triggered: bool
    state_intent: AgentIntent
    state_sql_query: str
    state_sql_validation_error: str
    state_sql_retry_count: int
    state_sql_limit: int
    state_sql_tool_description: str
    state_sql_tool_active: bool
    state_query_columns: List[str]
    state_visualization_code: str
    state_response: str
    state_error_message: str
    state_node_path: List[str]
    state_node_traces: List[Dict[str, Any]]


MULTI_TABLE_KEYWORDS = (
    "모든 테이블",
    "전체 테이블",
    "각 테이블",
    "순차적으로",
    "순차",
    "all tables",
    "each table",
    "sequential tables",
)
MULTI_TABLE_SAMPLE_LIMIT = 200
MAX_MULTI_TABLES = 5
LIMIT_DIRECTIVE_PATTERN = re.compile(r"%limit\s+(\d+)", re.IGNORECASE)
LOADING_DIRECTIVE_PATTERN = re.compile(r"%loading\b", re.IGNORECASE)
SQL_VALIDATION_MAX_RETRIES = 2
DEFAULT_SQL_LIMIT = 500
AGGREGATE_KEYWORDS = (
    "평균",
    "합계",
    "총",
    "총합",
    "top",
    "top ",
    "순위",
    "랭킹",
    "비율",
    "퍼센트",
    "percent",
    "ratio",
    "분포",
    "distribution",
    "추세",
    "trend",
    "집계",
    "count",
    "sum",
    "avg",
    "최대",
    "최소",
    "상위",
)
DETAIL_KEYWORDS = (
    "모든 데이터",
    "전체 데이터",
    "세부",
    "raw",
    "원시",
    "행 단위",
    "row",
    "레코드",
    "필터링",
    "where",
    "조건",
    "시간별",
    "분 단위",
    "초 단위",
    "timestamp",
)
ROW_COUNT_GUARDRAIL = 1_000_000


def _latest_limit_override(messages: List[ChatMessage]) -> int | None:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        override_match = LIMIT_DIRECTIVE_PATTERN.search(message.get("content", ""))
        if override_match:
            return int(override_match.group(1))
    return None


def _format_schema_context(table_metadata: Dict[str, Any] | None) -> str:
    return format_schema_context(table_metadata)


def _select_relevant_metadata(
    user_query: str, table_metadata: Dict[str, Any] | None
) -> Dict[str, Any]:
    return select_relevant_metadata(user_query, table_metadata, MAX_MULTI_TABLES, MULTI_TABLE_KEYWORDS)


def _s2w_tool_description(
    limit: int,
    table_metadata: Dict[str, Any] | None = None,
    mode: str | None = None,
) -> str:
    schema_context = _format_schema_context(table_metadata)
    clauses = [
        "Tool name: s2w (Safe-to-Visualize SQL Writer).",
        "Purpose: build Databricks SQL optimized for visualization while staying read-only.",
        "Guidance:",
        "- Always start filters with `WHERE 1=1` even when no additional conditions exist.",
        "- Wrap every table identifier in backticks and keep the fully-qualified names from the schema section.",
        "- Select only the columns required for the requested chart; avoid `SELECT *`.",
        f"- Apply an explicit LIMIT {limit} (do not choose a smaller value unless a lower user limit is given).",
        "- Keep the statement single-query and free of DDL/DML.",
        "- Use the fully-qualified table names exactly as listed below; do not invent or change catalogs/schemas.",
    ]
    if mode == "sql_aggregate":
        clauses.append(
            "- Favor GROUP BY + aggregates over raw rows to keep the result compact (e.g., counts, sums, averages)."
        )
    elif mode == "raw_detail":
        clauses.append(
            "- The user needs row-level detail; keep WHERE filters explicit and avoid unnecessary aggregates."
        )
    description = "\n".join(clauses)
    if schema_context:
        description = f"{description}\n\n{schema_context}"
    return (
        description
        + f"\n\nDefault LIMIT: {limit} (override when a message starts with `%limit <value>`)."
    )


def _latest_user_query(messages: List[ChatMessage]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def _estimate_row_count(table_metadata: Dict[str, Any] | None) -> int | None:
    if not table_metadata:
        return None
    estimates: List[int] = []
    for meta in table_metadata.values():
        if not isinstance(meta, dict):
            continue
        for key in ("row_count", "rows", "estimated_rows", "approx_row_count", "approx_rows"):
            value = meta.get(key)
            if isinstance(value, (int, float)):
                try:
                    estimates.append(int(value))
                except Exception:
                    continue
    if not estimates:
        return None
    return max(estimates)


def _infer_visualization_mode(
    user_query: str,
    table_metadata: Dict[str, Any] | None,
    limit: int | None = None,
) -> Tuple[Literal["raw_detail", "sql_aggregate", "ambiguous"], str, bool]:
    normalized = (user_query or "").lower()
    agg_hits = [kw for kw in AGGREGATE_KEYWORDS if kw.lower() in normalized]
    detail_hits = [kw for kw in DETAIL_KEYWORDS if kw.lower() in normalized]
    estimated_rows = _estimate_row_count(table_metadata)
    guardrail = bool(estimated_rows and estimated_rows >= ROW_COUNT_GUARDRAIL)
    if guardrail:
        reason = f"추정 행 수 {estimated_rows:,}행 이상으로 감지되어 집계/요약 모드가 안전합니다."
        return "sql_aggregate", reason, True
    if agg_hits and len(agg_hits) >= len(detail_hits):
        keywords = ", ".join(sorted(set(agg_hits)))
        return "sql_aggregate", f"집계/랭킹 관련 키워드 감지: {keywords}", False
    if detail_hits and len(detail_hits) > len(agg_hits):
        keywords = ", ".join(sorted(set(detail_hits)))
        return "raw_detail", f"세부/행 기반 키워드 감지: {keywords}", False
    if limit and limit <= 20000 and not agg_hits:
        return "raw_detail", f"LIMIT {limit} 기반 세부 샘플링 요청으로 추정", False
    return "ambiguous", "세부 vs 집계 의도가 불분명해 확인이 필요합니다.", False


def build_conversation_graph(provider: str | None = None):
    """Compile a LangGraph that routes requests through intent/SQL/visual nodes."""
    selected = (provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    workflow = StateGraph(AgentState)

    def with_path(state: AgentState, node_name: str, updates: AgentState) -> AgentState:
        payload = dict(updates)
        path = list(state.get("state_node_path", []))
        path.append(node_name)
        payload["state_node_path"] = path
        traces, input_snapshot, output_snapshot, merged_state = _append_node_trace(
            state, node_name, payload
        )
        payload["state_node_traces"] = traces
        _log_node_io(node_name, input_snapshot, output_snapshot, merged_state)
        return payload

    # ----------------------
    # Compact node functions
    # ----------------------

    def fn_ingest(state: AgentState) -> AgentState:
        """Extract latest user query, apply %limit, and classify intent in one step."""
        query = _latest_user_query(state.get("state_messages", []))
        cleaned_query = re.sub(LIMIT_DIRECTIVE_PATTERN, "", query or "", count=1).strip()
        limit_requested = bool(LIMIT_DIRECTIVE_PATTERN.search(query or ""))
        loading_requested = bool(LOADING_DIRECTIVE_PATTERN.search(query or "")) or bool(
            re.search(r"로딩해줘|로딩해|loading\s*해줘|loading", (query or ""), re.IGNORECASE)
        )
        messages = state.get("state_messages", [])
        limit = state.get("state_sql_limit", DEFAULT_SQL_LIMIT)
        override = _latest_limit_override(messages)
        if override is not None:
            limit = override
        limit_only = limit_requested and not cleaned_query
        default_mode_state: AgentState = {
            "state_visualization_mode": "",
            "state_visualization_mode_reason": "",
            "state_require_mode_confirmation": False,
            "state_size_guardrail_triggered": False,
        }
        if limit_only or not query:
            updates: AgentState = {
                "state_user_query": query,
                "state_clean_user_query": cleaned_query,
                "state_sql_limit": limit,
                "state_limit_only": limit_only,
                "state_limit_requested": limit_requested,
                "state_loading_requested": loading_requested,
                "state_intent": "simple_answer",
                **default_mode_state,
            }
            return with_path(state, "node_ingest", updates)

        system_prompt = "You categorize user requests for a Databricks analytics assistant."
        user_prompt = f"User query:\n{query}"
        try:
            parsed = _call_structured_llm(
                selected, system_prompt, user_prompt, Intent, history=state.get("state_messages")
            )
            updates: AgentState = {
                "state_intent": parsed.intent,
                "state_user_query": query,
                "state_clean_user_query": cleaned_query,
                "state_sql_limit": limit,
                "state_limit_only": limit_only,
                "state_limit_requested": limit_requested,
                "state_loading_requested": loading_requested,
            }
            updates.update(default_mode_state)
            if parsed.intent == "visualize":
                limit_hint = state.get("state_sql_limit") or DEFAULT_SQL_LIMIT
                mode, reason, guardrail = _infer_visualization_mode(
                    query, state.get("state_table_metadata"), limit_hint
                )
                updates.update(
                    {
                        "state_visualization_mode": mode,
                        "state_visualization_mode_reason": reason,
                        "state_require_mode_confirmation": mode == "ambiguous" or guardrail,
                        "state_size_guardrail_triggered": guardrail,
                    }
                )
            return with_path(state, "node_ingest", updates)
        except Exception as exc:  # Includes JSON decoding and Pydantic validation errors
            error_details = f"{exc.__class__.__name__}: {exc}"
            print(f"[intent] Intent classification failed: {error_details}")
            updates: AgentState = {
                "state_intent": "simple_answer",
                "state_error_message": f"의도 분류 실패: {error_details}",
                **default_mode_state,
            }
            updates.update(
                {
                    "state_user_query": query,
                    "state_clean_user_query": cleaned_query,
                    "state_sql_limit": limit,
                    "state_limit_only": limit_only,
                    "state_limit_requested": limit_requested,
                    "state_loading_requested": loading_requested,
                }
            )
            return with_path(state, "node_ingest", updates)

    def fn_plan_tables(state: AgentState) -> AgentState:
        """Only manage table queue/selection and build tool context."""
        user_query = state.get("state_clean_user_query") or state.get("state_user_query", "")
        metadata = state.get("state_table_metadata") or {}
        table_queue = list(state.get("state_table_queue") or [])
        current_table = state.get("state_current_table", "")
        active_meta = state.get("state_active_table_metadata") or {}
        loaded_table = state.get("state_loaded_table", "")
        if not current_table and table_queue:
            current_table = table_queue.pop(0)
        if not current_table and metadata:
            candidates = resolve_table_sequence(None, user_query, metadata, MAX_MULTI_TABLES, MULTI_TABLE_KEYWORDS)
            if candidates:
                current_table = candidates[0]
                table_queue = candidates[1:]
        if current_table and metadata:
            meta = metadata.get(current_table, {})
            active_meta = {current_table: meta}
        if not active_meta and loaded_table and loaded_table in metadata:
            active_meta = {loaded_table: metadata[loaded_table]}
            current_table = current_table or loaded_table
        if not active_meta:
            active_meta = _select_relevant_metadata(user_query, metadata, MAX_MULTI_TABLES, MULTI_TABLE_KEYWORDS)
        if metadata and not active_meta and not state.get("state_loaded_data"):
            return with_path(
                state,
                "node_plan_tables",
                {"state_error_message": "요청한 테이블 메타데이터를 찾을 수 없습니다."},
            )
        limit = state.get("state_sql_limit", DEFAULT_SQL_LIMIT)
        updates: AgentState = {
            "state_active_table_metadata": active_meta,
            "state_table_queue": table_queue,
            "state_current_table": current_table,
            "state_sql_tool_description": _s2w_tool_description(
                limit, active_meta or metadata, state.get("state_visualization_mode")
            ),
            "state_sql_tool_active": True,
        }
        return with_path(state, "node_plan_tables", updates)

    def fn_prepare_sql(state: AgentState) -> AgentState:
        """Generate and validate SQL with internal retry loop."""
        query = state.get("state_clean_user_query") or state.get("state_user_query", "")
        limit = state.get("state_sql_limit", DEFAULT_SQL_LIMIT)
        tool_prompt = state.get("state_sql_tool_description") or _s2w_tool_description(
            limit,
            state.get("state_active_table_metadata") or state.get("state_table_metadata"),
            state.get("state_visualization_mode"),
        )
        sql: str = state.get("state_sql_query", "")
        error_message = ""

        for attempt in range(SQL_VALIDATION_MAX_RETRIES + 1):
            if not sql:
                feedback = state.get("state_sql_validation_error") if attempt else ""
                user_prompt = (
                    f"User request:\n{query}\n\n"
                    "Populate the `sql` field with the final SQL query."
                )
                if limit:
                    user_prompt += f"\nUse an explicit LIMIT {limit} (do not pick a smaller value)."
                if feedback:
                    user_prompt += (
                        "\n\nA previous attempt failed automatic validation for this reason:\n"
                        f"{feedback}\n"
                        "Revise the SQL accordingly and ensure only one SELECT statement is returned."
                    )
                if attempt:
                    user_prompt += f"\n(Attempt #{attempt + 1} — pay extra attention to syntax.)"
                try:
                    structured = _call_structured_llm(
                        selected,
                        tool_prompt,
                        user_prompt,
                        GeneratedSQL,
                        history=state.get("state_messages"),
                    )
                    raw_sql = structured.sql.strip()
                except Exception:
                    raw_sql = _call_llm(
                        selected, tool_prompt, user_prompt, history=state.get("state_messages")
                    ).strip()
                sql = _strip_code_block(raw_sql)

            validation_error = validate_sql_statement(
                sql,
                max_limit=limit if state.get("state_sql_tool_active") else None,
                require_where=bool(state.get("state_sql_tool_active")),
                table_metadata=state.get("state_active_table_metadata") or state.get("state_table_metadata"),
            )
            if not validation_error:
                break
            if attempt >= SQL_VALIDATION_MAX_RETRIES:
                error_message = f"SQL 검증 실패: {validation_error}"
                break
            sql = ""
            state["state_sql_validation_error"] = validation_error

        updates: AgentState = {
            "state_sql_query": sql,
            "state_error_message": error_message,
            "state_sql_validation_error": "",
        }
        if error_message:
            updates["state_sql_validation_error"] = error_message
        return with_path(state, "node_prepare_sql", updates)

    def fn_execute_sql(state: AgentState) -> AgentState:
        statement = state.get("state_sql_query", "")
        if not statement:
            return with_path(state, "node_run_query", {"state_error_message": "SQL 문을 생성하지 못했습니다."})
        try:
            columns, rows = run_sql_query(statement)
        except Exception as exc:  # pragma: no cover - depends on environment
            return with_path(
                state, "node_run_query", {"state_error_message": str(exc), "state_loaded_data": []}
            )
        table_name = state.get("state_current_table", "") or ""
        updates: AgentState = {
            "state_query_columns": list(columns),
            "state_loaded_data": rows,
            "state_loaded_columns": list(columns),
            "state_loaded_table": table_name,
        }
        if state.get("state_intent") != "visualize":
            outputs = list(state.get("state_table_outputs") or [])
            if table_name:
                outputs.append(
                    {
                        "table": table_name,
                        "sql": state.get("state_sql_query", ""),
                        "row_count": len(rows),
                        "rows": rows[:20],
                    }
                )
            updates.update(
                {
                    "state_table_outputs": outputs,
                    "state_current_table": "",
                    "state_table_queue": state.get("state_table_queue") or [],
                }
            )
        return with_path(state, "node_run_query", updates)

    def fn_plan_visualization(state: AgentState) -> AgentState:
        rows = state.get("state_loaded_data", []) or []
        sample = rows[:10]
        data_json = _safe_json_dumps(sample)
        user_query = state.get("state_user_query", "")
        system_prompt = (
            "You create Matplotlib visualization code for tabular data. "
            "A pandas DataFrame named `df` is already constructed from the provided rows; use it directly. "
            "Imports for pandas as pd and matplotlib.pyplot as plt are already available. "
            "Close with plt.tight_layout(). Return only a Python code block."
        )
        table_name = state.get("state_current_table")
        table_context = f"Current table: {table_name}\n" if table_name else ""
        user_prompt = (
            f"{table_context}User request:\n{user_query}\n\n"
            f"Sample rows JSON (for schema only):\n{data_json}\n\n"
            "Generate concise visualization code that assumes a DataFrame `df` is already available. "
            "Populate `language` and `code` fields."
        )
        try:
            structured = _call_structured_llm(
                selected,
                system_prompt,
                user_prompt,
                VisualizationCodeResponse,
                history=state.get("state_messages"),
            )
            if structured.language.lower() != "python":
                raise ValueError("Visualization code must be in Python.")
            code = structured.code.strip()
        except Exception:
            code = _call_llm(
                selected, system_prompt, user_prompt, history=state.get("state_messages")
            ).strip()
        clean_code = _strip_code_block(code)
        wrapped = f"```python\n{clean_code}\n```"
        updates: AgentState = {"state_visualization_code": wrapped}
        table_name = state.get("state_current_table", "")
        outputs = list(state.get("state_table_outputs") or [])
        viz_blocks = list(state.get("state_visualization_blocks") or [])
        if table_name:
            outputs.append(
                {
                    "table": table_name,
                    "sql": state.get("state_sql_query", ""),
                    "row_count": len(rows),
                    "rows": rows[:20],
                }
            )
            viz_blocks.append(f"**{table_name}** 테이블 시각화\n\n{wrapped}")
            updates["state_current_table"] = ""
        else:
            viz_blocks.append(wrapped)
        updates["state_table_outputs"] = outputs
        updates["state_visualization_blocks"] = viz_blocks
        updates["state_visualization_code"] = wrapped
        updates["state_table_queue"] = state.get("state_table_queue") or []
        return with_path(state, "node_visualization", updates)

    def fn_respond(state: AgentState) -> AgentState:
        intent = state.get("state_intent", "simple_answer")
        user_query = state.get("state_user_query", "")
        table_outputs = state.get("state_table_outputs") or []
        visualization_blocks = state.get("state_visualization_blocks") or []
        sql_query = state.get("state_sql_query", "")
        loaded_rows = state.get("state_loaded_data") or []
        loaded_table = state.get("state_loaded_table", "")
        loaded_columns = state.get("state_loaded_columns") or []
        visualization_code = state.get("state_visualization_code")
        if state.get("state_require_mode_confirmation") or intent == "clarify":
            mode_reason = state.get("state_visualization_mode_reason", "")
            system_prompt = (
                "You must ask the user to choose between row-level visualization (DataFrame) "
                "and aggregated SQL summary. Keep it short, Korean, and provide 2 options with examples."
                if state.get("state_require_mode_confirmation")
                else "You need more detail from the user before running SQL. Ask a short follow-up question in Korean."
            )
            user_prompt = (
                f"User query:\n{user_query}\n\n"
                f"Explain why confirmation is needed: {mode_reason or '집계/세부 의도가 혼재되어 있음'}\n"
                "Offer two choices:\n"
                "1) 세부 데이터/행 기반 로딩 (예: '지난주 특정 제품의 시간별 판매량 라인차트')\n"
                "2) 집계/요약 SQL (예: '전체 기간 가장 많이 팔린 제품 TOP 10 막대 그래프', '월별 총 매출 추세')\n"
                "Ask the user to pick one of the above."
                if state.get("state_require_mode_confirmation")
                else f"User query:\n{user_query}\n\nAsk what detail is missing. Populate the `follow_up_question` field."
            )
            try:
                structured = _call_structured_llm(
                    selected,
                    system_prompt,
                    user_prompt,
                    ClarificationRequest,
                    history=state.get("state_messages"),
                )
                reply = structured.follow_up_question.strip()
            except Exception:
                reply = _call_llm(
                    selected, system_prompt, user_prompt, history=state.get("state_messages")
                ).strip()
            final_text = reply
        elif state.get("state_limit_only"):
            limit = state.get("state_sql_limit", DEFAULT_SQL_LIMIT)
            final_text = f"결과 조회 제한을 {limit}행으로 설정했어요. 새로운 요청으로 계속 진행할 수 있어요."
            response_path = list(state.get("state_node_path", [])) + ["node_respond"]
            final_text = f"{final_text}{_node_path_diagram(response_path)}"
            final_text = _append_trace_text(final_text, state)
            messages = state.get("state_messages", [])
            return with_path(
                state,
                "node_respond",
                {
                    "state_response": final_text,
                    "state_messages": messages + [{"role": "assistant", "content": final_text}],
                },
            )
        if table_outputs:
            previews_json = _safe_json_dumps(table_outputs)
            system_prompt = (
                "You review sequential table previews. Summarize each table in order, "
                "highlighting row counts and notable columns. Keep the tone concise and "
                "fn_respond in Korean."
            )
            user_prompt = (
                f"User query: {user_query}\nIntent: {intent}\n"
                f"Table previews JSON:\n{previews_json}\n\n"
                "Write sectioned summaries (one per table). Populate `answer` and optionally `highlights`."
            )
            try:
                structured = _call_structured_llm(
                    selected,
                    system_prompt,
                    user_prompt,
                    StructuredAnswer,
                    history=state.get("state_messages"),
                )
                reply = structured.answer.strip()
                if structured.highlights:
                    highlight_lines = "\n".join(
                        f"- {item.strip()}" for item in structured.highlights if item.strip()
                    )
                    if highlight_lines:
                        reply = f"{reply}\n\n하이라이트:\n{highlight_lines}"
            except Exception:
                reply = _call_llm(
                    selected, system_prompt, user_prompt, history=state.get("state_messages")
                ).strip()
            final_text = reply
            if visualization_blocks:
                blocks = "\n\n".join(visualization_blocks)
                final_text = f"{reply}\n\n{blocks}"
        elif loaded_rows or loaded_columns:
            count = len(loaded_rows)
            col_count = len(loaded_columns)
            col_list = ", ".join(loaded_columns) if loaded_columns else "N/A"
            table_label = loaded_table or "쿼리 결과"
            extra_note = "\n조회된 행은 없지만 컬럼 스키마를 먼저 보여드려요." if not loaded_rows else ""
            final_text = (
                f"`{table_label}` 데이터 {count}행, {col_count}개 컬럼을 로딩했어요.\n"
                f"컬럼: {col_list}{extra_note}"
            )
        else:
            rows = state.get("state_loaded_data", []) or []
            sample = rows[:20]
            data_json = _safe_json_dumps(sample)
            system_prompt = (
                "You are a helpful analytics assistant. Summarize query results clearly. "
                "If no data is available, fn_respond conversationally using general knowledge."
            )
            user_prompt = (
                f"Intent: {intent}\nUser query: {user_query}\n"
                f"SQL: {sql_query or 'N/A'}\nData preview JSON:\n{data_json}\n\n"
                "Write the final response in Korean, summarizing insights. "
                "Mention charts if visualization code is provided. "
                "Populate `answer` and optionally `highlights`."
            )
            try:
                structured = _call_structured_llm(
                    selected,
                    system_prompt,
                    user_prompt,
                    StructuredAnswer,
                    history=state.get("state_messages"),
                )
                reply = structured.answer.strip()
                if structured.highlights:
                    highlight_lines = "\n".join(
                        f"- {item.strip()}" for item in structured.highlights if item.strip()
                    )
                    if highlight_lines:
                        reply = f"{reply}\n\n하이라이트:\n{highlight_lines}"
            except Exception:
                reply = _call_llm(
                    selected, system_prompt, user_prompt, history=state.get("state_messages")
                ).strip()
            final_text = reply

        if intent == "visualize":
            mode = state.get("state_visualization_mode")
            reason = state.get("state_visualization_mode_reason") or ""
            if mode in {"raw_detail", "sql_aggregate"}:
                label = "세부 데이터(행 기반)" if mode == "raw_detail" else "집계/요약(SQL)"
                reason_text = f" (근거: {reason})" if reason else ""
                final_text = (
                    f"{final_text}\n\n모드 안내: 이번 요청은 {label}으로 처리했어요.{reason_text} "
                    "다른 방식이 필요하면 '세부 데이터로 로딩' 또는 'SQL 집계로 대신 보여줘'라고 알려주세요."
                )

        if visualization_code:
            final_text = f"{final_text}\n\n{visualization_code}"

        if sql_query:
            final_text = f"{final_text}\n\n생성된 SQL:\n```sql\n{sql_query}\n```"
        response_path = list(state.get("state_node_path", [])) + ["node_respond"]
        final_text = f"{final_text}{_node_path_diagram(response_path)}"
        final_text = _append_trace_text(final_text, state)
        messages = state.get("state_messages", [])
        return with_path(
            state,
            "node_respond",
            {
                "state_response": final_text,
                "state_messages": messages + [{"role": "assistant", "content": final_text}],
            },
        )

    def fn_handle_error(state: AgentState) -> AgentState:
        error_message = state.get("state_error_message", "알 수 없는 오류가 발생했습니다.")
        sql_query = state.get("state_sql_query", "")
        text = (
            "⚠️ Databricks 쿼리를 실행하는 동안 오류가 발생했습니다.\n"
            f"세부 정보: {error_message}\n다시 시도하거나 조건을 조정해 주세요."
        )
        if sql_query:
            text += f"\n\n생성된 SQL:\n```sql\n{sql_query}\n```"
        messages = state.get("state_messages", [])
        error_path = list(state.get("state_node_path", [])) + ["node_error"]
        final_text = f"{text}{_node_path_diagram(error_path)}"
        final_text = _append_trace_text(final_text, state)
        return with_path(
            state,
            "node_error",
            {
                "state_response": final_text,
                "state_messages": messages + [{"role": "assistant", "content": final_text}],
            },
        )

    def fn_route_ingest(state: AgentState) -> str:
        if state.get("state_limit_only"):
            return "node_respond"
        if state.get("state_require_mode_confirmation") or state.get("state_intent") == "clarify":
            return "node_respond"
        if state.get("state_intent") in {"visualize", "sql_query"} or state.get("state_loading_requested"):
            return "node_plan_tables"
        return "node_respond"

    def fn_route_plan(state: AgentState) -> str:
        if state.get("state_error_message"):
            return "node_error"
        return "node_prepare_sql"

    def fn_route_after_prepare(state: AgentState) -> str:
        if state.get("state_error_message"):
            return "node_error"
        return "node_run_query"

    def fn_route_after_execute(state: AgentState) -> str:
        if state.get("state_error_message"):
            return "node_error"
        if state.get("state_intent") == "visualize":
            return "node_visualization"
        if state.get("state_table_queue"):
            return "node_plan_tables"
        return "node_respond"

    def fn_route_visualization_next(state: AgentState) -> str:
        if state.get("state_table_queue"):
            return "node_plan_tables"
        return "node_respond"

    workflow.add_node("node_ingest", fn_ingest)
    workflow.add_node("node_plan_tables", fn_plan_tables)
    workflow.add_node("node_prepare_sql", fn_prepare_sql)
    workflow.add_node("node_run_query", fn_execute_sql)
    workflow.add_node("node_visualization", fn_plan_visualization)
    workflow.add_node("node_respond", fn_respond)
    workflow.add_node("node_error", fn_handle_error)

    workflow.set_entry_point("node_ingest")
    workflow.add_conditional_edges(
        "node_ingest",
        fn_route_ingest,
        {
            "node_respond": "node_respond",
            "node_plan_tables": "node_plan_tables",
        },
    )
    workflow.add_conditional_edges(
        "node_plan_tables",
        fn_route_plan,
        {
            "node_prepare_sql": "node_prepare_sql",
            "node_error": "node_error",
        },
    )
    workflow.add_conditional_edges(
        "node_prepare_sql",
        fn_route_after_prepare,
        {
            "node_run_query": "node_run_query",
            "node_error": "node_error",
        },
    )
    workflow.add_conditional_edges(
        "node_run_query",
        fn_route_after_execute,
        {
            "node_visualization": "node_visualization",
            "node_respond": "node_respond",
            "node_error": "node_error",
        },
    )
    workflow.add_conditional_edges(
        "node_visualization",
        fn_route_visualization_next,
        {
            "node_respond": "node_respond",
        },
    )
    workflow.add_edge("node_respond", END)
    workflow.add_edge("node_error", END)
    return workflow.compile()


def summary_seed(topic: str) -> List[ChatMessage]:
    """Seed messages for the summary workflow."""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Summarize the following topic in three bullet points: {topic}"},
    ]


def summarize_topic(topic: str, provider: str | None = None) -> str:
    """Run the summary LangGraph and return the assistant's reply."""
    graph = build_conversation_graph(provider)
    result = graph.invoke({"state_messages": summary_seed(topic)})
    return result["state_messages"][-1]["content"]


__all__ = [
    "AgentState",
    "DEFAULT_SQL_LIMIT",
    "build_conversation_graph",
    "summary_seed",
    "summarize_topic",
]
