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


def _s2w_tool_description(limit: int, table_metadata: Dict[str, Any] | None = None) -> str:
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

    def fn_extract_user_query(state: AgentState) -> AgentState:
        query = _latest_user_query(state.get("state_messages", []))
        cleaned_query = re.sub(LIMIT_DIRECTIVE_PATTERN, "", query or "", count=1).strip()
        limit_requested = bool(LIMIT_DIRECTIVE_PATTERN.search(query or ""))
        loading_requested = bool(LOADING_DIRECTIVE_PATTERN.search(query or "")) or bool(
            re.search(r"로딩해줘|로딩해|loading\s*해줘|loading", (query or ""), re.IGNORECASE)
        )
        return with_path(
            state,
            "node_extract_user",
            {
                "state_user_query": query,
                "state_clean_user_query": cleaned_query,
                "state_sql_tool_active": False,
                "state_limit_requested": limit_requested,
                "state_loading_requested": loading_requested,
            },
        )

    def fn_configure_limits(state: AgentState) -> AgentState:
        messages = state.get("state_messages", [])
        limit = state.get("state_sql_limit") or DEFAULT_SQL_LIMIT
        override = _latest_limit_override(messages)
        if override is not None:
            limit = override
        user_query = state.get("state_user_query", "")
        cleaned_query = (state.get("state_clean_user_query") or "").strip()
        limit_only = bool(LIMIT_DIRECTIVE_PATTERN.search(user_query)) and not cleaned_query
        return with_path(
            state,
            "node_configure_limits",
            {
                "state_sql_limit": limit,
                "state_limit_only": limit_only,
            },
        )

    def fn_classify_intent(state: AgentState) -> AgentState:
        query = state.get("state_clean_user_query") or state.get("state_user_query", "")
        limit_only = state.get("state_limit_requested") and not (state.get("state_clean_user_query") or "").strip()
        if limit_only:
            return with_path(state, "node_intent_classifier", {"state_intent": "simple_answer"})
        if not query:
            return with_path(state, "node_intent_classifier", {"state_intent": "simple_answer"})

        system_prompt = "You categorize user requests for a Databricks analytics assistant."
        user_prompt = f"User query:\n{query}"
        try:
            parsed = _call_structured_llm(selected, system_prompt, user_prompt, Intent)
            return with_path(state, "node_intent_classifier", {"state_intent": parsed.intent})
        except Exception as exc:  # Includes JSON decoding and Pydantic validation errors
            error_details = f"{exc.__class__.__name__}: {exc}"
            print(f"[intent] Intent classification failed: {error_details}")
            return with_path(
                state,
                "node_intent_classifier",
                {
                    "state_intent": "simple_answer",
                    "state_error_message": f"의도 분류 실패: {error_details}",
                },
            )

    def fn_prepare_s2w_tool_context(state: AgentState) -> AgentState:
        limit = state.get("state_sql_limit", DEFAULT_SQL_LIMIT)
        user_query = state.get("state_user_query", "")
        metadata = state.get("state_table_metadata") or {}
        relevant_meta = _select_relevant_metadata(user_query, metadata)
        updates: AgentState = {"state_active_table_metadata": relevant_meta}
        if metadata and not relevant_meta:
            if state.get("state_intent") == "visualize" and state.get("state_loaded_data"):
                updates["state_active_table_metadata"] = {}
            else:
                updates.update(
                    {
                        "state_sql_validation_error": "요청한 테이블 메타데이터를 찾을 수 없습니다.",
                        "state_sql_retry_count": SQL_VALIDATION_MAX_RETRIES,
                        "state_error_message": "요청한 테이블 메타데이터를 찾을 수 없습니다.",
                    }
                )
                return with_path(state, "node_s2w_tool", updates)
        tool_description = _s2w_tool_description(limit, relevant_meta)
        updates.update(
            {
                "state_sql_tool_description": tool_description,
                "state_sql_tool_active": True,
            }
        )
        return with_path(
            state,
            "node_s2w_tool",
            updates,
        )

    def fn_generate_sql(state: AgentState) -> AgentState:
        query = state.get("state_clean_user_query") or state.get("state_user_query", "")
        feedback = state.get("state_sql_validation_error")
        retry_count = state.get("state_sql_retry_count", 0)
        limit = state.get("state_sql_limit", DEFAULT_SQL_LIMIT)
        system_prompt = _s2w_tool_description(
            limit, state.get("state_active_table_metadata") or state.get("state_table_metadata")
        )
        tool_details = state.get("state_sql_tool_description")
        if tool_details:
            system_prompt = tool_details
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
        if retry_count:
            user_prompt += f"\n(Attempt #{retry_count + 1} — pay extra attention to syntax.)"
        try:
            structured = _call_structured_llm(
                selected, system_prompt, user_prompt, GeneratedSQL
            )
            raw_sql = structured.sql.strip()
        except Exception:
            raw_sql = _call_llm(selected, system_prompt, user_prompt).strip()
        sql = _strip_code_block(raw_sql)
        return with_path(state, "node_sql_generator", {"state_sql_query": sql})

    def fn_select_next_table(state: AgentState) -> AgentState:
        queue = list(state.get("state_table_queue") or [])
        updates: AgentState = {}
        if not queue:
            updates["state_current_table"] = ""
            return with_path(
                state, "node_table_select", updates
            )
        current = queue.pop(0)
        updates["state_current_table"] = current
        updates["state_table_queue"] = queue
        updates["state_sql_query"] = ""
        updates["state_sql_validation_error"] = ""
        updates["state_sql_retry_count"] = 0
        updates["state_sql_tool_active"] = False
        return with_path(state, "node_table_select", updates)

    def fn_build_table_sql(state: AgentState) -> AgentState:
        table_name = state.get("state_current_table")
        if not table_name:
            return with_path(
                state, "node_table_sql", {"state_error_message": "순차적으로 조회할 테이블이 없습니다."}
            )
        metadata = (state.get("state_table_metadata") or {}).get(table_name, {})
        limit = metadata.get("preview_limit") or MULTI_TABLE_SAMPLE_LIMIT
        if not isinstance(limit, int) or limit <= 0:
            limit = MULTI_TABLE_SAMPLE_LIMIT
        full_name = table_full_name(table_name, metadata)
        statement = f"SELECT * FROM {full_name} LIMIT {limit}"
        return with_path(state, "node_table_sql", {"state_sql_query": statement})

    def fn_validate_sql(state: AgentState) -> AgentState:
        statement = state.get("state_sql_query", "")
        max_limit = state.get("state_sql_limit") if state.get("state_sql_tool_active") else None
        require_where = bool(state.get("state_sql_tool_active"))
        validation_error = validate_sql_statement(
            statement,
            max_limit=max_limit,
            require_where=require_where,
            table_metadata=state.get("state_active_table_metadata") or state.get("state_table_metadata"),
        )
        retries = state.get("state_sql_retry_count", 0)
        updates: AgentState = {}
        if validation_error:
            retries += 1
            updates["state_sql_validation_error"] = validation_error
            updates["state_sql_retry_count"] = retries
            if retries >= SQL_VALIDATION_MAX_RETRIES:
                updates["state_error_message"] = f"SQL 검증 실패: {validation_error}"
        else:
            updates["state_sql_validation_error"] = ""
        return with_path(state, "node_sql_validator", updates)

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
        return with_path(state, "node_run_query", updates)

    def fn_load_data(state: AgentState) -> AgentState:
        rows = state.get("state_loaded_data", []) or []
        columns = state.get("state_query_columns") or []
        table_name = state.get("state_current_table", "") or ""
        full_meta = state.get("state_table_metadata") or {}
        selected_meta = {table_name: full_meta[table_name]} if table_name and table_name in full_meta else {}
        updates: AgentState = {
            "state_loaded_data": rows,
            "state_loaded_table": table_name,
            "state_loaded_columns": list(columns),
            "state_active_table_metadata": selected_meta,
        }
        return with_path(state, "node_load_data", updates)

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
                selected, system_prompt, user_prompt, VisualizationCodeResponse
            )
            if structured.language.lower() != "python":
                raise ValueError("Visualization code must be in Python.")
            code = structured.code.strip()
        except Exception:
            code = _call_llm(selected, system_prompt, user_prompt).strip()
        clean_code = _strip_code_block(code)
        wrapped = f"```python\n{clean_code}\n```"
        return with_path(state, "node_visualization", {"state_visualization_code": wrapped})

    def fn_use_loaded_data(state: AgentState) -> AgentState:
        rows = state.get("state_loaded_data") or []
        columns = state.get("state_loaded_columns") or []
        table_name = state.get("state_loaded_table", "")
        if not rows:
            return with_path(
                state,
                "node_use_loaded_data",
                {"state_error_message": "로딩된 데이터가 없어 시각화를 진행할 수 없습니다."},
            )
        updates: AgentState = {
            "state_query_columns": columns,
            "state_current_table": table_name,
        }
        return with_path(state, "node_use_loaded_data", updates)

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
        if state.get("state_limit_only"):
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
                    selected, system_prompt, user_prompt, StructuredAnswer
                )
                reply = structured.answer.strip()
                if structured.highlights:
                    highlight_lines = "\n".join(
                        f"- {item.strip()}" for item in structured.highlights if item.strip()
                    )
                    if highlight_lines:
                        reply = f"{reply}\n\n하이라이트:\n{highlight_lines}"
            except Exception:
                reply = _call_llm(selected, system_prompt, user_prompt).strip()
            final_text = reply
            if visualization_blocks:
                blocks = "\n\n".join(visualization_blocks)
                final_text = f"{reply}\n\n{blocks}"
        elif loaded_rows:
            count = len(loaded_rows)
            col_count = len(loaded_columns)
            col_list = ", ".join(loaded_columns) if loaded_columns else "N/A"
            table_label = loaded_table or "쿼리 결과"
            final_text = (
                f"`{table_label}` 데이터 {count}행, {col_count}개 컬럼을 로딩했어요.\n"
                f"컬럼: {col_list}"
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
                    selected, system_prompt, user_prompt, StructuredAnswer
                )
                reply = structured.answer.strip()
                if structured.highlights:
                    highlight_lines = "\n".join(
                        f"- {item.strip()}" for item in structured.highlights if item.strip()
                    )
                    if highlight_lines:
                        reply = f"{reply}\n\n하이라이트:\n{highlight_lines}"
            except Exception:
                reply = _call_llm(selected, system_prompt, user_prompt).strip()
            final_text = reply

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

    def fn_clarify(state: AgentState) -> AgentState:
        user_query = state.get("state_user_query", "")
        system_prompt = (
            "You need more detail from the user before running SQL. "
            "Ask a short follow-up question in Korean."
        )
        user_prompt = (
            f"User query:\n{user_query}\n\nAsk what detail is missing. "
            "Populate the `follow_up_question` field."
        )
        try:
            structured = _call_structured_llm(
                selected, system_prompt, user_prompt, ClarificationRequest
            )
            reply = structured.follow_up_question.strip()
        except Exception:
            reply = _call_llm(selected, system_prompt, user_prompt).strip()
        messages = state.get("state_messages", [])
        clarify_path = list(state.get("state_node_path", [])) + ["node_clarify"]
        final_text = f"{reply}{_node_path_diagram(clarify_path)}"
        final_text = _append_trace_text(final_text, state)
        return with_path(
            state,
            "node_clarify",
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

    def fn_collect_table_results(state: AgentState) -> AgentState:
        table_name = state.get("state_current_table", "")
        rows = state.get("state_loaded_data", []) or []
        visualization_code = state.get("state_visualization_code", "")
        outputs = list(state.get("state_table_outputs") or [])
        if table_name:
            entry = {
                "table": table_name,
                "sql": state.get("state_sql_query", ""),
                "row_count": len(rows),
                "rows": rows[:20],
            }
            outputs.append(entry)
        viz_blocks = list(state.get("state_visualization_blocks") or [])
        if visualization_code:
            labeled_code = visualization_code
            if table_name:
                labeled_code = f"**{table_name}** 테이블 시각화\n\n{visualization_code}"
            viz_blocks.append(labeled_code)
        updates: AgentState = {
            "state_table_outputs": outputs,
            "state_visualization_blocks": viz_blocks,
            "state_visualization_code": "",
            "state_current_table": "",
        }
        return with_path(state, "node_table_results", updates)

    def fn_route_intent(state: AgentState) -> str:
        if state.get("state_limit_requested"):
            return "node_configure_limits"
        if state.get("state_table_queue"):
            return "node_table_select"
        intent = state.get("state_intent", "simple_answer")
        if intent == "visualize":
            user_query = state.get("state_clean_user_query") or state.get("state_user_query", "")
            table_meta = state.get("state_table_metadata")
            relevant = _select_relevant_metadata(user_query, table_meta) if table_meta else {}
            if state.get("state_loaded_data") and not relevant:
                return "node_use_loaded_data"
        if intent in {"visualize", "sql_query"}:
            return "node_s2w_tool"
        if state.get("state_loading_requested"):
            return "node_s2w_tool"
        if intent == "clarify":
            return "node_clarify"
        return "node_respond"

    def fn_route_validation(state: AgentState) -> str:
        if state.get("state_sql_validation_error"):
            if state.get("state_sql_retry_count", 0) >= SQL_VALIDATION_MAX_RETRIES:
                return "node_error"
            return "node_sql_generator"
        return "node_run_query"

    def fn_route_query(state: AgentState) -> str:
        if state.get("state_error_message"):
            return "node_error"
        if state.get("state_current_table"):
            if state.get("state_loaded_data"):
                return "node_visualization"
            return "node_table_results"
        if state.get("state_loaded_data") and state.get("state_intent") == "visualize":
            return "node_use_loaded_data"
        if state.get("state_loading_requested"):
            return "node_load_data"
        if state.get("state_intent") == "visualize" and state.get("state_loaded_data"):
            return "node_visualization"
        return "node_respond"

    def fn_route_visualization_next(state: AgentState) -> str:
        if state.get("state_current_table"):
            return "node_table_results"
        if state.get("state_loading_requested"):
            return "node_load_data"
        return "node_respond"

    def fn_route_table_progress(state: AgentState) -> str:
        if state.get("state_table_queue"):
            return "node_table_select"
        return "node_respond"

    def fn_route_s2w(state: AgentState) -> str:
        if state.get("state_error_message") or state.get("state_sql_validation_error"):
            return "node_error"
        return "node_sql_generator"

    def fn_route_limits(state: AgentState) -> str:
        if state.get("state_limit_only"):
            return "node_respond"
        if state.get("state_table_queue"):
            return "node_table_select"
        intent = state.get("state_intent", "simple_answer")
        if intent in {"visualize", "sql_query"}:
            return "node_s2w_tool"
        if intent == "clarify":
            return "node_clarify"
        return "node_respond"

    workflow.add_node("node_extract_user", fn_extract_user_query)
    workflow.add_node("node_configure_limits", fn_configure_limits)
    workflow.add_node("node_intent_classifier", fn_classify_intent)
    workflow.add_node("node_s2w_tool", fn_prepare_s2w_tool_context)
    workflow.add_node("node_table_select", fn_select_next_table)
    workflow.add_node("node_table_sql", fn_build_table_sql)
    workflow.add_node("node_sql_generator", fn_generate_sql)
    workflow.add_node("node_sql_validator", fn_validate_sql)
    workflow.add_node("node_run_query", fn_execute_sql)
    workflow.add_node("node_use_loaded_data", fn_use_loaded_data)
    workflow.add_node("node_load_data", fn_load_data)
    workflow.add_node("node_visualization", fn_plan_visualization)
    workflow.add_node("node_table_results", fn_collect_table_results)
    workflow.add_node("node_respond", fn_respond)
    workflow.add_node("node_clarify", fn_clarify)
    workflow.add_node("node_error", fn_handle_error)

    workflow.set_entry_point("node_extract_user")
    workflow.add_edge("node_extract_user", "node_intent_classifier")
    workflow.add_conditional_edges(
        "node_configure_limits",
        fn_route_limits,
        {
            "node_respond": "node_respond",
            "node_table_select": "node_table_select",
            "node_s2w_tool": "node_s2w_tool",
            "node_clarify": "node_clarify",
        },
    )
    workflow.add_conditional_edges(
        "node_intent_classifier",
        fn_route_intent,
        {
            "node_configure_limits": "node_configure_limits",
            "node_s2w_tool": "node_s2w_tool",
            "node_use_loaded_data": "node_use_loaded_data",
            "node_clarify": "node_clarify",
            "node_respond": "node_respond",
            "node_table_select": "node_table_select",
        },
    )
    workflow.add_conditional_edges(
        "node_s2w_tool",
        fn_route_s2w,
        {
            "node_sql_generator": "node_sql_generator",
            "node_error": "node_error",
        },
    )
    workflow.add_edge("node_table_select", "node_table_sql")
    workflow.add_edge("node_table_sql", "node_run_query")
    workflow.add_edge("node_sql_generator", "node_sql_validator")
    workflow.add_conditional_edges(
        "node_sql_validator",
        fn_route_validation,
        {
            "node_sql_generator": "node_sql_generator",
            "node_run_query": "node_run_query",
            "node_error": "node_error",
        },
    )
    workflow.add_conditional_edges(
        "node_run_query",
        fn_route_query,
        {
            "node_table_results": "node_table_results",
            "node_visualization": "node_visualization",
            "node_load_data": "node_load_data",
            "node_use_loaded_data": "node_use_loaded_data",
            "node_respond": "node_respond",
            "node_error": "node_error",
        },
    )
    workflow.add_conditional_edges(
        "node_visualization",
        fn_route_visualization_next,
        {
            "node_table_results": "node_table_results",
            "node_use_loaded_data": "node_use_loaded_data",
            "node_respond": "node_respond",
        },
    )
    workflow.add_conditional_edges(
        "node_table_results",
        fn_route_table_progress,
        {
            "node_table_select": "node_table_select",
            "node_respond": "node_respond",
        },
    )
    workflow.add_edge("node_clarify", END)
    workflow.add_edge("node_respond", END)
    workflow.add_edge("node_use_loaded_data", "node_visualization")
    workflow.add_edge("node_load_data", END)
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
