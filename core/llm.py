"""LLM helpers and LangGraph utilities shared across examples."""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Literal, TypedDict

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv():
        return None

from langgraph.graph import END, StateGraph

from core.databricks import run_sql_query

load_dotenv()

SUMMARY_SYSTEM_PROMPT = "You are a concise assistant who summarizes topics clearly."
MessageRole = Literal["system", "user", "assistant"]
DEFAULT_PROVIDER = (os.environ.get("LLM_PROVIDER") or "google").lower()


class ChatMessage(TypedDict):
    role: MessageRole
    content: str


AgentIntent = Literal["visualize", "sql_query", "simple_answer", "clarify"]


class AgentState(TypedDict, total=False):
    messages: List[ChatMessage]
    table_metadata: Dict[str, Any]
    table_sequence: List[str]
    table_queue: List[str]
    current_table: str
    table_outputs: List[Dict[str, Any]]
    visualization_blocks: List[str]
    user_query: str
    intent: AgentIntent
    sql_query: str
    sql_validation_error: str
    sql_retry_count: int
    query_result: List[Dict[str, Any]]
    visualization_code: str
    response: str
    error_message: str
    node_path: List[str]


def _ensure_key(var_name: str, provider_label: str) -> str:
    api_key = os.environ.get(var_name)
    if not api_key:
        raise RuntimeError(f"Set {var_name} before using the {provider_label} provider.")
    return api_key


def _call_llm(provider: str, system_prompt: str, user_prompt: str) -> str:
    messages: List[ChatMessage] = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
    reply = _invoke_provider(messages, provider)
    return reply["content"]


def _format_metadata(table_metadata: Dict[str, Any] | None) -> str:
    if not table_metadata:
        return "No schema information available."
    lines: List[str] = []
    for table_name, data in table_metadata.items():
        if not isinstance(data, dict):
            lines.append(f"{table_name}: (metadata unavailable)")
            continue
        columns = data.get("columns") or []
        column_text = ", ".join(
            f"{column.get('name')} {column.get('type')}" for column in columns if column.get("name")
        )
        full_name = data.get("full_name")
        qualified = full_name or table_name
        lines.append(f"{qualified}: {column_text}")
    return "\n".join(lines)


def _safe_json_dumps(value: Any) -> str:
    def _default(obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return str(obj)

    return json.dumps(value, ensure_ascii=False, default=_default)


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


def _requests_all_tables(query: str) -> bool:
    normalized = (query or "").lower()
    return any(keyword in normalized for keyword in MULTI_TABLE_KEYWORDS)


def _resolve_table_sequence(
    requested: List[str] | None,
    query: str,
    table_metadata: Dict[str, Any] | None,
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
        return _dedup(requested)[:MAX_MULTI_TABLES]

    normalized_query = (query or "").lower()
    hits = []
    for lowered, actual in normalized_map.items():
        index = normalized_query.find(lowered)
        if index != -1:
            hits.append((index, actual))
    if hits:
        hits.sort()
        return [name for _, name in hits][:MAX_MULTI_TABLES]
    if _requests_all_tables(normalized_query):
        return list(table_metadata.keys())[:MAX_MULTI_TABLES]
    return []


def _table_full_name(table_name: str, metadata: Dict[str, Any] | None) -> str:
    full_name = (metadata or {}).get("full_name")
    if full_name:
        return full_name
    return table_name


_CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*([\s\S]+?)```", re.IGNORECASE)
SQL_DANGER_PATTERN = re.compile(r"(?i)\b(drop|delete|alter|update|insert|merge|truncate)\b")
SQL_SELECT_PATTERN = re.compile(r"(?i)\bselect\b")
SQL_VALIDATION_MAX_RETRIES = 2


def _strip_code_block(text: str) -> str:
    match = _CODE_FENCE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def _node_path_diagram(node_path: List[str]) -> str:
    if not node_path:
        return ""
    arrow_line = " \u2192 ".join(node_path)
    lines = [
        "digraph {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded,filled", fillcolor="#EEF3FF"];',
    ]
    node_ids: List[str] = []
    for idx, node_name in enumerate(node_path):
        node_id = f"n{idx}"
        node_ids.append(node_id)
        safe_label = node_name.replace('"', r"\"")
        lines.append(f'  {node_id} [label="{safe_label}"];')
    for left, right in zip(node_ids, node_ids[1:]):
        lines.append(f"  {left} -> {right};")
    lines.append("}")
    diagram = "\n".join(lines)
    return (
        f"\n\n사용된 LangGraph 경로: {arrow_line}\n"
        f"```dot\n{diagram}\n```"
    )


def _count_select_columns(statement: str) -> int:
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


def _validate_sql_statement(statement: str) -> str | None:
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
    column_count = _count_select_columns(normalized)
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
    return None


def _openai_response(messages: List[ChatMessage]) -> str:
    from openai import OpenAI

    api_key = _ensure_key("OPENAI_API_KEY", "OpenAI")
    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
        temperature=0.3,
    )
    return completion.choices[0].message.content or ""


def _azure_response(messages: List[ChatMessage]) -> str:
    from openai import AzureOpenAI

    api_key = _ensure_key("AZURE_OPENAI_API_KEY", "Azure OpenAI")
    endpoint = _ensure_key("AZURE_OPENAI_ENDPOINT", "Azure OpenAI")
    deployment = _ensure_key("AZURE_OPENAI_DEPLOYMENT", "Azure OpenAI")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
    completion = client.chat.completions.create(
        model=deployment,
        messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
        temperature=0.3,
    )
    return completion.choices[0].message.content or ""


def _google_response(messages: List[ChatMessage]) -> str:
    import google.generativeai as genai

    api_key = _ensure_key("GOOGLE_API_KEY", "Google Gemini")
    genai.configure(api_key=api_key)
    model_name = os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash")
    system_instruction = None
    conversation = []
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
            continue
        role = "user" if msg["role"] == "user" else "model"
        conversation.append({"role": role, "parts": [msg["content"]]})

    model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
    response = model.generate_content(conversation)
    if hasattr(response, "text") and response.text:
        return response.text
    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content.parts:
            part = candidate.content.parts[0]
            return getattr(part, "text", str(part))
    return ""


def _invoke_provider(messages: List[ChatMessage], provider: str) -> ChatMessage:
    if provider == "google":
        content = _google_response(messages)
    elif provider == "azure":
        content = _azure_response(messages)
    else:
        content = _openai_response(messages)
    return {"role": "assistant", "content": content.strip()}


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
        path = list(state.get("node_path", []))
        path.append(node_name)
        updates["node_path"] = path
        return updates

    def extract_user_query(state: AgentState) -> AgentState:
        query = _latest_user_query(state.get("messages", []))
        return with_path(state, "extract_user", {"user_query": query})

    def prepare_table_sequence(state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        metadata = state.get("table_metadata") or {}
        requested = state.get("table_sequence")
        sequence = _resolve_table_sequence(requested, query, metadata)
        updates: AgentState = {
            "table_queue": sequence,
            "table_outputs": [],
            "visualization_blocks": [],
        }
        return with_path(state, "prepare_tables", updates)

    def classify_intent(state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        if not query:
            return with_path(state, "intent", {"intent": "simple_answer"})
        system_prompt = (
            "You categorize user requests for a Databricks analytics assistant. "
            "Valid intents: VISUALIZE (asks for charts/plots), SQL_QUERY (asks for tabular data), "
            "CLARIFY (question is ambiguous or missing info), SIMPLE_ANSWER (general question "
            "answerable without running SQL). Reply with one label only."
        )
        user_prompt = f"User query:\n{query}\n\nIntent label:"
        intent_text = _call_llm(selected, system_prompt, user_prompt).strip().lower()
        if "visual" in intent_text:
            intent: AgentIntent = "visualize"
        elif "sql" in intent_text or "data" in intent_text:
            intent = "sql_query"
        elif "clarify" in intent_text or "clarification" in intent_text:
            intent = "clarify"
        else:
            intent = "simple_answer"
        return with_path(state, "intent", {"intent": intent})

    def generate_sql(state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        metadata = _format_metadata(state.get("table_metadata"))
        feedback = state.get("sql_validation_error")
        retry_count = state.get("sql_retry_count", 0)
        system_prompt = (
            "You are a Databricks SQL expert. Generate safe SQL answers that only read data. "
            "Always include an explicit LIMIT <= 500 if the query might return many rows. "
            "Table schemas below show fully-qualified names already wrapped in backticks. "
            "When referencing those tables, copy the entire qualified name verbatim."
        )
        user_prompt = (
            f"Schema information:\n{metadata}\n\nUser request:\n{query}\n\n"
            "Return only the SQL query."
        )
        if feedback:
            user_prompt += (
                "\n\nA previous attempt failed automatic validation for this reason:\n"
                f"{feedback}\n"
                "Revise the SQL accordingly and ensure only one SELECT statement is returned."
            )
        if retry_count:
            user_prompt += f"\n(Attempt #{retry_count + 1} — pay extra attention to syntax.)"
        raw_sql = _call_llm(selected, system_prompt, user_prompt).strip()
        sql = _strip_code_block(raw_sql)
        return with_path(state, "sql_generator", {"sql_query": sql})

    def select_next_table(state: AgentState) -> AgentState:
        queue = list(state.get("table_queue") or [])
        updates: AgentState = {}
        if not queue:
            updates["current_table"] = ""
            return with_path(state, "table_select", updates)
        current = queue.pop(0)
        updates["current_table"] = current
        updates["table_queue"] = queue
        updates["sql_query"] = ""
        updates["sql_validation_error"] = ""
        updates["sql_retry_count"] = 0
        return with_path(state, "table_select", updates)

    def build_table_sql(state: AgentState) -> AgentState:
        table_name = state.get("current_table")
        if not table_name:
            return with_path(
                state, "table_sql", {"error_message": "순차적으로 조회할 테이블이 없습니다."}
            )
        metadata = (state.get("table_metadata") or {}).get(table_name, {})
        limit = metadata.get("preview_limit") or MULTI_TABLE_SAMPLE_LIMIT
        if not isinstance(limit, int) or limit <= 0:
            limit = MULTI_TABLE_SAMPLE_LIMIT
        full_name = _table_full_name(table_name, metadata)
        statement = f"SELECT * FROM {full_name} LIMIT {limit}"
        return with_path(state, "table_sql", {"sql_query": statement})

    def validate_sql(state: AgentState) -> AgentState:
        statement = state.get("sql_query", "")
        validation_error = _validate_sql_statement(statement)
        retries = state.get("sql_retry_count", 0)
        updates: AgentState = {}
        if validation_error:
            retries += 1
            updates["sql_validation_error"] = validation_error
            updates["sql_retry_count"] = retries
            if retries >= SQL_VALIDATION_MAX_RETRIES:
                updates["error_message"] = f"SQL 검증 실패: {validation_error}"
        else:
            updates["sql_validation_error"] = ""
        return with_path(state, "sql_validator", updates)

    def execute_sql(state: AgentState) -> AgentState:
        statement = state.get("sql_query", "")
        if not statement:
            return with_path(state, "run_query", {"error_message": "SQL 문을 생성하지 못했습니다."})
        try:
            _columns, rows = run_sql_query(statement)
        except Exception as exc:  # pragma: no cover - depends on environment
            return with_path(
                state, "run_query", {"error_message": str(exc), "query_result": []}
            )
        return with_path(state, "run_query", {"query_result": rows})

    def plan_visualization(state: AgentState) -> AgentState:
        rows = state.get("query_result", []) or []
        sample = rows[:50]
        data_json = _safe_json_dumps(sample)
        user_query = state.get("user_query", "")
        system_prompt = (
            "You create Matplotlib visualization code for tabular data. "
            "Always import matplotlib.pyplot as plt and use pandas as pd to build a DataFrame "
            "from the provided rows. Close with plt.tight_layout(). "
            "Return only a Python code block."
        )
        table_name = state.get("current_table")
        table_context = f"Current table: {table_name}\n" if table_name else ""
        user_prompt = (
            f"{table_context}User request:\n{user_query}\n\nRows JSON:\n{data_json}\n\n"
            "Generate concise visualization code."
        )
        code = _call_llm(selected, system_prompt, user_prompt).strip()
        if "```" not in code:
            code = f"```python\n{code}\n```"
        return with_path(state, "visualization", {"visualization_code": code})

    def respond(state: AgentState) -> AgentState:
        intent = state.get("intent", "simple_answer")
        user_query = state.get("user_query", "")
        table_outputs = state.get("table_outputs") or []
        visualization_blocks = state.get("visualization_blocks") or []
        if table_outputs:
            previews_json = _safe_json_dumps(table_outputs)
            system_prompt = (
                "You review sequential table previews. Summarize each table in order, "
                "highlighting row counts and notable columns. Keep the tone concise and "
                "respond in Korean."
            )
            user_prompt = (
                f"User query: {user_query}\nIntent: {intent}\n"
                f"Table previews JSON:\n{previews_json}\n\n"
                "Write sectioned summaries (one per table)."
            )
            reply = _call_llm(selected, system_prompt, user_prompt).strip()
            final_text = reply
            if visualization_blocks:
                blocks = "\n\n".join(visualization_blocks)
                final_text = f"{reply}\n\n{blocks}"
        else:
            rows = state.get("query_result", []) or []
            sample = rows[:20]
            data_json = _safe_json_dumps(sample)
            sql_query = state.get("sql_query", "")
            system_prompt = (
                "You are a helpful analytics assistant. Summarize query results clearly. "
                "If no data is available, respond conversationally using general knowledge."
            )
            user_prompt = (
                f"Intent: {intent}\nUser query: {user_query}\n"
                f"SQL: {sql_query or 'N/A'}\nData preview JSON:\n{data_json}\n\n"
                "Write the final response in Korean, summarizing insights. "
                "Mention charts if visualization code is provided."
            )
            reply = _call_llm(selected, system_prompt, user_prompt).strip()
            final_text = reply
            visualization_code = state.get("visualization_code")
            if visualization_code:
                final_text = f"{reply}\n\n{visualization_code}"
        response_path = list(state.get("node_path", [])) + ["response"]
        final_text = f"{final_text}{_node_path_diagram(response_path)}"
        messages = state.get("messages", [])
        return with_path(
            state,
            "response",
            {
                "response": final_text,
                "messages": messages + [{"role": "assistant", "content": final_text}],
            },
        )

    def clarify(state: AgentState) -> AgentState:
        user_query = state.get("user_query", "")
        system_prompt = (
            "You need more detail from the user before running SQL. "
            "Ask a short follow-up question in Korean."
        )
        user_prompt = f"User query:\n{user_query}\n\nAsk what detail is missing."
        reply = _call_llm(selected, system_prompt, user_prompt).strip()
        messages = state.get("messages", [])
        clarify_path = list(state.get("node_path", [])) + ["clarify"]
        final_text = f"{reply}{_node_path_diagram(clarify_path)}"
        return with_path(
            state,
            "clarify",
            {
                "response": final_text,
                "messages": messages + [{"role": "assistant", "content": final_text}],
            },
        )

    def handle_error(state: AgentState) -> AgentState:
        error_message = state.get("error_message", "알 수 없는 오류가 발생했습니다.")
        text = (
            "⚠️ Databricks 쿼리를 실행하는 동안 오류가 발생했습니다.\n"
            f"세부 정보: {error_message}\n다시 시도하거나 조건을 조정해 주세요."
        )
        messages = state.get("messages", [])
        error_path = list(state.get("node_path", [])) + ["error"]
        final_text = f"{text}{_node_path_diagram(error_path)}"
        return with_path(
            state,
            "error",
            {
                "response": final_text,
                "messages": messages + [{"role": "assistant", "content": final_text}],
            },
        )

    def collect_table_results(state: AgentState) -> AgentState:
        table_name = state.get("current_table", "")
        rows = state.get("query_result", []) or []
        visualization_code = state.get("visualization_code", "")
        outputs = list(state.get("table_outputs") or [])
        if table_name:
            entry = {
                "table": table_name,
                "sql": state.get("sql_query", ""),
                "row_count": len(rows),
                "rows": rows[:20],
            }
            outputs.append(entry)
        viz_blocks = list(state.get("visualization_blocks") or [])
        if visualization_code:
            labeled_code = visualization_code
            if table_name:
                labeled_code = f"**{table_name}** 테이블 시각화\n\n{visualization_code}"
            viz_blocks.append(labeled_code)
        updates: AgentState = {
            "table_outputs": outputs,
            "visualization_blocks": viz_blocks,
            "visualization_code": "",
            "current_table": "",
        }
        return with_path(state, "table_results", updates)

    def route_intent(state: AgentState) -> str:
        if state.get("table_queue"):
            return "table_select"
        intent = state.get("intent", "simple_answer")
        if intent in {"visualize", "sql_query"}:
            return "sql_generator"
        if intent == "clarify":
            return "clarify"
        return "response"

    def route_validation(state: AgentState) -> str:
        if state.get("sql_validation_error"):
            if state.get("sql_retry_count", 0) >= SQL_VALIDATION_MAX_RETRIES:
                return "error"
            return "sql_generator"
        return "run_query"

    def route_query(state: AgentState) -> str:
        if state.get("error_message"):
            return "error"
        if state.get("current_table"):
            if state.get("query_result"):
                return "visualization"
            return "table_results"
        if state.get("intent") == "visualize" and state.get("query_result"):
            return "visualization"
        return "response"

    def route_visualization_next(state: AgentState) -> str:
        if state.get("current_table"):
            return "table_results"
        return "response"

    def route_table_progress(state: AgentState) -> str:
        if state.get("table_queue"):
            return "table_select"
        return "response"

    workflow.add_node("extract_user", extract_user_query)
    workflow.add_node("prepare_tables", prepare_table_sequence)
    workflow.add_node("intent", classify_intent)
    workflow.add_node("table_select", select_next_table)
    workflow.add_node("table_sql", build_table_sql)
    workflow.add_node("sql_generator", generate_sql)
    workflow.add_node("sql_validator", validate_sql)
    workflow.add_node("run_query", execute_sql)
    workflow.add_node("visualization", plan_visualization)
    workflow.add_node("table_results", collect_table_results)
    workflow.add_node("response", respond)
    workflow.add_node("clarify", clarify)
    workflow.add_node("error", handle_error)

    workflow.set_entry_point("extract_user")
    workflow.add_edge("extract_user", "prepare_tables")
    workflow.add_edge("prepare_tables", "intent")
    workflow.add_conditional_edges(
        "intent",
        route_intent,
        {
            "sql_generator": "sql_generator",
            "clarify": "clarify",
            "response": "response",
            "table_select": "table_select",
        },
    )
    workflow.add_edge("table_select", "table_sql")
    workflow.add_edge("table_sql", "run_query")
    workflow.add_edge("sql_generator", "sql_validator")
    workflow.add_conditional_edges(
        "sql_validator",
        route_validation,
        {
            "sql_generator": "sql_generator",
            "run_query": "run_query",
            "error": "error",
        },
    )
    workflow.add_conditional_edges(
        "run_query",
        route_query,
        {
            "table_results": "table_results",
            "visualization": "visualization",
            "response": "response",
            "error": "error",
        },
    )
    workflow.add_conditional_edges(
        "visualization",
        route_visualization_next,
        {
            "table_results": "table_results",
            "response": "response",
        },
    )
    workflow.add_conditional_edges(
        "table_results",
        route_table_progress,
        {
            "table_select": "table_select",
            "response": "response",
        },
    )
    workflow.add_edge("clarify", END)
    workflow.add_edge("response", END)
    workflow.add_edge("error", END)
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
    result = graph.invoke({"messages": summary_seed(topic)})
    return result["messages"][-1]["content"]
