"""Streamlit chatbot UI backed by the shared LangGraph helpers."""
from __future__ import annotations

import os
import re
import json
import asyncio
from typing import Any, Dict, Iterable, List, Optional, Tuple

import streamlit as st

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv():
        return None

from core.databricks import fetch_table_metadata, fetch_table_preview, list_tables
from core.llm import ChatMessage, DEFAULT_SQL_LIMIT
from core.graph import build_conversation_graph
from app.handlers import _handle_visualization_blocks, _render_visualizations

load_dotenv()


SYSTEM_PROMPT = "You are a helpful assistant who keeps answers concise and clear."
# To avoid exceeding provider context limits we only send the most recent
# portion of the conversation to the LLM. These limits are intentionally
# conservative because downstream prompts also include table previews and
# node_visualization code.
MAX_HISTORY_MESSAGES = 12
MAX_MESSAGE_CHARS = 4000
TABLE_PREVIEW_LIMIT = 50
TABLE_LOAD_PATTERN = re.compile(
    r"(?P<table>[A-Za-z0-9_]+)\s+table\s+(?:데이터|데이타)\s*로딩(?:해줘|해줘요|해줘라|해줘봐|해줄래|해줄수있어|해|줘)?",
    re.IGNORECASE,
)
LIMIT_DIRECTIVE_PATTERN = re.compile(r"%limit\s+(\d+)", re.IGNORECASE)
# Bump this to force Streamlit to rebuild the LangGraph when graph logic changes.
GRAPH_BUILD_ID = "2025-03-02-b"
LANGGRAPH_NODES = {
    "node_ingest": "질문 추출 + %limit 반영 + 의도 분류",
    "node_plan_tables": "테이블 큐/스키마 컨텍스트 준비",
    "node_prepare_sql": "SQL 생성+검증(재시도 포함)",
    "node_run_query": "Databricks 쿼리 실행 및 누적",
    "node_visualization": "시각화 코드 생성 및 누적",
    "node_respond": "최종 답변/요약/추가질문",
    "node_error": "쿼리 실패 등 오류 안내",
}
LANGGRAPH_DOT = """
digraph {
    rankdir=LR;
    node [shape=rectangle, style=rounded];
    node_ingest -> node_plan_tables [label="visualize/sql/loading"];
    node_ingest -> node_respond [label="simple/%limit/clarify"];
    node_plan_tables -> node_prepare_sql;
    node_plan_tables -> node_error [label="meta error"];
    node_prepare_sql -> node_run_query [label="pass"];
    node_prepare_sql -> node_error [label="max fail"];
    node_run_query -> node_visualization [label="visualize"];
    node_run_query -> node_plan_tables [label="multi-next"];
    node_run_query -> node_respond [label="sql/loading"];
    node_run_query -> node_error [label="runtime"];
    node_visualization -> node_plan_tables [label="multi-next"];
    node_visualization -> node_respond;
    node_respond -> end;
    node_error -> end;
}
"""


def parse_debug_command(user_input: str) -> Optional[bool]:
    """Return True/False if %debug on/off is requested, else None."""

    if not user_input:
        return None
    normalized = user_input.strip().lower()
    if normalized == "%debug on":
        return True
    if normalized == "%debug off":
        return False
    return None


def render_debug_sidebar():
    """Render sidebar debug container when debug_mode is enabled."""

    if not st.session_state.get("debug_mode"):
        return None
    with st.sidebar:
        st.markdown("### 🔍 Debug Events")
        return st.container()


def run_async(coro):
    """Run coroutine regardless of existing event loop state."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        return asyncio.ensure_future(coro)
    return asyncio.run(coro)


async def run_graph_with_events(app, query, debug_box):
    """Execute graph with event streaming and surface events to sidebar."""

    final_state = None
    async for event in app.astream_events(query, version="v1"):
        event_type = event.get("type") or event.get("event")
        data = event.get("data") or {}
        if debug_box:
            if event_type == "node.started":
                debug_box.write(f"🟡 Node Started: {data.get('name') or data.get('node')}")
            elif event_type == "node.completed":
                debug_box.write(f"🟢 Node Completed: {data.get('name') or data.get('node')}")
            elif event_type == "llm.streaming.chunk":
                token = data.get("chunk") or data.get("token") or data.get("text") or data
                debug_box.write(f"✏️ Token: {token}")
            elif event_type == "tool.started":
                debug_box.write(f"🔧 Tool Started: {data.get('name') or data.get('tool')}")
            elif event_type == "tool.completed":
                debug_box.write(f"🔨 Tool Completed: {data.get('name') or data.get('tool')}")
            elif event_type == "state.diff":
                debug_box.write("🧩 State Diff:")
                try:
                    debug_box.json(data)
                except Exception:
                    debug_box.write(data)
            elif event_type == "interrupt":
                debug_box.write(f"⏸️ Interrupt: {data}")
        if event_type == "interrupt":
            final_state = {"interrupt": data}
            break
        if event_type in {"node.completed", "graph.completed", "graph.end"}:
            if isinstance(data, dict):
                final_state = data.get("output") or data.get("state") or final_state
    if final_state is None:
        final_state = await app.ainvoke(query)
    return final_state


def _find_table_references(prompt: str, candidates: Iterable[str]) -> List[str]:
    """Return all candidate tables mentioned in the prompt (case-insensitive)."""

    normalized_prompt = prompt.lower()
    matches: List[Tuple[int, str]] = []
    for table in candidates:
        pattern = re.compile(rf"(?<!\w){re.escape(table.lower())}(?!\w)")
        match = pattern.search(normalized_prompt)
        if match:
            matches.append((match.start(), table))
    matches.sort(key=lambda item: item[0])
    return [table for _pos, table in matches]


def _store_node_path(message_index: int, path: List[str]) -> None:
    if not path:
        return
    traces = st.session_state.setdefault("node_paths", {})
    traces[message_index] = path


def _render_node_flow(message_index: int) -> None:
    traces = st.session_state.get("node_paths", {})
    path = traces.get(message_index)
    if not path:
        return
    lines = [
        "digraph {",
        "rankdir=LR;",
        'node [shape=rectangle, style="rounded,filled", fillcolor="#EEF3FF"];',
    ]
    node_ids: List[str] = []
    for idx, node_name in enumerate(path):
        node_id = f"node_{message_index}_{idx}"
        node_ids.append(node_id)
        description = LANGGRAPH_NODES.get(node_name, "")
        label_description = description.replace('"', r"\"")
        label_name = node_name.replace('"', r"\"")
        label = label_name
        if label_description:
            label = f"{label_name}\\n{label_description}"
        lines.append(f'{node_id} [label="{label}"];')
    for left, right in zip(node_ids, node_ids[1:]):
        lines.append(f"{left} -> {right};")
    lines.append("}")
    st.graphviz_chart("\n".join(lines))


def _truncate_message_content(content: str) -> str:
    """Trim very long message text while indicating truncation."""

    if len(content) <= MAX_MESSAGE_CHARS:
        return content
    notice = "(이전 대화 일부는 길이 제한으로 생략되었습니다.)\n"
    return f"{notice}{content[-MAX_MESSAGE_CHARS:]}"


def _lc_messages(history: List[Tuple[str, str]]) -> List[ChatMessage]:
    """Convert (role, content) tuples to LangGraph-compatible message dicts."""

    messages: List[ChatMessage] = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent_history = history[-MAX_HISTORY_MESSAGES:]
    for role, content in recent_history:
        messages.append({"role": role, "content": _truncate_message_content(content)})
    return messages


def _build_graph_input(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the baseline graph input with optional overrides."""

    payload = {
        "state_messages": _lc_messages(st.session_state.history),
        "state_table_metadata": st.session_state.get("state_table_metadata", {}),
        "state_sql_limit": st.session_state.get("state_sql_limit", DEFAULT_SQL_LIMIT),
        "state_loaded_data": st.session_state.get("state_loaded_data"),
        "state_loaded_columns": st.session_state.get("state_loaded_columns"),
        "state_loaded_table": st.session_state.get("state_loaded_table"),
        "state_active_table_metadata": st.session_state.get("state_active_table_metadata"),
        "state_sql_query": st.session_state.get("state_sql_query", ""),
        "state_visualization_code": st.session_state.get("state_visualization_code", ""),
        "state_sql_approved": st.session_state.get("state_sql_approved", False),
        "state_viz_approved": st.session_state.get("state_viz_approved", False),
    }
    if extra:
        payload.update(extra)
    return payload


def _handle_response_state(
    response_state: Dict[str, Any] | None, assistant_segments: List[str]
) -> None:
    """Render response/interrupt results, update history, and cache viz data."""

    if isinstance(response_state, dict) and "interrupt" in response_state:
        interrupt = response_state["interrupt"] or {}
        st.session_state.pending_interrupt = interrupt
        name = interrupt.get("name") or interrupt.get("type") or "승인 요청"
        response_text = f"⏸️ {name} 승인이 필요합니다. 아래 승인 폼을 완료해 주세요."
        node_path: List[str] = []
    elif response_state:
        latest = response_state["state_messages"][-1]
        response_text = latest["content"]
        node_path = response_state.get("state_node_path", [])
        st.session_state.state_sql_limit = response_state.get("state_sql_limit", DEFAULT_SQL_LIMIT)
    else:
        response_text = "⚠️ 알 수 없는 오류가 발생했습니다."
        node_path = []

    assistant_segments.append(response_text)
    combined_text = "\n\n".join(segment for segment in assistant_segments if segment)
    st.session_state.history.append(("assistant", combined_text))
    message_index = len(st.session_state.history) - 1
    viz_rows = st.session_state.setdefault("viz_rows", {})
    viz_rows[message_index] = response_state.get("state_loaded_data") if response_state else []
    viz_tables = st.session_state.setdefault("viz_table_outputs", {})
    viz_tables[message_index] = response_state.get("state_table_outputs") if response_state else []
    if response_state and "interrupt" not in response_state:
        st.session_state.state_loaded_data = response_state.get("state_loaded_data")
        st.session_state.state_loaded_columns = response_state.get("state_loaded_columns")
        st.session_state.state_loaded_table = response_state.get("state_loaded_table")
        st.session_state.state_active_table_metadata = response_state.get("state_active_table_metadata")
        st.session_state.state_sql_query = response_state.get("state_sql_query", "")
        st.session_state.state_visualization_code = response_state.get("state_visualization_code", "")
        st.session_state.state_sql_approved = response_state.get("state_sql_approved", False)
        st.session_state.state_viz_approved = response_state.get("state_viz_approved", False)
    if node_path and node_path[-1] == "node_error":
        st.error(response_text)
    else:
        st.markdown(response_text)
    _handle_visualization_blocks(response_text, message_index)
    _store_node_path(message_index, node_path)
    _render_node_flow(message_index)


def main() -> None:
    st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬")
    st.title("LangGraph Chatbot")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "visualizations" not in st.session_state:
        st.session_state.visualizations = {}
    if "visualization_messages" not in st.session_state:
        st.session_state.visualization_messages = {}
    if "state_table_metadata" not in st.session_state:
        st.session_state.state_table_metadata = {}
    if "state_active_table_metadata" not in st.session_state:
        st.session_state.state_active_table_metadata = {}
    if "state_loaded_data" not in st.session_state:
        st.session_state.state_loaded_data = []
    if "state_loaded_columns" not in st.session_state:
        st.session_state.state_loaded_columns = []
    if "state_loaded_table" not in st.session_state:
        st.session_state.state_loaded_table = ""
    if "state_sql_query" not in st.session_state:
        st.session_state.state_sql_query = ""
    if "state_visualization_code" not in st.session_state:
        st.session_state.state_visualization_code = ""
    if "state_sql_approved" not in st.session_state:
        st.session_state.state_sql_approved = False
    if "state_viz_approved" not in st.session_state:
        st.session_state.state_viz_approved = False
    if "pending_interrupt" not in st.session_state:
        st.session_state.pending_interrupt = None
    if "node_paths" not in st.session_state:
        st.session_state.node_paths = {}
    if "state_sql_limit" not in st.session_state:
        st.session_state.state_sql_limit = DEFAULT_SQL_LIMIT
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    with st.sidebar:
        provider = os.environ.get("LLM_PROVIDER", "google").lower()
        if provider not in {"google", "openai", "azure"}:
            provider = "google"
        st.markdown(
            f"**LLM Provider:** `{provider}` (set `LLM_PROVIDER` in your environment to change)"
        )
        st.caption(
            "Configure the matching API settings: `GOOGLE_API_KEY` (and optional "
            "`GOOGLE_MODEL`), or `OPENAI_API_KEY` (plus optional `OPENAI_MODEL`), or "
            "Azure values `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, "
            "`AZURE_OPENAI_DEPLOYMENT` (plus optional `AZURE_OPENAI_API_VERSION`)."
        )
        if (
            "graph" not in st.session_state
            or st.session_state.get("graph_provider") != provider
            or st.session_state.get("graph_build_id") != GRAPH_BUILD_ID
        ):
            st.session_state.graph = build_conversation_graph(provider=provider)
            st.session_state.graph_provider = provider
            st.session_state.graph_build_id = GRAPH_BUILD_ID

        with st.expander("Databricks Tables", expanded=True):
            refresh_requested = st.button("Refresh tables", use_container_width=True)
            if refresh_requested:
                st.session_state.pop("tables", None)
                st.session_state.pop("tables_error", None)
                st.session_state.pop("state_table_metadata", None)
            if "tables" not in st.session_state and "tables_error" not in st.session_state:
                try:
                    st.session_state.tables = list_tables()
                    metadata: Dict[str, Dict[str, Any]] = {}
                    for name in st.session_state.tables:
                        try:
                            metadata[name] = fetch_table_metadata(name)
                        except Exception as meta_exc:
                            metadata[name] = {"node_error": str(meta_exc)}
                    st.session_state.state_table_metadata = metadata
                except Exception as exc:  # pragma: no cover - UI path
                    st.session_state.tables_error = str(exc)

            tables_error = st.session_state.get("tables_error")
            tables = st.session_state.get("tables", [])

            if tables_error:
                st.warning(tables_error)
            elif tables:
                table_metadata = st.session_state.get("state_table_metadata", {})
                selected_table = st.selectbox(
                    "테이블 선택",
                    tables,
                    key="sidebar_table_select",
                )
                if selected_table:
                    metadata = table_metadata.get(selected_table, {})
                    full_name = metadata.get("full_name")
                    if full_name:
                        st.caption(f"식별자: {full_name}")
                    columns = metadata.get("columns") or []
                    if columns:
                        column_names = [col.get("name", "") for col in columns if col.get("name")]
                        column_display = [
                            f"{col.get('name', '')} ({col.get('type', '')})"
                            for col in columns
                            if col.get("name")
                        ]
                        if column_names:
                            display_map = dict(zip(column_names, column_display))
                            selected_col = st.selectbox(
                                "컬럼 선택",
                                column_names,
                                format_func=lambda name: display_map.get(name, name),
                                key=f"sidebar_column_select_{selected_table}",
                            )
                            if selected_col:
                                st.caption(f"선택된 컬럼: {selected_col}")
                        else:
                            st.info("표시할 컬럼 이름이 없습니다.")
                    elif "node_error" in metadata:
                        st.caption(f"⚠️ {metadata['node_error']}")
                    else:
                        st.info("컬럼 정보를 가져올 수 없습니다.")
            else:
                st.info("No tables available.")

        with st.expander("LangGraph Flow", expanded=False):
            st.caption("현재 에이전트 그래프 노드 및 동작 요약")
            for node_key, description in LANGGRAPH_NODES.items():
                st.markdown(f"- **{node_key}**: {description}")
            st.graphviz_chart(LANGGRAPH_DOT)

    debug_box = render_debug_sidebar()

    for index, (role, content) in enumerate(st.session_state.history):
        with st.chat_message(role):
            st.markdown(content)
            _render_visualizations(index)
            _render_node_flow(index)

    pending_interrupt = st.session_state.get("pending_interrupt")
    if pending_interrupt:
        interrupt_name = (
            pending_interrupt.get("name")
            or pending_interrupt.get("type")
            or pending_interrupt.get("event")
            or "승인 요청"
        )
        interrupt_payload = pending_interrupt.get("data") or {}
        with st.expander(f"⏸️ {interrupt_name} 승인 필요", expanded=True):
            st.caption("그래프가 사용자 승인 대기 중입니다. 승인/수정/거절을 선택하세요.")
            if interrupt_name == "approve_sql":
                sql_text = interrupt_payload.get("sql", "")
                st.code(sql_text or "SQL 없음", language="sql")
                with st.form("approve_sql_form"):
                    decision = st.radio(
                        "SQL 실행 여부",
                        ("승인", "수정 후 승인", "거절"),
                        index=0,
                        horizontal=True,
                    )
                    edited_sql = st.text_area("SQL 수정", value=sql_text, height=200)
                    submitted = st.form_submit_button("응답 전송")
                    if submitted:
                        if decision == "거절":
                            st.session_state.pending_interrupt = None
                            st.session_state.state_sql_approved = False
                            _handle_response_state(
                                {
                                    "state_messages": [
                                        {
                                            "role": "assistant",
                                            "content": "사용자가 SQL 실행을 거절했습니다. 요청을 다시 입력해 주세요.",
                                        }
                                    ]
                                },
                                [],
                            )
                        else:
                            approved_sql = sql_text if decision == "승인" else edited_sql.strip()
                            if not approved_sql:
                                st.warning("승인할 SQL이 없습니다.")
                            else:
                                st.session_state.pending_interrupt = None
                                st.session_state.state_sql_query = approved_sql
                                st.session_state.state_sql_approved = True
                                graph = st.session_state.graph
                                updates = {
                                    "state_sql_query": approved_sql,
                                    "state_sql_approved": True,
                                    "state_error_message": "",
                                    "state_sql_validation_error": "",
                                }
                                graph_input = _build_graph_input(updates)
                                resume_state = run_async(
                                    run_graph_with_events(graph, graph_input, debug_box)
                                )
                                if asyncio.isfuture(resume_state):
                                    resume_state = resume_state.result()
                                _handle_response_state(resume_state, [])
            elif interrupt_name == "approve_viz":
                code_text = interrupt_payload.get("code", "")
                st.code(code_text or "코드 없음", language="python")
                with st.form("approve_viz_form"):
                    decision = st.radio(
                        "시각화 코드 실행 여부",
                        ("승인", "거절"),
                        index=0,
                        horizontal=True,
                    )
                    submitted = st.form_submit_button("응답 전송")
                    if submitted:
                        if decision == "거절":
                            st.session_state.pending_interrupt = None
                            st.session_state.state_viz_approved = False
                            _handle_response_state(
                                {
                                    "state_messages": [
                                        {
                                            "role": "assistant",
                                            "content": "사용자가 시각화 실행을 거절했습니다. 다른 요청을 입력해 주세요.",
                                        }
                                    ]
                                },
                                [],
                            )
                        else:
                            st.session_state.pending_interrupt = None
                            wrapped = f"```python\n{code_text}\n```" if code_text else ""
                            st.session_state.state_visualization_code = wrapped
                            st.session_state.state_viz_approved = True
                            graph = st.session_state.graph
                            updates = {
                                "state_visualization_code": wrapped,
                                "state_viz_approved": True,
                            }
                            graph_input = _build_graph_input(updates)
                            resume_state = run_async(
                                run_graph_with_events(graph, graph_input, debug_box)
                            )
                            if asyncio.isfuture(resume_state):
                                resume_state = resume_state.result()
                            _handle_response_state(resume_state, [])

    if prompt := st.chat_input("Enter your message"):
        command_state = parse_debug_command(prompt)
        if command_state is not None:
            st.session_state.debug_mode = command_state
            status_text = "Debug mode enabled." if command_state else "Debug mode disabled."
            with st.chat_message("assistant"):
                st.markdown(status_text)
            st.session_state.history.append(("assistant", status_text))
            rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
            if rerun:
                rerun()

        st.session_state.history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        limit_override_match = LIMIT_DIRECTIVE_PATTERN.search(prompt)
        if limit_override_match:
            limit_value = int(limit_override_match.group(1))
            st.session_state.state_sql_limit = limit_value

        table_names: List[str] = []
        table_matches = [match.group("table") for match in TABLE_LOAD_PATTERN.finditer(prompt)]
        if table_matches:
            table_names = table_matches
        elif tables := st.session_state.get("tables"):
            table_names = _find_table_references(prompt, tables)

        if table_names:
            seen: set[str] = set()
            table_names = [name for name in table_names if not (name in seen or seen.add(name))]

        with st.chat_message("assistant"):
            assistant_segments: List[str] = []
            for table_name in table_names:
                try:
                    sql_statement, rows = fetch_table_preview(
                        table_name, limit=TABLE_PREVIEW_LIMIT
                    )
                    row_count = len(rows)
                    preview_text = (
                        f"Detected reference to `{table_name}` and loaded "
                        f"{row_count} row(s) (limit {TABLE_PREVIEW_LIMIT})."
                    )
                    assistant_segments.append(preview_text)
                    st.markdown(preview_text)
                    st.code(sql_statement, language="sql")
                    if rows:
                        st.dataframe(rows, use_container_width=True)
                    else:
                        st.info("No data returned for this table.")
                except Exception as exc:  # pragma: no cover - Streamlit surface
                    preview_text = f"⚠️ Error loading `{table_name}`: {exc}"
                    assistant_segments.append(preview_text)
                    st.markdown(preview_text)

            node_path: List[str] = []
            response_state: Dict[str, Any] = {}
            try:
                graph = st.session_state.graph
                graph_input = _build_graph_input()
                if st.session_state.get("debug_mode"):
                    response_state = run_async(run_graph_with_events(graph, graph_input, debug_box))
                    if asyncio.isfuture(response_state):
                        response_state = response_state.result()
                else:
                    response_state = run_async(run_graph_with_events(graph, graph_input, debug_box))
                    if asyncio.isfuture(response_state):
                        response_state = response_state.result()
            except Exception as exc:  # pragma: no cover - Streamlit surface
                response_state = {"state_messages": [{"role": "assistant", "content": f"⚠️ Error: {exc}"}]}
            _handle_response_state(response_state, assistant_segments)


if __name__ == "__main__":
    main()
