"""Streamlit chatbot UI backed by the shared LangGraph helpers."""
from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, Iterable, List, Tuple

import streamlit as st

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv():
        return None

try:
    # Newer langchain
    from langchain.callbacks import StreamlitCallbackHandler
    _streamlit_cb_error = None
except Exception as exc1:  # pragma: no cover - optional dependency
    try:
        # Older path
        from langchain.callbacks.streamlit import StreamlitCallbackHandler
        _streamlit_cb_error = None
    except Exception as exc2:  # pragma: no cover - optional dependency
        StreamlitCallbackHandler = None
        _streamlit_cb_error = f"{exc1} / {exc2}"

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
    if "node_paths" not in st.session_state:
        st.session_state.node_paths = {}
    if "state_sql_limit" not in st.session_state:
        st.session_state.state_sql_limit = DEFAULT_SQL_LIMIT

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

        trace_container = st.expander("LLM Trace (Streamlit callback)", expanded=False)
        if StreamlitCallbackHandler is None:
            trace_container.info(
                "StreamlitCallbackHandler를 불러오지 못했습니다. "
                "langchain 버전을 확인하거나 `pip install langchain` 후 다시 실행해 주세요."
                f"{f' (import node_error: {_streamlit_cb_error})' if _streamlit_cb_error else ''}"
            )
        else:
            trace_container.caption("여기에서 LLM 호출 로그를 확인할 수 있어요.")

    for index, (role, content) in enumerate(st.session_state.history):
        with st.chat_message(role):
            st.markdown(content)
            _render_visualizations(index)
            _render_node_flow(index)

    if prompt := st.chat_input("Enter your message"):
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
                metadata = st.session_state.get("state_table_metadata", {})
                callbacks = []
                if StreamlitCallbackHandler is not None:
                    callbacks.append(StreamlitCallbackHandler(trace_container))
                response_state = graph.invoke(
                    {
                        "state_messages": _lc_messages(st.session_state.history),
                        "state_table_metadata": metadata,
                        "state_sql_limit": st.session_state.get("state_sql_limit", DEFAULT_SQL_LIMIT),
                        "state_loaded_data": st.session_state.get("state_loaded_data"),
                        "state_loaded_columns": st.session_state.get("state_loaded_columns"),
                        "state_loaded_table": st.session_state.get("state_loaded_table"),
                        "state_active_table_metadata": st.session_state.get("state_active_table_metadata"),
                    },
                    config={"callbacks": callbacks} if callbacks else None,
                )
                latest = response_state["state_messages"][-1]
                response_text = latest["content"]
                node_path = response_state.get("state_node_path", [])
                st.session_state.state_sql_limit = response_state.get(
                    "state_sql_limit", DEFAULT_SQL_LIMIT
                )
            except Exception as exc:  # pragma: no cover - Streamlit surface
                response_text = f"⚠️ Error: {exc}"
                response_state = {}
            assistant_segments.append(response_text)
            combined_text = "\n\n".join(segment for segment in assistant_segments if segment)
            st.session_state.history.append(("assistant", combined_text))
            message_index = len(st.session_state.history) - 1
            # Cache data for node_visualization execution without embedding it in the code text.
            viz_rows = st.session_state.setdefault("viz_rows", {})
            viz_rows[message_index] = response_state.get("state_loaded_data") if response_state else []
            viz_tables = st.session_state.setdefault("viz_table_outputs", {})
            viz_tables[message_index] = response_state.get("state_table_outputs") if response_state else []
            # Persist loaded data/columns/table across turns for reuse.
            if response_state:
                st.session_state.state_loaded_data = response_state.get("state_loaded_data")
                st.session_state.state_loaded_columns = response_state.get("state_loaded_columns")
                st.session_state.state_loaded_table = response_state.get("state_loaded_table")
                st.session_state.state_active_table_metadata = response_state.get("state_active_table_metadata")
            if node_path and node_path[-1] == "node_error":
                st.error(response_text)
            else:
                st.markdown(response_text)
            _handle_visualization_blocks(response_text, message_index)
            _store_node_path(message_index, node_path)
            _render_node_flow(message_index)


if __name__ == "__main__":
    main()
