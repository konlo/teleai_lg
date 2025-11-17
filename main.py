"""Streamlit chatbot UI backed by the shared LangGraph helpers."""
from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

import streamlit as st

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv():
        return None

from core.databricks import fetch_table_metadata, fetch_table_preview, list_tables
from core.llm import ChatMessage, build_conversation_graph

load_dotenv()


SYSTEM_PROMPT = "You are a helpful assistant who keeps answers concise and clear."
TABLE_PREVIEW_LIMIT = 50
TABLE_LOAD_PATTERN = re.compile(
    r"(?P<table>[A-Za-z0-9_]+)\s+table\s+(?:데이터|데이타)\s*로딩(?:해줘|해줘요|해줘라|해줘봐|해줄래|해줄수있어|해|줘)?",
    re.IGNORECASE,
)
CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*([\s\S]+?)```", re.IGNORECASE)
# Replace JSON-style literals with Python equivalents when the LLM emits
# visualization code snippets. This avoids NameError when the model returns
# data samples containing null/true/false.
JSON_LITERAL_PATTERN = re.compile(r"\b(null|true|false)\b")
LANGGRAPH_NODES = {
    "extract_user": "대화 내 최신 사용자 질문 추출",
    "intent": "질문 의도 분류 (visualize/sql/simple/clarify)",
    "sql_generator": "스키마를 참고해 SQL 생성",
    "run_query": "Databricks에서 SQL 실행",
    "visualization": "쿼리 결과를 이용한 시각화 코드 작성",
    "response": "최종 답변/요약 작성",
    "clarify": "정보가 부족할 때 추가 질문",
    "error": "쿼리 실패 등 오류 안내",
}
LANGGRAPH_DOT = """
digraph {
    rankdir=LR;
    node [shape=rectangle, style=rounded];
    extract_user -> intent;
    intent -> sql_generator [label="visualize/sql"];
    intent -> clarify [label="clarify"];
    intent -> response [label="simple"];
    sql_generator -> run_query;
    run_query -> visualization [label="visualize"];
    run_query -> response [label="sql only"];
    run_query -> error [label="error"];
    visualization -> response;
    clarify -> end;
    response -> end;
    error -> end;
}
"""


def _match_table_reference(prompt: str, candidates: Iterable[str]) -> str | None:
    """Return first table whose name appears in the prompt (case-insensitive)."""
    normalized_prompt = prompt.lower()
    for table in candidates:
        pattern = re.compile(rf"(?<!\w){re.escape(table.lower())}(?!\w)")
        if pattern.search(normalized_prompt):
            return table
    return None


def _extract_code_blocks(text: str) -> List[str]:
    """Return fenced python code blocks from text."""
    return [match.group(1).strip() for match in CODE_BLOCK_PATTERN.finditer(text)]


def _normalize_json_literals(code: str) -> str:
    """Convert JSON-style literals (null/true/false) to Python values."""

    replacements = {"null": "None", "true": "True", "false": "False"}
    return JSON_LITERAL_PATTERN.sub(lambda match: replacements[match.group(1)], code)


def _is_graphviz_code(code: str) -> bool:
    """Heuristically detect Graphviz/DOT snippets to avoid exec syntax errors."""

    normalized = code.lstrip()
    candidates = ("digraph", "graph", "strict digraph", "strict graph")
    return normalized.lower().startswith(candidates) or "rankdir=" in code


def _execute_visualization_code(code: str) -> List[bytes]:
    """Run plotting code and return rendered figure PNG bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized_code = _normalize_json_literals(code)
    exec_globals: Dict[str, Any] = {"__name__": "__main__"}
    before = set(plt.get_fignums())
    try:
        exec(normalized_code, exec_globals, exec_globals)
    except SyntaxError as exc:  # pragma: no cover - handled in UI path
        details = exc.msg
        if exc.text:
            details += f" -> {exc.text.strip()}"
        if exc.lineno is not None:
            details += f" (line {exc.lineno})"
        raise SyntaxError(details) from exc
    after = set(plt.get_fignums())
    new_figs = [plt.figure(num) for num in sorted(after - before)]
    images: List[bytes] = []
    for fig in new_figs:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        images.append(buffer.getvalue())
        plt.close(fig)
    return images


def _render_visualizations(message_index: int) -> None:
    """Show stored visualization outputs for a conversation turn."""
    visualizations = st.session_state.get("visualizations", {})
    messages = st.session_state.get("visualization_messages", {})
    for idx, image in enumerate(visualizations.get(message_index, []), start=1):
        caption = f"Visualization result #{idx}"
        st.image(image, caption=caption)
    for entry in messages.get(message_index, []):
        level = entry.get("level", "info")
        text = entry.get("text", "")
        if level == "error":
            st.error(text)
        elif level == "warning":
            st.warning(text)
        else:
            st.info(text)


def _append_visualization_images(message_index: int, images: List[bytes]) -> None:
    if not images:
        return
    visualizations = st.session_state.setdefault("visualizations", {})
    visualizations.setdefault(message_index, []).extend(images)


def _append_visualization_message(message_index: int, level: str, text: str) -> None:
    messages = st.session_state.setdefault("visualization_messages", {})
    messages.setdefault(message_index, []).append({"level": level, "text": text})


def _handle_visualization_blocks(text: str) -> None:
    code_blocks = _extract_code_blocks(text)
    if not code_blocks:
        return
    message_index = len(st.session_state.history) - 1
    for idx, code in enumerate(code_blocks, start=1):
        if _is_graphviz_code(code):
            st.graphviz_chart(code)
            _append_visualization_message(
                message_index, "info", f"Code block {idx} Graphviz 렌더링 완료.")
            continue
        try:
            images = _execute_visualization_code(code)
        except Exception as exc:  # pragma: no cover - Streamlit surface
            _append_visualization_message(
                message_index, "error", f"⚠️ Code block {idx} 실행 중 오류: {exc}"
            )
            continue
        if images:
            _append_visualization_images(message_index, images)
        else:
            _append_visualization_message(
                message_index, "info", f"Code block {idx} 실행 완료 (생성된 도표 없음)."
            )
    _render_visualizations(message_index)


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


def _lc_messages(history: List[Tuple[str, str]]) -> List[ChatMessage]:
    """Convert (role, content) tuples to LangGraph-compatible message dicts."""
    messages: List[ChatMessage] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in history:
        messages.append({"role": role, "content": content})
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
    if "table_metadata" not in st.session_state:
        st.session_state.table_metadata = {}
    if "node_paths" not in st.session_state:
        st.session_state.node_paths = {}

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
        ):
            st.session_state.graph = build_conversation_graph(provider=provider)
            st.session_state.graph_provider = provider

        with st.expander("Databricks Tables", expanded=True):
            refresh_requested = st.button("Refresh tables", use_container_width=True)
            if refresh_requested:
                st.session_state.pop("tables", None)
                st.session_state.pop("tables_error", None)
                st.session_state.pop("table_metadata", None)
            if "tables" not in st.session_state and "tables_error" not in st.session_state:
                try:
                    st.session_state.tables = list_tables()
                    metadata: Dict[str, Dict[str, Any]] = {}
                    for name in st.session_state.tables:
                        try:
                            metadata[name] = fetch_table_metadata(name)
                        except Exception as meta_exc:
                            metadata[name] = {"error": str(meta_exc)}
                    st.session_state.table_metadata = metadata
                except Exception as exc:  # pragma: no cover - UI path
                    st.session_state.tables_error = str(exc)

            tables_error = st.session_state.get("tables_error")
            tables = st.session_state.get("tables", [])

            if tables_error:
                st.warning(tables_error)
            elif tables:
                table_metadata = st.session_state.get("table_metadata", {})
                for name in tables:
                    st.write(f"- {name}")
                    metadata = table_metadata.get(name, {})
                    columns = metadata.get("columns")
                    if columns:
                        schema_text = ", ".join(
                            f"{col['name']} ({col['type']})" for col in columns
                        )
                        st.caption(schema_text)
                    elif "error" in metadata:
                        st.caption(f"⚠️ {metadata['error']}")
            else:
                st.info("No tables available.")

        with st.expander("LangGraph Flow", expanded=False):
            st.caption("현재 에이전트 그래프 노드 및 동작 요약")
            for node_key, description in LANGGRAPH_NODES.items():
                st.markdown(f"- **{node_key}**: {description}")
            st.graphviz_chart(LANGGRAPH_DOT)

    for index, (role, content) in enumerate(st.session_state.history):
        with st.chat_message(role):
            st.markdown(content)
            _render_visualizations(index)
            _render_node_flow(index)

    if prompt := st.chat_input("Enter your message"):
        st.session_state.history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        table_name = None
        table_match = TABLE_LOAD_PATTERN.search(prompt)
        if table_match:
            table_name = table_match.group("table")
        elif tables := st.session_state.get("tables"):
            table_name = _match_table_reference(prompt, tables)

        with st.chat_message("assistant"):
            if table_name:
                try:
                    sql_statement, rows = fetch_table_preview(
                        table_name, limit=TABLE_PREVIEW_LIMIT
                    )
                    row_count = len(rows)
                    text = (
                        f"Detected reference to `{table_name}` and loaded "
                        f"{row_count} row(s) (limit {TABLE_PREVIEW_LIMIT})."
                    )
                    st.markdown(text)
                    st.code(sql_statement, language="sql")
                    if rows:
                        st.dataframe(rows, use_container_width=True)
                    else:
                        st.info("No data returned for this table.")
                except Exception as exc:  # pragma: no cover - Streamlit surface
                    text = f"⚠️ Error loading `{table_name}`: {exc}"
                    st.markdown(text)
                st.session_state.history.append(("assistant", text))
            else:
                node_path: List[str] = []
                try:
                    graph = st.session_state.graph
                    metadata = st.session_state.get("table_metadata", {})
                    response_state = graph.invoke(
                        {
                            "messages": _lc_messages(st.session_state.history),
                            "table_metadata": metadata,
                        }
                    )
                    latest = response_state["messages"][-1]
                    text = latest["content"]
                    node_path = response_state.get("node_path", [])
                except Exception as exc:  # pragma: no cover - Streamlit surface
                    text = f"⚠️ Error: {exc}"
                st.session_state.history.append(("assistant", text))
                message_index = len(st.session_state.history) - 1
                st.markdown(text)
                _handle_visualization_blocks(text)
                _store_node_path(message_index, node_path)
                _render_node_flow(message_index)


if __name__ == "__main__":
    main()
