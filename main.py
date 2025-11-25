"""Streamlit chatbot UI backed by the shared LangGraph helpers."""
from __future__ import annotations

import io
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
from core.llm import ChatMessage, DEFAULT_SQL_LIMIT, build_conversation_graph

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
CODE_BLOCK_PATTERN = re.compile(r"```(?P<lang>\w+)?\s*([\s\S]+?)```", re.IGNORECASE)
# Replace JSON-style literals with Python equivalents when the LLM emits
# node_visualization code snippets. This avoids NameError when the model returns
# data samples containing null/true/false.
JSON_LITERAL_PATTERN = re.compile(r"\b(null|true|false)\b")
# Bump this to force Streamlit to rebuild the LangGraph when graph logic changes.
GRAPH_BUILD_ID = "2024-09-09-e"
LANGGRAPH_NODES = {
    "node_extract_user": "대화 내 최신 사용자 질문 추출",
    "node_configure_limits": "사용자 %limit 설정을 내부 변수로 반영",
    "node_intent_classifier": "질문 의도 분류 (visualize/sql/simple/clarify)",
    "node_s2w_tool": "시각화용 안전 SQL Tool 컨텍스트 준비",
    "node_table_select": "다음 테이블 선택 및 상태 업데이트",
    "node_table_sql": "선택된 테이블용 안전한 SQL 생성",
    "node_sql_generator": "스키마를 참고해 SQL 생성",
    "node_sql_validator": "생성된 SQL 안전성 및 구문 검증",
    "node_run_query": "Databricks에서 SQL 실행",
    "node_load_data": "쿼리 결과를 로딩 상태로 저장",
    "node_use_loaded_data": "기존 로딩 데이터로 시각화 준비",
    "node_visualization": "쿼리 결과를 이용한 시각화 코드 작성",
    "node_table_results": "각 테이블 결과/시각화 누적",
    "node_respond": "최종 답변/요약 작성",
    "node_clarify": "정보가 부족할 때 추가 질문",
    "node_error": "쿼리 실패 등 오류 안내",
}
LANGGRAPH_DOT = """
digraph {
    rankdir=LR;
    node [shape=rectangle, style=rounded];
    node_extract_user -> node_intent_classifier;
    node_intent_classifier -> node_configure_limits [label="%limit"];
    node_configure_limits -> node_respond [label="%limit only"];
    node_intent_classifier -> node_table_select [label="multi-table"];
    node_table_select -> node_table_sql -> node_run_query;
    node_intent_classifier -> node_s2w_tool [label="visualize/sql/loading"];
    node_intent_classifier -> node_use_loaded_data [label="visualize+cached"];
    node_s2w_tool -> node_sql_generator [label="ok"];
    node_s2w_tool -> node_error [label="meta error"];
    node_intent_classifier -> node_clarify [label="clarify"];
    node_intent_classifier -> node_respond [label="simple"];
    node_sql_generator -> node_sql_validator;
    node_sql_validator -> node_sql_generator [label="retry"];
    node_sql_validator -> node_run_query [label="pass"];
    node_sql_validator -> node_error [label="max fail"];
    node_run_query -> node_visualization [label="visualize"];
    node_run_query -> node_table_results [label="multi-table"];
    node_run_query -> node_load_data [label="loading"];
    node_run_query -> node_respond [label="sql only"];
    node_run_query -> node_error [label="error"];
    node_use_loaded_data -> node_visualization;
    node_visualization -> node_table_results [label="multi-table"];
    node_load_data -> node_respond [label="done"];
    node_visualization -> node_respond;
    node_table_results -> node_table_select [label="next"];
    node_table_results -> node_respond [label="done"];
    node_clarify -> end;
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


def _extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Return fenced code blocks and their language labels."""

    blocks: List[Tuple[str, str]] = []
    for match in CODE_BLOCK_PATTERN.finditer(text):
        lang = (match.group("lang") or "").strip().lower()
        code = match.group(2).strip()
        blocks.append((lang, code))
    return blocks


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    )


def _unescape_code_block(code: str) -> str:
    """Best-effort cleanup for code blocks that arrive as escaped strings."""

    fixed = code
    if "\\n" in fixed:
        fixed = fixed.replace("\\n", "\n")
    if '\\"' in fixed:
        fixed = fixed.replace('\\"', '"')
    if "\\'" in fixed:
        fixed = fixed.replace("\\'", "'")
    return fixed


def _normalize_json_literals(code: str) -> str:
    """Convert JSON-style literals (null/true/false) to Python values."""

    replacements = {"null": "None", "true": "True", "false": "False"}
    return JSON_LITERAL_PATTERN.sub(lambda match: replacements[match.group(1)], code)


def _is_graphviz_code(code: str) -> bool:
    """Heuristically detect Graphviz/DOT snippets to avoid exec syntax errors."""

    normalized = code.lstrip()
    candidates = ("digraph", "graph", "strict digraph", "strict graph")
    return normalized.lower().startswith(candidates) or "rankdir=" in code


def _execute_visualization_code(code: str, rows: List[Dict[str, Any]] | None = None) -> List[bytes]:
    """Run plotting code and return rendered figure PNG bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    cleaned_code = _unescape_code_block(code)
    normalized_code = _normalize_json_literals(cleaned_code)
    exec_globals: Dict[str, Any] = {
        "__name__": "__main__",
        "pd": pd,
        "plt": plt,
    }
    if rows is not None:
        exec_globals["df"] = pd.DataFrame(rows)
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
    new_fig_nums = sorted(after - before)
    new_figs = [plt.figure(num) for num in new_fig_nums]
    # If no new figure IDs but there is at least one active figure, reuse the latest one.
    if not new_figs and after:
        latest_num = max(after)
        new_figs = [plt.figure(latest_num)]
    images: List[bytes] = []
    for fig in new_figs:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        images.append(buffer.getvalue())
        plt.close(fig)
    return images


def _render_visualizations(message_index: int) -> None:
    """Show stored node_visualization outputs for a conversation turn."""
    visualizations = st.session_state.get("visualizations", {})
    code_blocks = st.session_state.get("visualization_codes", {})
    graphviz_blocks = st.session_state.get("visualization_graphs", {})
    messages = st.session_state.get("visualization_messages", {})
    # for code in code_blocks.get(message_index, []):
    #     st.code(code, language="python")
    # for code in graphviz_blocks.get(message_index, []):
    #     st.graphviz_chart(code)
    for idx, image in enumerate(visualizations.get(message_index, []), start=1):
        caption = f"Visualization result #{idx}"
        st.image(image, caption=caption)
    for entry in messages.get(message_index, []):
        level = entry.get("level", "info")
        text = entry.get("text", "")
        if level == "node_error":
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


def _handle_visualization_blocks(text: str, message_index: int) -> None:
    code_blocks = _extract_code_blocks(text)
    if not code_blocks:
        return
    viz_rows_by_msg = st.session_state.get("viz_rows", {})
    viz_tables_by_msg = st.session_state.get("viz_table_outputs", {})
    rows_for_msg = viz_rows_by_msg.get(message_index) or []
    if not rows_for_msg:
        table_outputs = viz_tables_by_msg.get(message_index) or []
        if table_outputs:
            rows_for_msg = table_outputs[0].get("rows") or []
    # Fall back to globally cached loaded data when no rows are attached to this message.
    if not rows_for_msg:
        rows_for_msg = st.session_state.get("state_loaded_data") or []
    for idx, (lang, code) in enumerate(code_blocks, start=1):
        lang_lower = lang or ""
        lang_lower = lang_lower.lower()
        if lang_lower in {"dot", "graphviz"} or _is_graphviz_code(code):
            # st.graphviz_chart(code)
            graphs = st.session_state.setdefault("visualization_graphs", {})
            graphs.setdefault(message_index, []).append(code)
            _append_visualization_message(
                message_index, "info", f"Code block {idx} Graphviz 렌더링 완료.")
            continue
        if lang_lower == "json" or _looks_like_json(code):
            st.code(code, language="json")
            _append_visualization_message(
                message_index,
                "info",
                f"Code block {idx}은 JSON 정보 블록이라 실행하지 않았어요.",
            )
            continue
        if lang_lower and lang_lower != "python":
            _append_visualization_message(
                message_index,
                "warning",
                f"Code block {idx}은 지원되지 않는 언어({lang})라 실행하지 않았어요.",
            )
            continue
        if not lang_lower:
            # Treat language-omitted blocks as Python for node_visualization.
            lang_lower = "python"
        if lang_lower and lang_lower != "python":
            _append_visualization_message(
                message_index,
                "warning",
                f"Code block {idx}은 지원되지 않는 언어({lang})라 실행하지 않았어요.",
            )
            continue
        # For Python node_visualization blocks, show the code before execution.
        codes = st.session_state.setdefault("visualization_codes", {})
        codes.setdefault(message_index, []).append(code)
        try:
            images = _execute_visualization_code(code, rows_for_msg)
        except Exception as exc:  # pragma: no cover - Streamlit surface
            _append_visualization_message(
                message_index, "node_error", f"⚠️ Code block {idx} 실행 중 오류: {exc}"
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
                for name in tables:
                    st.write(f"- {name}")
                    metadata = table_metadata.get(name, {})
                    columns = metadata.get("columns")
                    if columns:
                        schema_text = ", ".join(
                            f"{col['name']} ({col['type']})" for col in columns
                        )
                        st.caption(schema_text)
                    elif "node_error" in metadata:
                        st.caption(f"⚠️ {metadata['node_error']}")
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
