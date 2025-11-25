"""Visualization-related helpers for the Streamlit app."""
from __future__ import annotations

import hashlib
import io
import json
import re
from typing import Any, Dict, List, Tuple

import streamlit as st

CODE_BLOCK_PATTERN = re.compile(r"```(?P<lang>\w+)?\s*([\s\S]+?)```", re.IGNORECASE)
# Replace JSON-style literals with Python equivalents when the LLM emits
# node_visualization code snippets. This avoids NameError when the model returns
# data samples containing null/true/false.
JSON_LITERAL_PATTERN = re.compile(r"\b(null|true|false)\b")


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


def _sanitize_python_block(code: str) -> str:
    """Remove trailing backslashes/newline artifacts that break syntax."""
    cleaned_lines: List[str] = []
    for line in code.splitlines():
        cleaned_lines.append(line.rstrip("\\"))
    return "\n".join(cleaned_lines)


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
    sanitized_code = _sanitize_python_block(cleaned_code)
    normalized_code = _normalize_json_literals(sanitized_code)
    exec_globals: Dict[str, Any] = {
        "__name__": "__main__",
        "pd": pd,
        "plt": plt,
    }
    if rows is not None:
        exec_globals["df"] = pd.DataFrame(rows)
    before = set(plt.get_fignums())
    exec(normalized_code, exec_globals)
    after = set(plt.get_fignums())
    new_figs = [plt.figure(num) for num in sorted(after - before)] or (
        [plt.gcf()] if plt.gcf() else []
    )
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
    for idx, image in enumerate(visualizations.get(message_index, []), start=1):
        caption = f"Visualization result #{idx}"
        st.image(image, caption=caption)


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
    st.session_state.setdefault("visualizations", {})[message_index] = []
    st.session_state.setdefault("visualization_codes", {})[message_index] = []
    st.session_state.setdefault("visualization_graphs", {})[message_index] = []
    st.session_state.setdefault("visualization_messages", {})[message_index] = []
    viz_rows_by_msg = st.session_state.get("viz_rows", {})
    viz_tables_by_msg = st.session_state.get("viz_table_outputs", {})
    rows_for_msg = viz_rows_by_msg.get(message_index) or []
    if not rows_for_msg:
        table_outputs = viz_tables_by_msg.get(message_index) or []
        if table_outputs:
            rows_for_msg = table_outputs[0].get("rows") or []
    if not rows_for_msg:
        rows_for_msg = st.session_state.get("state_loaded_data") or []
    for idx, (lang, code) in enumerate(code_blocks, start=1):
        lang_lower = lang or ""
        lang_lower = lang_lower.lower()
        if lang_lower in {"dot", "graphviz"} or _is_graphviz_code(code):
            st.graphviz_chart(code)
            graphs = st.session_state.setdefault("visualization_graphs", {})
            graphs.setdefault(message_index, []).append(code)
            _append_visualization_message(
                message_index, "info", f"Code block {idx} Graphviz 렌더링 완료."
            )
            continue
        if lang_lower == "json" or _looks_like_json(code):
            st.code(code, language="json")
            continue
        if lang_lower and lang_lower != "python":
            continue
        if not lang_lower:
            st.code(code)
            continue
        if lang_lower and lang_lower != "python":
            _append_visualization_message(
                message_index,
                "warning",
                f"Code block {idx}은 지원되지 않는 언어({lang})라 실행하지 않았어요.",
            )
            continue
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


__all__ = [
    "_handle_visualization_blocks",
    "_render_visualizations",
]
