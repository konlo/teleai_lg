"""Trace and path formatting helpers for LangGraph nodes."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from core.llm_io import _safe_json_dumps


def debug_snapshot(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe copy of the state or update payload for tracing."""

    sanitized = {
        key: value
        for key, value in data.items()
        if key not in {"state_node_traces", "state_table_metadata", "state_loaded_data"}
    }
    try:
        return json.loads(_safe_json_dumps(sanitized))
    except Exception:
        return {key: str(value) for key, value in sanitized.items()}


def append_node_trace(
    state: Dict[str, Any],
    node_name: str,
    updates: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    traces = list(state.get("state_node_traces") or [])
    input_snapshot = debug_snapshot(state)
    output_snapshot = debug_snapshot(updates)
    merged_state = debug_snapshot({**input_snapshot, **output_snapshot})
    traces.append(
        {
            "node": node_name,
            "input": input_snapshot,
            "output": output_snapshot,
            "state": merged_state,
        }
    )
    return traces, input_snapshot, output_snapshot, merged_state


def log_node_io(
    node_name: str,
    input_snapshot: Dict[str, Any],
    output_snapshot: Dict[str, Any],
    merged_state: Dict[str, Any],
) -> None:
    try:
        _ = _safe_json_dumps(input_snapshot)
        _ = _safe_json_dumps(output_snapshot)
        _ = _safe_json_dumps(merged_state)
    except Exception:
        pass


def format_node_traces(traces: List[Dict[str, Any]] | None) -> str:
    if not traces:
        return ""
    segments: List[str] = ["\n\n노드 입출력 기록"]
    # 2025.11.25 konlo.na off 시킴
    # for trace in traces:
    #     node_name = trace.get("node", "unknown")
    #     segments.append(f"\n[{node_name}] 입력:")
    #     input_payload = _safe_json_dumps(trace.get("input", {}))
    #     segments.append(f"```json\n{input_payload}\n```")
    #     segments.append(f"[{node_name}] 출력:")
    #     output_payload = _safe_json_dumps(trace.get("output", {}))
    #     segments.append(f"```json\n{output_payload}\n```")
    # return "\n".join(segments)


def append_trace_text(text: str, state: Dict[str, Any]) -> str:
    trace_text = format_node_traces(state.get("state_node_traces"))
    if not trace_text:
        return text
    return f"{text}{trace_text}"


def node_path_diagram(node_path: List[str]) -> str:
    if not node_path:
        return ""
    arrow_line = " → ".join(node_path)
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
