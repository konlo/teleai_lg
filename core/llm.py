"""Compatibility shim for LLM helpers and LangGraph builder.

The LangGraph nodes and routing now live in `core.graph`, and shared LLM I/O
utilities/models live in `core.llm_io`. This module simply re-exports the
public surface so existing imports keep working while the codebase stays
modular.
"""
from __future__ import annotations

from core.graph import AgentState, DEFAULT_SQL_LIMIT, build_conversation_graph, summary_seed, summarize_topic
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

__all__ = [
    "AgentState",
    "AgentIntent",
    "ChatMessage",
    "ClarificationRequest",
    "DEFAULT_PROVIDER",
    "DEFAULT_SQL_LIMIT",
    "GeneratedSQL",
    "Intent",
    "StructuredAnswer",
    "VisualizationCodeResponse",
    "SUMMARY_SYSTEM_PROMPT",
    "_call_llm",
    "_call_structured_llm",
    "_safe_json_dumps",
    "_strip_code_block",
    "build_conversation_graph",
    "summary_seed",
    "summarize_topic",
]
