"""LLM invocation helpers, models, and shared prompt utilities."""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Literal, Type, TypeVar

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    BaseModel = object

    def Field(*_args, **_kwargs):
        return None

SUMMARY_SYSTEM_PROMPT = "You are a concise assistant who summarizes topics clearly."
MessageRole = Literal["system", "user", "assistant"]
DEFAULT_PROVIDER = (os.environ.get("LLM_PROVIDER") or "google").lower()


class ChatMessage(dict):
    role: MessageRole
    content: str


AgentIntent = Literal["visualize", "sql_query", "simple_answer", "clarify"]

TModel = TypeVar("TModel", bound=BaseModel)


def _ensure_key(var_name: str, provider_label: str) -> str:
    api_key = os.environ.get(var_name)
    if not api_key:
        raise RuntimeError(f"Set {var_name} before using the {provider_label} provider.")
    return api_key


def _safe_json_dumps(value: Any) -> str:
    def _default(obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return str(obj)

    return json.dumps(value, ensure_ascii=False, default=_default)


_CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*([\s\S]+?)```", re.IGNORECASE)


def _strip_code_block(text: str) -> str:
    match = _CODE_FENCE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def _prepare_messages(
    system_prompt: str,
    user_prompt: str,
    history: List[ChatMessage] | None = None,
) -> List[ChatMessage]:
    """Build a chat message list that preserves recent history for context."""

    messages: List[ChatMessage] = []
    system_text = (system_prompt or "").strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if history:
        for msg in history:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            if role == "system" and system_text and content == system_text:
                continue  # avoid duplicating the active system prompt
            messages.append({"role": role, "content": content})
    user_text = (user_prompt or "").strip()
    if user_text:
        messages.append({"role": "user", "content": user_text})
    return messages


def _openai_response(messages: List[ChatMessage], json_mode: bool = False) -> str:
    from openai import OpenAI

    api_key = _ensure_key("OPENAI_API_KEY", "OpenAI")
    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(
        **kwargs,
        model=model,
        messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
        temperature=0.3,
    )
    return completion.choices[0].message.content or ""


def _azure_response(messages: List[ChatMessage], json_mode: bool = False) -> str:
    from openai import AzureOpenAI, BadRequestError

    api_key = _ensure_key("AZURE_OPENAI_API_KEY", "Azure OpenAI")
    endpoint = _ensure_key("AZURE_OPENAI_ENDPOINT", "Azure OpenAI")
    deployment = _ensure_key("AZURE_OPENAI_DEPLOYMENT", "Azure OpenAI")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        completion = client.chat.completions.create(
            **kwargs,
            model=deployment,
            messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
            temperature=0.3,
        )
        return completion.choices[0].message.content or ""
    except BadRequestError as exc:
        error = getattr(exc, "response", None)
        details = "Azure OpenAI content filter blocked the request. Please rephrase the prompt to avoid restricted content."
        try:
            payload = (error.json() if callable(getattr(error, "json", None)) else {}) or {}
            info = payload.get("error", {})
            inner = info.get("innererror", {})
            if inner.get("code") == "ResponsibleAIPolicyViolation":
                filters = inner.get("content_filter_result") or {}
                blocked = [name for name, outcome in filters.items() if outcome.get("filtered")]
                if blocked:
                    categories = ", ".join(sorted(blocked))
                    details = (
                        "Azure OpenAI blocked the request due to safety filters "
                        f"({categories}). Try simplifying the wording or using a different provider."
                    )
            message = info.get("message")
            if message:
                details = message
        except Exception as parse_exc:
            details = f"{details} (error payload parse failed: {parse_exc})"
        return details


def _google_response(messages: List[ChatMessage], json_mode: bool = False) -> str:
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

    generation_config = {}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
    model.generation_config = generation_config
    response = model.generate_content(conversation)
    if hasattr(response, "text") and response.text:
        return response.text
    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content.parts:
            part = candidate.content.parts[0]
            return getattr(part, "text", str(part))
    return ""


def _invoke_provider(messages: List[ChatMessage], provider: str, json_mode: bool = False) -> ChatMessage:
    if provider == "google":
        content = _google_response(messages, json_mode=json_mode)
    elif provider == "azure":
        content = _azure_response(messages, json_mode=json_mode)
    else:
        content = _openai_response(messages, json_mode=json_mode)
    return {"role": "assistant", "content": content.strip()}


def _call_llm(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    history: List[ChatMessage] | None = None,
) -> str:
    messages = _prepare_messages(system_prompt, user_prompt, history)
    reply = _invoke_provider(messages, provider)
    return reply["content"]


def _call_structured_llm(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    model_cls: Type[TModel],
    history: List[ChatMessage] | None = None,
) -> TModel:
    """Request a JSON response matching the supplied Pydantic schema."""

    schema_getter = getattr(model_cls, "model_json_schema", None)
    validator = getattr(model_cls, "model_validate_json", None)
    if not callable(schema_getter) or not callable(validator):
        raise RuntimeError("Structured responses require Pydantic to be installed.")
    schema = schema_getter()
    augmented_system = (
        system_prompt.strip()
        + "\n\nRespond with a JSON object that matches this schema:\n"
        + json.dumps(schema, indent=2)
    )
    attempts = [history] if history else [None]
    if history:
        attempts.append(None)  # Fallback: retry without history if validation fails.

    last_error: Exception | None = None
    for attempt_history in attempts:
        messages = _prepare_messages(augmented_system, user_prompt, attempt_history)
        response = _invoke_provider(messages, provider, json_mode=True)
        raw_content = response["content"]
        cleaned = _strip_code_block(raw_content)
        try:
            return validator(cleaned)
        except Exception as exc:
            last_error = exc
            continue

    # If we reach here, validation failed for all attempts.
    if last_error:
        raise last_error
    raise RuntimeError("Structured LLM call failed without an error detail.")


class Intent(BaseModel):
    """The classified intent of the user query."""

    intent: AgentIntent = Field(
        ..., description="The classified intent of the user query."
    )


class GeneratedSQL(BaseModel):
    """Structured response for SQL generation."""

    sql: str = Field(
        ...,
        description="The full SQL query text respecting safety requirements.",
    )


class VisualizationCodeResponse(BaseModel):
    """Structured response for visualization code generation."""

    language: Literal["python"] = Field(
        ..., description="Language name for the emitted code. Must be 'python'."
    )
    code: str = Field(..., description="Runnable Python code for the visualization.")


class StructuredAnswer(BaseModel):
    """Structured response for Streamlit UI rendering."""

    answer: str = Field(..., description="Primary user-facing answer in Korean.")
    highlights: List[str] = Field(
        default_factory=list,
        description="Optional bullet points for supplemental details.",
    )


class ClarificationRequest(BaseModel):
    """Structured follow-up question for multi-agent style prompts."""

    follow_up_question: str = Field(
        ...,
        description="A short Korean question requesting missing details.",
    )


class RouteDecision(BaseModel):
    """LLM-chosen next node for LangGraph routing."""

    next_node: str = Field(
        ...,
        description="The next LangGraph node label to run.",
    )
    reason: str = Field(
        default="",
        description="Short Korean rationale for the node choice.",
    )


def choose_route_with_llm(
    provider: str,
    options: List[str],
    state_summary: Dict[str, Any],
    history: List[ChatMessage] | None = None,
    hint: str | None = None,
) -> RouteDecision:
    """Ask the LLM to pick the next node based on the current state snapshot."""

    if not options:
        raise ValueError("At least one routing option is required.")
    option_text = ", ".join(options)
    state_json = _safe_json_dumps(state_summary)
    system_prompt = (
        "You are the routing brain for a Databricks analytics LangGraph. "
        "Select the best next node label from the allowed options. "
        "Prefer safety: if an error is present, choose node_error; if the user only needs a reply without SQL, choose node_respond."
    )
    hint_text = f"\nHint: {hint}" if hint else ""
    user_prompt = (
        f"Available nodes: {option_text}\n"
        f"State summary JSON:\n{state_json}\n"
        "Pick one `next_node` from the available nodes and explain briefly in Korean."
        f"{hint_text}"
    )
    return _call_structured_llm(
        provider, system_prompt, user_prompt, RouteDecision, history=history
    )
