# llm/llm_client.py

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from openai import OpenAI


_DEFAULT_MODEL = os.getenv("MEM_EVAL_MODEL", "gpt-4o-mini")


class LLMClient:
    """
    Thin wrapper around OpenAI Chat Completions (messages-based).

    Design goals:
    - One instance can be created during manager initialization and reused.
    - Centralized handling of API key path / env, model default, and response parsing.
    - Supports:
        - call_with_tools(): function-calling style (returns tool calls)
        - call_text(): plain text generation
    """

    def __init__(
        self,
        *,
        api_key_path: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        api_key: Optional[str] = None

        if api_key_path:
            path = os.path.expanduser(api_key_path)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    api_key = f.read().strip()

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "API key not found. "
                "Provide api_key_path or set OPENAI_API_KEY."
            )

        self._client = OpenAI(api_key=api_key)
        self._default_model = default_model or os.getenv("MEM_EVAL_MODEL", _DEFAULT_MODEL)

    @property
    def default_model(self) -> str:
        return self._default_model

    def _parse_tool_calls_from_chat_completion(self, response: Any) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []
        if not hasattr(response, "choices") or not response.choices:
            return tool_calls
        message = getattr(response.choices[0], "message", None)
        if not message:
            return tool_calls
        tool_calls_data = getattr(message, "tool_calls", None)
        if not tool_calls_data:
            return tool_calls
        for tc in tool_calls_data:
            func = getattr(tc, "function", None)
            if not func:
                continue
            name = getattr(func, "name", None)
            args = getattr(func, "arguments", None)
            if name is None or args is None:
                continue
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            tool_calls.append({"name": name, "arguments": args})
        return tool_calls

    def call_with_tools_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> List[Dict[str, Any]]:
        use_model = model or self._default_model
        response = self._client.chat.completions.create(
            model=use_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )
        return self._parse_tool_calls_from_chat_completion(response)

    def call_text_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        use_model = model or self._default_model
        response = self._client.chat.completions.create(
            model=use_model,
            messages=messages,
            temperature=temperature,
        )
        content = ""
        if hasattr(response, "choices") and response.choices:
            message = getattr(response.choices[0], "message", None)
            if message:
                content = getattr(message, "content", "")
        return (content or "").strip()

    def call_with_tools(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """\
        Call LLM with tools (messages-based function calling).

        Returns:
            A list of tool calls, each item like:
            {
                "name": "<tool_name>",
                "arguments": { ... }
            }

        If the model does not call any tool, returns [].

        Note:
            This project intentionally uses ONLY the Chat Completions messages API
            for deterministic tool calling.
        """
        if messages is None:
            raise ValueError("'messages' must be provided for call_with_tools in this project.")
        return self.call_with_tools_messages(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
        )

    def call_text(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Simple text-only LLM call (messages-based, no tools)."""
        return self.call_text_messages(
            messages=messages,
            model=model,
            temperature=temperature,
        )

    def chat_json(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Return a JSON object (dict) from the model using response_format=json_object."""
        use_model = model or self._default_model
        response = self._client.chat.completions.create(
            model=use_model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = ""
        if hasattr(response, "choices") and response.choices:
            message = getattr(response.choices[0], "message", None)
            if message:
                content = getattr(message, "content", "")
        content = (content or "").strip()
        try:
            return json.loads(content)
        except Exception as e:
            raise ValueError(f"chat_json: failed to parse JSON. raw_content={content!r}") from e

    def embed_texts(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI's embedding API.
        """
        use_model = model or "text-embedding-3-small"
        response = self._client.embeddings.create(
            model=use_model,
            input=texts,
        )
        return [d.embedding for d in response.data]


# -----------------------------
# Backward-compatible module API
# -----------------------------

_global_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    """
    Lazily create a process-wide default client.
    Prefer constructing your own LLMClient db_manager __init__ for clarity,
    but this keeps older call sites working.
    """
    global _global_client
    if _global_client is None:
        _global_client = LLMClient()
    return _global_client


def call_with_tools(
    *,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    return get_default_client().call_with_tools(
        messages=messages,
        tools=tools,
        model=model,
        temperature=temperature,
    )


def call_text(
    *,
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    return get_default_client().call_text(
        messages=messages,
        model=model,
        temperature=temperature,
    )


def embed_texts(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    return get_default_client().embed_texts(texts, model=model)


__all__ = ["LLMClient", "get_default_client", "call_with_tools", "call_text", "embed_texts"]