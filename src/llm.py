"""Single structured LLM call with a strict JSON schema."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

_JSON_TYPE = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
    type(None): "null",
}


def _type_of(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    return _JSON_TYPE[type(value)]


def to_openai_strict(schema: Any) -> Any:
    if isinstance(schema, list):
        return [to_openai_strict(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    node = {key: to_openai_strict(value) for key, value in schema.items()}

    if "oneOf" in node:
        node["anyOf"] = node.pop("oneOf")

    if "type" not in node:
        if "const" in node:
            value = schema["const"]
            if value is None:
                node.pop("const")
                node["type"] = "null"
            else:
                node["type"] = _type_of(value)
        elif "enum" in node:
            kinds = {_type_of(item) for item in schema["enum"]}
            if len(kinds) == 1:
                node["type"] = kinds.pop()
    return node


DEFAULT_BASE_URL = "https://api.openai.com/v1"


REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in REASONING_PREFIXES)


def api_key() -> str:
    key = os.environ.get("AUTOMATION_LLM_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "AUTOMATION_LLM_API_KEY is not set. "
            "Set the AUTOMATION_LLM_API_KEY environment variable and run again."
        )
    return key


def _strict_response_format(response_format: dict) -> dict:
    out = dict(response_format)
    inner = dict(out["json_schema"])
    inner["schema"] = to_openai_strict(inner["schema"])
    out["json_schema"] = inner
    return out


def call_openai_structured_once(
    payload: dict,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 900.0,
    reasoning_effort: str | None = None,
):
    model = payload["model"]
    body = {
        "model": model,
        "messages": payload["messages"],
        "response_format": _strict_response_format(payload["response_format"]),
    }
    if is_reasoning_model(model):
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
    else:
        body["temperature"] = float(payload.get("temperature", 0.0))

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key()}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"OpenAI HTTP {error.code}: {detail}") from error

    choice = raw["choices"][0]
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            f"OpenAI structured call returned no content "
            f"(finish_reason={choice.get('finish_reason')})"
        )
    usage = raw.get("usage", {})
    adapted = {
        "choices": [{"message": {"content": content}}],
        "native_meta": {
            "model": raw.get("model"),
            "created": raw.get("created"),
            "finish_reason": choice.get("finish_reason"),
            "system_fingerprint": raw.get("system_fingerprint"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_effort": body.get("reasoning_effort"),
            "temperature": body.get("temperature"),
        },
    }

    sent = dict(body)
    return adapted, {"endpoint": f"{base_url.rstrip('/')}/chat/completions",
                     "body": sent, "api_key_recorded": False}
