# Copyright (c) Microsoft. All rights reserved.

"""Rewrite teacher-model replies into the MiniCPM smith bash-fence protocol.

MiniCPM eval still requires exactly one ```bash (or ```mswea_bash_command) fence.
Qwen / flash / tool-calling teachers often emit OpenAI ``tool_calls``, Hermes
``<tool_call>`` XML, or leftover ``<think>`` blocks instead. This module is the
teacher-only adapter: it never loosens ``parse_action`` for MiniCPM.

If a reply cannot be turned into a single bash fence, the original content is
returned unchanged so the harness records a format error instead of inventing a
command.
"""

from __future__ import annotations

import json
import re
from typing import Any

_BASH_FENCE_RE = re.compile(r"```(?:bash|mswea_bash_command|sh|shell)[^\S\n]*\n(.*?)```", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_HERMES_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_HERMES_FN_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>\s]+)>\s*(?P<body>.*?)</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_RE = re.compile(
    r"<parameter=(?P<key>[^>\s]+)>\s*(?P<val>.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_COMMAND_KEYS = ("command", "bash", "cmd", "script", "code", "input", "arguments")
_BASH_LANGS = {"", "bash", "sh", "shell", "mswea_bash_command"}


def strip_think(text: str) -> str:
    """Drop chain-of-thought tags; keep any trailing action text."""
    if not text:
        return ""
    stripped = _THINK_RE.sub("", text)
    stripped = _THINKING_RE.sub("", stripped)
    return stripped.strip()


def wrap_bash(command: str) -> str:
    """Render one smith-legal bash fence."""
    return f"```bash\n{command.strip()}\n```"


def _command_from_mapping(payload: Any) -> str | None:
    if isinstance(payload, str):
        raw = payload.strip()
        if not raw:
            return None
        if raw.startswith("{") or raw.startswith("["):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                return raw
            return _command_from_mapping(decoded)
        return raw
    if isinstance(payload, list):
        for item in payload:
            found = _command_from_mapping(item)
            if found:
                return found
        return None
    if not isinstance(payload, dict):
        return None
    for key in _COMMAND_KEYS:
        if key in payload and payload[key] not in (None, ""):
            found = _command_from_mapping(payload[key])
            if found:
                return found
    return None


def command_from_tool_calls(tool_calls: Any) -> str | None:
    """Pull a shell command out of OpenAI-style tool_calls."""
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    first = tool_calls[0]
    if not isinstance(first, dict):
        return None
    function = first.get("function") if isinstance(first.get("function"), dict) else first
    arguments = function.get("arguments") if isinstance(function, dict) else None
    return _command_from_mapping(arguments)


def command_from_hermes_xml(text: str) -> str | None:
    """Pull a shell command out of Hermes / Qwen ``<tool_call>`` markup."""
    if not text:
        return None
    match = _HERMES_JSON_RE.search(text)
    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            payload = None
        found = _command_from_mapping(payload)
        if found:
            return found
        if isinstance(payload, dict):
            found = _command_from_mapping(payload.get("arguments"))
            if found:
                return found
    match = _HERMES_FN_RE.search(text)
    if match:
        params = {item.group("key"): item.group("val") for item in _PARAM_RE.finditer(match.group("body"))}
        found = _command_from_mapping(params)
        if found:
            return found
        body = match.group("body").strip()
        if body and "<parameter=" not in body:
            return body
    return None


def first_bash_command(text: str) -> str | None:
    """Return the first bash/sh fence body, or a single unlabeled fence."""
    if not text:
        return None
    match = _BASH_FENCE_RE.search(text)
    if match:
        return match.group(1).strip() or None
    fences = list(_ANY_FENCE_RE.finditer(text))
    if len(fences) != 1:
        return None
    lang = fences[0].group(1).strip().split()[0].lower() if fences[0].group(1).strip() else ""
    if lang not in _BASH_LANGS:
        return None
    body = fences[0].group(2).strip()
    return body or None


def render_single_fence(command: str, thought: str = "") -> str:
    fence = wrap_bash(command)
    thought = thought.strip()
    if not thought:
        return fence
    return f"{thought}\n\n{fence}"


def rewrite_assistant_message(message: dict[str, Any] | None, content: str | None = None) -> tuple[str, bool]:
    """Return ``(content, rewritten)`` for one assistant message.

    ``rewritten`` is True only when the returned text is a single bash fence
    produced from tool_calls / XML / extra fences. Already-legal single-fence
    replies are returned as-is with ``rewritten=False`` after think-stripping
    if needed. Failures return the original content unchanged.
    """
    message = message or {}
    original = content if content is not None else message.get("content")
    if not isinstance(original, str):
        original = "" if original is None else str(original)

    stripped = strip_think(original)
    bash = first_bash_command(stripped)
    extra_fences = len(_ANY_FENCE_RE.findall(stripped))
    already_legal = bool(_BASH_FENCE_RE.search(stripped)) and extra_fences == 1
    if bash and extra_fences <= 1:
        # Drop think tags and promote unlabeled/sh fences to ```bash.
        if already_legal and stripped == original:
            return original, False
        thought = _BASH_FENCE_RE.sub("", stripped)
        thought = _ANY_FENCE_RE.sub("", thought).strip()
        return render_single_fence(bash, thought), True

    if bash and extra_fences > 1:
        thought = stripped[: stripped.find("```")].strip()
        return render_single_fence(bash, thought), True

    command = command_from_tool_calls(message.get("tool_calls"))
    if not command:
        command = command_from_hermes_xml(stripped) or command_from_hermes_xml(original)
    if not command and stripped:
        # AGL / some proxies dump tool_calls as a JSON string in content.
        try:
            dumped = json.loads(stripped)
        except json.JSONDecodeError:
            dumped = None
        if isinstance(dumped, list):
            command = command_from_tool_calls(dumped)
        elif isinstance(dumped, dict):
            command = _command_from_mapping(dumped) or command_from_tool_calls([dumped])

    if not command:
        return original, False
    thought = stripped
    thought = _HERMES_JSON_RE.sub("", thought)
    thought = _HERMES_FN_RE.sub("", thought).strip()
    if thought.startswith("{") or thought.startswith("["):
        thought = ""
    return render_single_fence(command, thought), True


def rewrite_choice(choice: dict[str, Any]) -> bool:
    """Rewrite ``choice.message`` in place. Returns whether content changed."""
    message = choice.get("message")
    if not isinstance(message, dict):
        return False
    new_content, changed = rewrite_assistant_message(message)
    if not changed:
        # Still strip think so a legal fence is not hidden behind <think>.
        stripped = strip_think(message.get("content") or "")
        bash = first_bash_command(stripped)
        if bash and stripped != (message.get("content") or ""):
            message["content"] = render_single_fence(bash)
            return True
        return False
    message["content"] = new_content
    # Harness only reads content; drop tool_calls so finish_reason is not
    # treated as a truncated tool call by format_error_message.
    message.pop("tool_calls", None)
    if choice.get("finish_reason") == "tool_calls":
        choice["finish_reason"] = "stop"
    return True
