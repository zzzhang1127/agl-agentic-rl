#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""OpenCode SWE-smith agent for Agent Lightning k8s Jobs.

Runs **inside** the SWE-smith image (same as ``smith_agent.py``). The k8s
reconciler injects ``AGL_OPENAI_BASE_URL`` / ``AGL_EVENT_URL`` / ``AGL_KEY``.
This process:

  1. Checks out the bug commit and relocates ``.git`` (via ``smith_agent``).
  2. Proxies OpenAI traffic so tool results are truncated to ~4k tokens.
  3. Runs OpenCode (40k window, 6k output, compact at 12k remaining, 10 min).
  4. Grades with ``smith_agent.evaluate`` and POSTs a ``reward`` event.

Exit 0 is a finished rollout (including reward 0).
"""

from __future__ import annotations

import http.client
import http.server
import json
import logging
import os
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

# ConfigMap mounts this directory; smith_agent.py sits beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import smith_agent as smith  # noqa: E402

log = logging.getLogger(__name__)

SAMPLE_TIMEOUT = int(os.environ.get("AGL_SAMPLE_TIMEOUT", "600"))
OUTPUT_LIMIT = int(os.environ.get("AGL_MAX_TOKENS", "6144"))
CONTEXT_LIMIT = int(os.environ.get("AGL_OPENCODE_CONTEXT", "40960"))
COMPACT_RESERVED = int(os.environ.get("AGL_OPENCODE_RESERVED", "12000"))
TOOL_CHAR_LIMIT = int(os.environ.get("AGL_TOOL_CHAR_LIMIT", "16000"))
TOOL_COMBINED_CHAR_LIMIT = int(os.environ.get("AGL_TOOL_COMBINED_CHAR_LIMIT", "32000"))
OPENCODE_BIN = os.environ.get("AGL_OPENCODE_BIN", "/usr/local/bin/opencode")
LOG_DIR = Path(os.environ.get("AGL_LOG_DIR", "/agl-logs/new"))
# Episode brakes (2026-09-17, after the step-76 "explore-only loop" collapse, §45):
#   MAX_TURNS      hard cap on model calls; the proxy answers the next call with a
#                  synthetic "stop" so opencode ends the episode (upstream default 40; we use 60).
#   LOOP_REPEAT    N identical consecutive tool-call sets (same names + arguments)
#                  = a loop: episode ends, no evaluation, reward 0.
#   LEN_PEN_*      upstream length_penalized_reward: a SOLVED train rollout that
#                  needs more than T0 turns loses up to LAMBDA at MAX_TURNS.
#   PROMPT_PEN_*   upstream prompt_length_penalty on the longest single prompt.
MAX_TURNS = int(os.environ.get("AGL_MAX_TURNS", "60"))  # 09-17 实测:40 会砍掉 12% 已解决轨迹,60 只砍 4.8%
LOOP_REPEAT = int(os.environ.get("AGL_LOOP_REPEAT", "3"))
LEN_PEN_T0 = int(os.environ.get("AGL_LEN_PEN_T0", "25"))
LEN_PEN_LAMBDA = float(os.environ.get("AGL_LEN_PEN_LAMBDA", "0.2"))
PROMPT_PEN_SOFT = int(os.environ.get("AGL_PROMPT_PEN_SOFT", "30000"))
PROMPT_PEN_HARD = int(os.environ.get("AGL_PROMPT_PEN_HARD", str(CONTEXT_LIMIT)))
PROMPT_PEN_MAX = float(os.environ.get("AGL_PROMPT_PEN_MAX", "0.2"))


class EpisodeGuard:
    """Per-episode turn counter, loop detector and token bookkeeping.

    Lives on the proxy so every model call passes through it.  ``before_request``
    says whether the next call must be refused (returns the stop reason);
    ``after_response`` consumes the upstream completion and says whether the
    episode must end now (a loop was just completed).  Both return ``None`` to
    let the call through.  Pure bookkeeping, no I/O, so it is unit-testable.
    """

    def __init__(self, max_turns: int = MAX_TURNS, loop_repeat: int = LOOP_REPEAT) -> None:
        self.max_turns = max_turns
        self.loop_repeat = loop_repeat
        self.n_turns = 0
        self.max_prompt_tokens = 0
        self.completion_tokens = 0
        self.last_signature: str | None = None
        self.repeat_count = 0
        self.loop_detected = False
        self.turn_cap_hit = False
        self.stop_reason: str | None = None
        self._lock = threading.Lock()

    @staticmethod
    def tool_signature(completion: dict[str, Any]) -> str | None:
        """Canonical string of the tool calls in a completion; None when it has none."""
        try:
            message = completion["choices"][0].get("message") or {}
        except (KeyError, IndexError, TypeError, AttributeError):
            return None
        calls = message.get("tool_calls") or []
        if not calls:
            return None
        parts = []
        for call in calls:
            fn = call.get("function") or {}
            parts.append([fn.get("name"), fn.get("arguments", "")])
        return json.dumps(parts, ensure_ascii=False, sort_keys=True)

    def before_request(self) -> str | None:
        with self._lock:
            if self.stop_reason is not None:
                return self.stop_reason
            if self.max_turns > 0 and self.n_turns >= self.max_turns:
                self.turn_cap_hit = True
                self.stop_reason = f"turn cap {self.max_turns} reached"
                return self.stop_reason
            return None

    def after_response(self, completion: dict[str, Any]) -> str | None:
        with self._lock:
            self.n_turns += 1
            usage = completion.get("usage") or {}
            try:
                self.max_prompt_tokens = max(self.max_prompt_tokens, int(usage.get("prompt_tokens") or 0))
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
            except (TypeError, ValueError):
                pass
            signature = self.tool_signature(completion)
            if signature is not None and signature == self.last_signature:
                self.repeat_count += 1
            else:
                self.repeat_count = 1 if signature is not None else 0
            self.last_signature = signature
            if self.loop_repeat > 0 and self.repeat_count >= self.loop_repeat:
                self.loop_detected = True
                self.stop_reason = f"loop: identical tool call repeated {self.repeat_count}x"
                return self.stop_reason
            return None


def synthetic_stop_completion(text: str) -> bytes:
    """A chat.completion with no tool calls, so opencode finishes the episode."""
    comp = {
        "id": "agl-episode-stop",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "agl/Qwen3-8B",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return json.dumps(comp, ensure_ascii=False).encode("utf-8")


def _clip_text(value: Any, limit: int) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    omitted = len(value) - limit
    return value[:limit] + f"\n...[truncated {omitted} chars to {limit} (~4k tokens)]"


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "output"):
                    if isinstance(item.get(key), str):
                        parts.append(item[key])
        return "".join(parts)
    return ""


def _set_message_text(message: dict[str, Any], text: str) -> None:
    content = message.get("content")
    if isinstance(content, str) or content is None:
        message["content"] = text
        return
    if isinstance(content, list) and content:
        for item in reversed(content):
            if isinstance(item, dict):
                for key in ("text", "content", "output"):
                    if key in item and isinstance(item[key], str):
                        item[key] = text
                        return
        content[-1] = {"type": "text", "text": text}
        return
    message["content"] = text


def _is_tool_message(message: dict[str, Any]) -> bool:
    role = str(message.get("role") or "")
    return role == "tool" or (role == "user" and bool(message.get("tool_call_id")))


def truncate_openai_body(raw: bytes) -> bytes:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return raw
    for message in messages:
        if isinstance(message, dict) and _is_tool_message(message):
            _set_message_text(message, _clip_text(_message_text(message), TOOL_CHAR_LIMIT))
    # The combined budget applies to EVERY maximal run of consecutive tool messages
    # (i.e. every burst of parallel tool calls), not just the final one.
    #
    # It used to apply only to the tools after the last assistant message, and that
    # made the rendering of an unchanged message depend on its POSITION: a burst
    # that got clipped at turn k sat mid-history at turn k+1, where nothing clipped
    # it, so its full text came back.  The re-rendered prompt then no longer
    # continued the server's own context and the trajectory was cut into two rows
    # (§44.11).  Scoping the budget to the run itself makes each rendering a pure
    # function of that run's contents, so it is stable for the rest of the episode.
    run: list[dict[str, Any]] = []
    for message in list(messages) + [None]:
        if isinstance(message, dict) and _is_tool_message(message):
            run.append(message)
            continue
        if sum(len(_message_text(m)) for m in run) > TOOL_COMBINED_CHAR_LIMIT:
            # Front-to-back: earlier tool results keep their text; the one crossing the
            # shared budget is clipped with a marker; anything after becomes marker-only.
            budget = TOOL_COMBINED_CHAR_LIMIT
            for m in run:
                text = _message_text(m)
                if len(text) <= budget:
                    budget -= len(text)
                    continue
                if budget > 0:
                    omitted = len(text) - budget
                    _set_message_text(m, text[:budget] + f"\n...[truncated {omitted} chars: tool output limit]")
                else:
                    _set_message_text(m, f"...[truncated {len(text)} chars: tool output limit]")
                budget = 0
        run = []
    payload["max_tokens"] = min(int(payload.get("max_tokens") or OUTPUT_LIMIT), OUTPUT_LIMIT)
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = min(int(payload.get("max_completion_tokens") or OUTPUT_LIMIT), OUTPUT_LIMIT)
    # AGL training gateway records traces from non-streaming JSON only.
    payload["stream"] = False
    payload.pop("stream_options", None)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def completion_json_to_sse(raw: bytes) -> bytes | None:
    """Convert a non-streaming chat.completion JSON into the SSE chunk stream the
    client expects when it asked for stream=true (the gateway only records
    non-streaming traces, so upstream is always called with stream=false)."""
    try:
        comp = json.loads(raw.decode("utf-8"))
        choices = comp["choices"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(choices, list) or comp.get("object") == "chat.completion.chunk":
        return None
    head = {k: comp[k] for k in ("id", "created", "model", "system_fingerprint") if k in comp}
    head["object"] = "chat.completion.chunk"
    events: list[str] = []

    def emit(choice: dict[str, Any], usage: Any = None) -> None:
        chunk = dict(head)
        chunk["choices"] = [choice]
        if usage is not None:
            chunk["usage"] = usage
        events.append("data: " + json.dumps(chunk, ensure_ascii=False))

    for ch in choices:
        idx = ch.get("index", 0)
        msg = ch.get("message") or {}
        delta: dict[str, Any] = {"role": msg.get("role", "assistant")}
        if msg.get("content"):
            delta["content"] = msg["content"]
        if msg.get("reasoning_content"):
            delta["reasoning_content"] = msg["reasoning_content"]
        emit({"index": idx, "delta": delta, "finish_reason": None})
        for ti, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            emit(
                {
                    "index": idx,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": ti,
                                "id": tc.get("id"),
                                "type": tc.get("type", "function"),
                                "function": {"name": fn.get("name"), "arguments": fn.get("arguments", "")},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            )
        emit({"index": idx, "delta": {}, "finish_reason": ch.get("finish_reason") or "stop"}, usage=comp.get("usage"))
    events.append("data: [DONE]")
    return ("\n\n".join(events) + "\n\n").encode("utf-8")


class _GatewayProxyHandler(http.server.BaseHTTPRequestHandler):
    agl_host = "127.0.0.1"
    agl_port = 8080
    agl_prefix = ""
    agl_key = ""
    guard: EpisodeGuard = EpisodeGuard()

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("proxy " + fmt, *args)

    def _send_payload(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except BrokenPipeError:
            return

    def _send_stop(self, reason: str, wants_stream: bool) -> None:
        payload = synthetic_stop_completion(f"[episode ended by trainer: {reason}]")
        content_type = "application/json"
        if wants_stream:
            sse = completion_json_to_sse(payload)
            if sse is not None:
                payload, content_type = sse, "text/event-stream; charset=utf-8"
        self._send_payload(200, payload, content_type)

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        wants_stream = False
        is_chat = method == "POST" and "chat/completions" in self.path
        if is_chat:
            try:
                wants_stream = bool(json.loads(body.decode("utf-8")).get("stream"))
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                wants_stream = False
            stop = self.guard.before_request()
            if stop is not None:
                log.warning("proxy refusing model call %d: %s", self.guard.n_turns + 1, stop)
                self._send_stop(stop, wants_stream)
                return
            body = truncate_openai_body(body)
        suffix = self.path
        if suffix.startswith("/v1"):
            suffix = suffix[3:]
        if not suffix.startswith("/"):
            suffix = "/" + suffix
        target_path = self.agl_prefix.rstrip("/") + suffix
        headers = {k: v for k, v in self.headers.items() if k.lower() not in {"host", "content-length"}}
        if self.agl_key:
            headers["Authorization"] = f"Bearer {self.agl_key}"
        if body:
            headers["Content-Length"] = str(len(body))
        query = urllib.parse.urlparse(self.path).query
        if query and "?" not in target_path:
            target_path = target_path + "?" + query
        conn = http.client.HTTPConnection(self.agl_host, self.agl_port, timeout=SAMPLE_TIMEOUT + 60)
        try:
            conn.request(method, target_path, body=body or None, headers=headers)
            upstream = conn.getresponse()
            payload = upstream.read()
            content_type_override = None
            if is_chat and upstream.status == 200:
                try:
                    completion = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    completion = None
                if isinstance(completion, dict):
                    stop = self.guard.after_response(completion)
                    if stop is not None:
                        # The looping reply is already traced upstream; hand opencode a
                        # stop instead so the episode ends here.
                        log.warning("proxy ending episode after model call %d: %s", self.guard.n_turns, stop)
                        self._send_stop(stop, wants_stream)
                        return
            if wants_stream and upstream.status == 200:
                sse = completion_json_to_sse(payload)
                if sse is not None:
                    payload = sse
                    content_type_override = "text/event-stream; charset=utf-8"
            self.send_response(upstream.status)
            for key, value in upstream.getheaders():
                if key.lower() in {"transfer-encoding", "connection", "content-length"}:
                    continue
                if content_type_override and key.lower() == "content-type":
                    continue
                self.send_header(key, value)
            if content_type_override:
                self.send_header("Content-Type", content_type_override)
                self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if payload:
                try:
                    self.wfile.write(payload)
                except BrokenPipeError:
                    return
        finally:
            conn.close()

    def do_GET(self) -> None:  # noqa: N802
        self._forward("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._forward("POST")


class _ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def start_gateway_proxy(agl_openai_base_url: str, agl_key: str) -> tuple[_ReusableTCPServer, int]:
    parsed = urllib.parse.urlparse(agl_openai_base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    _GatewayProxyHandler.agl_host = host
    _GatewayProxyHandler.agl_port = port
    _GatewayProxyHandler.agl_prefix = parsed.path or ""
    _GatewayProxyHandler.agl_key = agl_key
    _GatewayProxyHandler.guard = EpisodeGuard()
    server = _ReusableTCPServer(("127.0.0.1", 0), _GatewayProxyHandler)
    bound = int(server.server_address[1])
    threading.Thread(target=server.serve_forever, name="agl-opencode-proxy", daemon=True).start()
    log.info("tool-truncate proxy on 127.0.0.1:%d -> %s:%d%s", bound, host, port, parsed.path or "")
    return server, bound


def write_opencode_json(path: Path, base_url: str, api_key: str) -> None:
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "model": "agl/Qwen3-8B",
        "small_model": "agl/Qwen3-8B",
        "provider": {
            "agl": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "agl-gateway",
                "options": {"baseURL": base_url, "apiKey": api_key or "dummy"},
                "models": {
                    "Qwen3-8B": {
                        "name": "Qwen3-8B",
                        "limit": {"context": CONTEXT_LIMIT, "input": CONTEXT_LIMIT, "output": OUTPUT_LIMIT},
                    }
                },
            }
        },
        "permission": {"*": "allow"},
        "compaction": {"auto": True, "prune": True, "reserved": COMPACT_RESERVED},
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def prompt_text(problem: str) -> str:
    return f"""You are a software engineer. Fix the bug described below in the repository at /testbed.

Work only on source files under /testbed. Do not modify tests, conftest.py, pytest.ini, tox.ini, setup.cfg, or pyproject.toml.

git is disabled and the .git directory is not present. Do not try to recover history, check out other commits, or read git metadata.

The network is disabled except the local training API. Do not use curl, wget, pip install, or fetch code from the internet. Everything you need is already in /testbed.

Make a general fix consistent with the codebase. When you believe the bug is fixed, stop and summarize the source files you changed.

<pr_description>
{problem}
</pr_description>
"""


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
    )
    problem = os.environ.get("AGL_TASK_INPUT", "").strip()
    if not problem:
        raise SystemExit("AGL_TASK_INPUT (problem statement) not set")
    eval_meta = json.loads(os.environ.get("AGL_EVAL_META", "") or "{}")
    if not eval_meta:
        eval_meta = smith.fetch_eval_meta()
    instance_id = eval_meta.get("instance_id", "")
    if not instance_id:
        raise SystemExit("eval meta unavailable: no instance_id; aborting before rollout")

    base_url = os.environ["AGL_OPENAI_BASE_URL"]
    api_key = os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
    eval_timeout = int(os.environ.get("SMITH_EVAL_TIMEOUT", "600"))
    f2p_only = os.environ.get("SMITH_F2P_ONLY", "1").lower() not in ("0", "false", "no")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    oc_home = Path("/tmp/oc-home")
    cfg_dir = oc_home / ".config" / "opencode"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (oc_home / ".local" / "share").mkdir(parents=True, exist_ok=True)

    log.info(
        "OpenCodeAgent start: instance=%s timeout=%s output=%s reserved=%s",
        instance_id,
        SAMPLE_TIMEOUT,
        OUTPUT_LIMIT,
        COMPACT_RESERVED,
    )
    if not Path(OPENCODE_BIN).exists():
        raise SystemExit(f"opencode binary missing: {OPENCODE_BIN}")

    smith.checkout_bug_commit(instance_id)
    smith.relocate_git()

    proxy, proxy_port = start_gateway_proxy(base_url, api_key)
    oc_cfg = cfg_dir / "opencode.json"
    write_opencode_json(oc_cfg, f"http://127.0.0.1:{proxy_port}/v1", api_key)
    prompt_path = LOG_DIR / "prompt.txt"
    prompt_path.write_text(prompt_text(problem), encoding="utf-8")
    oc_log = LOG_DIR / "opencode.jsonl"
    oc_err = LOG_DIR / "opencode.stderr"

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(oc_home),
            "XDG_CONFIG_HOME": str(oc_home / ".config"),
            "XDG_DATA_HOME": str(oc_home / ".local" / "share"),
            "OPENCODE_CONFIG": str(oc_cfg),
            "OPENAI_API_KEY": api_key or "dummy",
        }
    )
    t0 = time.time()
    runner = LOG_DIR / "inside.sh"
    runner.write_text(
        "#!/bin/bash\n"
        "set -u\n"
        f"timeout {SAMPLE_TIMEOUT} {OPENCODE_BIN} run --pure --auto --format json "
        "--dir /testbed -m agl/Qwen3-8B --title train "
        f"\"$(cat {prompt_path})\"\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    with oc_log.open("w", encoding="utf-8") as out, oc_err.open("w", encoding="utf-8") as err:
        oc_proc = subprocess.run(
            ["bash", str(runner)],
            cwd="/testbed",
            stdout=out,
            stderr=err,
            text=True,
            env=env,
        )
    wall = time.time() - t0
    log.info("opencode finished rc=%s wall=%.1fs log=%s", oc_proc.returncode, wall, oc_log)

    patch = smith.capture_patch()
    guard = _GatewayProxyHandler.guard
    if guard.loop_detected:
        # A looping episode is a failure by definition; skip the (up to 600 s) evaluation.
        reward, resolved, reason, timed_out = 0.0, False, f"loop detected ({guard.stop_reason}); not evaluated", False
    else:
        # 5th value is the dense F2P pass ratio, a train-only signal for the
        # MiniCPM RL runs; the teacher/eval path stays on the binary score so its
        # numbers stay comparable with every previously reported val result.
        reward, resolved, reason, timed_out, _ = smith.evaluate(eval_meta, eval_timeout, f2p_only=f2p_only)
    # Reward floor is 0 (human ruling 2026-09-07): a timeout logs its reason but is
    # never scored below a finished-but-unresolved rollout.  The old -0.2 timeout
    # penalty lived on in this copy until 2026-09-17 and fed the step-76 collapse (§45).
    if oc_proc.returncode == 124 and not resolved:
        reason += " | opencode timeout (no penalty)"
    if guard.turn_cap_hit:
        reason += f" | turn cap {guard.max_turns}"
    raw_reward = reward
    # Train mode is the /mode/train/ marker in the base URL (fail-safe to val, unshaped).
    is_train = "/mode/train/" in base_url
    reward = smith.length_penalized_reward(
        reward, guard.n_turns, MAX_TURNS, t0=LEN_PEN_T0, lam=LEN_PEN_LAMBDA, is_train=is_train
    )
    reward = smith.prompt_length_penalty(
        reward,
        guard.max_prompt_tokens,
        soft_start=PROMPT_PEN_SOFT,
        hard_cap=PROMPT_PEN_HARD,
        max_pen=PROMPT_PEN_MAX,
        is_train=is_train,
        solved=resolved,
    )
    reward = max(0.0, reward)
    event_base = {"instance_id": instance_id, "repo": eval_meta.get("repo", "")}
    summary = {
        **event_base,
        "reward": reward,
        "raw_reward": raw_reward,
        "resolved": resolved,
        "reason": reason,
        "eval_timeout": timed_out,
        "opencode_rc": oc_proc.returncode,
        "wall_seconds": wall,
        "patch_size": len(patch),
        "opencode_jsonl": str(oc_log),
        "n_turns": guard.n_turns,
        "max_prompt_tokens": guard.max_prompt_tokens,
        "completion_tokens": guard.completion_tokens,
        "loop_detected": guard.loop_detected,
        "turn_cap_hit": guard.turn_cap_hit,
        "is_train": is_train,
    }
    (LOG_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    smith.post_event("agent_output", {**event_base, "patch": patch, "patch_size": len(patch), "resolved": resolved})
    smith.post_event(
        "reward",
        {
            **event_base,
            "value": reward,
            "raw_reward": raw_reward,
            "resolved": resolved,
            "reason": reason,
            "eval_timeout": timed_out,
            "n_turns": guard.n_turns,
            "max_prompt_tokens": guard.max_prompt_tokens,
            "loop_detected": guard.loop_detected,
            "turn_cap_hit": guard.turn_cap_hit,
            "source": "opencode",
        },
        retry=True,
    )
    log.info(
        "reward=%.3f raw=%.3f resolved=%s n_turns=%d max_prompt_tokens=%d loop=%s cap=%s reason=%s",
        reward,
        raw_reward,
        resolved,
        guard.n_turns,
        guard.max_prompt_tokens,
        guard.loop_detected,
        guard.turn_cap_hit,
        reason,
    )
    proxy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
