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
import select
import signal
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ConfigMap mounts this directory; smith_agent.py sits beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import smith_agent as smith  # noqa: E402

log = logging.getLogger(__name__)

SAMPLE_TIMEOUT = int(os.environ.get("AGL_SAMPLE_TIMEOUT", "600"))
MAX_STEPS = int(os.environ.get("AGL_OPENCODE_MAX_STEPS", "100"))
OUTPUT_LIMIT = int(os.environ.get("AGL_MAX_TOKENS", "6144"))
CONTEXT_LIMIT = int(os.environ.get("AGL_OPENCODE_CONTEXT", "40960"))
COMPACT_RESERVED = int(os.environ.get("AGL_OPENCODE_RESERVED", "12000"))
TOOL_CHAR_LIMIT = int(os.environ.get("AGL_TOOL_CHAR_LIMIT", "16000"))
OPENCODE_BIN = os.environ.get("AGL_OPENCODE_BIN", "/usr/local/bin/opencode")
LOG_DIR = Path(os.environ.get("AGL_LOG_DIR", "/agl-logs/new"))


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
    last_assistant = -1
    for i, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            last_assistant = i
    last_tools = [
        message
        for message in messages[last_assistant + 1 :]
        if isinstance(message, dict) and _is_tool_message(message)
    ]
    combined = sum(len(_message_text(message)) for message in last_tools)
    if combined > TOOL_CHAR_LIMIT and last_tools:
        budget = TOOL_CHAR_LIMIT
        for message in reversed(last_tools):
            text = _message_text(message)
            if len(text) <= budget:
                budget -= len(text)
                continue
            _set_message_text(message, _clip_text(text, max(budget, 0)))
            budget = 0
    payload["max_tokens"] = min(int(payload.get("max_tokens") or OUTPUT_LIMIT), OUTPUT_LIMIT)
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = min(int(payload.get("max_completion_tokens") or OUTPUT_LIMIT), OUTPUT_LIMIT)
    # AGL training gateway records traces from non-streaming JSON only.
    payload["stream"] = False
    payload.pop("stream_options", None)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def completion_to_sse(payload: bytes) -> bytes:
    """Return buffered SSE to OpenCode while the AGL gateway records JSON."""
    completion = json.loads(payload)
    chunks = []
    common = {key: completion[key] for key in ("id", "created", "model") if key in completion}
    common["object"] = "chat.completion.chunk"

    def emit(choices, **extra):
        chunk = {**common, "choices": choices, **extra}
        chunks.append("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n")

    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("completion has no choices")
    for choice in choices:
        if choice.get("finish_reason") is None:
            raise ValueError("completion has no finish reason")
        delta = dict(choice["message"])
        if delta.get("tool_calls"):
            delta["tool_calls"] = [dict(call, index=i) for i, call in enumerate(delta["tool_calls"])]
        index = choice["index"]
        emit([{"index": index, "delta": delta, "finish_reason": None}])
        emit([{"index": index, "delta": {}, "finish_reason": choice["finish_reason"]}])
    if "usage" in completion:
        emit([], usage=completion["usage"])
    return ("".join(chunks) + "data: [DONE]\n\n").encode("utf-8")


class _GatewayProxyHandler(http.server.BaseHTTPRequestHandler):
    agl_host = "127.0.0.1"
    agl_port = 8080
    agl_prefix = ""
    agl_key = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("proxy " + fmt, *args)

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        wants_stream = False
        if method == "POST" and "chat/completions" in self.path:
            wants_stream = json.loads(body).get("stream") is True
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
            as_sse = wants_stream and upstream.status == 200
            if as_sse:
                try:
                    payload = completion_to_sse(payload)
                except (ValueError, KeyError, TypeError):
                    self.send_error(502, "Invalid upstream completion")
                    return
            self.send_response(upstream.status)
            for key, value in upstream.getheaders():
                if key.lower() in {"transfer-encoding", "connection", "content-length"}:
                    continue
                if as_sse and key.lower() == "content-type":
                    continue
                self.send_header(key, value)
            if as_sse:
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
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
                        "limit": {"context": CONTEXT_LIMIT, "output": OUTPUT_LIMIT},
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
    cmd = [
        OPENCODE_BIN,
        "run",
        "--pure",
        "--auto",
        "--format",
        "json",
        "--agent",
        "build",
        "--dir",
        "/testbed",
        "-m",
        "agl/Qwen3-8B",
        "--title",
        "train",
        prompt_path.read_text(encoding="utf-8"),
    ]
    terminal_seen = False
    step_count = 0
    oc_returncode = 124
    deadline = time.time() + SAMPLE_TIMEOUT
    with oc_log.open("w", encoding="utf-8") as out, oc_err.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            cmd,
            cwd="/testbed",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            env=env,
            start_new_session=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        while True:
            if proc.poll() is not None:
                for line in proc.stdout:
                    out.write(line)
                oc_returncode = proc.returncode or 0
                break
            if time.time() >= deadline:
                log.warning("opencode wall timeout %.0fs; force terminating process group", SAMPLE_TIMEOUT)
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    proc.wait(timeout=10)
                oc_returncode = 124
                break
            readable, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not readable:
                continue
            line = proc.stdout.readline()
            if not line:
                continue
            out.write(line)
            out.flush()
            try:
                event = json.loads(line)
            except Exception:
                continue
            if not isinstance(event, dict):
                continue
            part = event.get("part")
            if event.get("type") == "step_finish":
                step_count += 1
                if isinstance(part, dict) and part.get("reason") == "stop":
                    from opencode_terminal import classify_terminal_event
                    terminal_kind = classify_terminal_event(event, binary=OPENCODE_BIN, env=env, cwd="/testbed")
                    log.info("OPENCODE_TERMINAL_CLASSIFICATION message=%s kind=%s", part.get("messageID"), terminal_kind)
                    if terminal_kind == "task_stop":
                        terminal_seen = True
                        log.info("opencode terminal stop after %d steps; ending rollout before synthetic continue loop", step_count)
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                            proc.wait(timeout=5)
                        except Exception:
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except Exception:
                                pass
                            proc.wait(timeout=10)
                        oc_returncode = 0
                        break
                if step_count >= MAX_STEPS:
                    log.warning("opencode max steps %d reached; force terminating process group", MAX_STEPS)
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except Exception:
                            pass
                        proc.wait(timeout=10)
                    oc_returncode = 124
                    break
    wall = time.time() - t0

    @dataclass
    class _Completed:
        returncode: int

    oc_proc = _Completed(oc_returncode)
    log.info(
        "opencode finished rc=%s terminal_seen=%s steps=%d wall=%.1fs log=%s",
        oc_proc.returncode,
        terminal_seen,
        step_count,
        wall,
        oc_log,
    )

    patch = smith.capture_patch()
    evaluation_metadata = {}
    if os.environ.get("SMITH_STRUCTURED_EVAL", "0") == "1":
        from structured_eval import evaluate as evaluate_structured

        from rollout_termination import validate_opencode_terminal

        evaluation = evaluate_structured(smith, eval_meta, eval_timeout, f2p_only=f2p_only)
        termination = validate_opencode_terminal(oc_log, oc_proc.returncode)
        if termination["trajectory_terminal"]:
            # A naturally exited CLI may have buffered its final stdout event;
            # recheck the exact final message rather than trusting reason=stop.
            from opencode_terminal import classify_terminal_event
            finish_events = []
            for line in oc_log.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict) and item.get("type") == "step_finish":
                    finish_events.append(item)
            final_kind = classify_terminal_event(finish_events[-1], binary=OPENCODE_BIN, env=env, cwd="/testbed") if finish_events else "unknown"
            if final_kind != "task_stop":
                termination.update(trajectory_terminal=False, termination_reason="no_verified_task_terminal")
        evaluation.update(termination)
        if os.environ.get("SMITH_REQUIRE_TERMINAL", "0") == "1" and not termination["trajectory_terminal"]:
            evaluation["reward_eligible"] = False
            evaluation["evaluation_state"] = "unavailable"
            evaluation["approval_error"] = "nonterminal trajectory requires explicit bootstrap semantics"
        if oc_proc.returncode != 0:
            evaluation["reward_eligible"] = False
            evaluation["evaluation_state"] = "unavailable"
            evaluation["approval_error"] = f"OpenCode runner exited with code {oc_proc.returncode}"
        (LOG_DIR / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n")
        reward = evaluation["reward"]
        resolved = evaluation["resolved"]
        reason = evaluation["reason"]
        timed_out = evaluation.get("timed_out", False)
        f2p_ratio = (evaluation.get("n_f2p_pass", 0) / evaluation["n_f2p"]
                     if evaluation.get("n_f2p") and evaluation.get("reward_eligible") is True
                     and evaluation.get("n_p2p_ok") == evaluation.get("n_p2p") else 0.0)
        evaluation_metadata = {key: evaluation.get(key) for key in (
            "reward_eligible", "evaluation_state", "grader_protocol", "p2p_scope", "report_sha256",
            "trajectory_terminal", "termination_reason",
        )}
    else:
        reward, resolved, reason, timed_out, f2p_ratio = smith.evaluate(eval_meta, eval_timeout, f2p_only=f2p_only)
    event_base = {"instance_id": instance_id, "repo": eval_meta.get("repo", "")}
    summary = {
        **event_base,
        **evaluation_metadata,
        "reward": reward,
        "f2p_ratio": f2p_ratio,
        "resolved": resolved,
        "reason": reason,
        "eval_timeout": timed_out,
        "opencode_rc": oc_proc.returncode,
        "wall_seconds": wall,
        "patch_size": len(patch),
        "opencode_jsonl": str(oc_log),
    }
    (LOG_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    smith.post_event("agent_output", {**event_base, "patch": patch, "patch_size": len(patch), "resolved": resolved})
    smith.post_event(
        "reward",
        {
            **event_base,
            "value": reward,
            "resolved": resolved,
            "reason": reason,
            "eval_timeout": timed_out,
            "source": "opencode",
            **evaluation_metadata,
        },
        retry=True,
    )
    log.info("reward=%.3f resolved=%s reason=%s", reward, resolved, reason)
    proxy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
