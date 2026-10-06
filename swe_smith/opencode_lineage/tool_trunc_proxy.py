#!/usr/bin/env python3
"""Tool-output truncating proxy for the MiniCPM 40k val sweep.

Listens on LISTEN_PORT (0.0.0.0, containers reach it via host.docker.internal),
forwards to 127.0.0.1:UPSTREAM_PORT (vLLM).

Rules (human decision 2026-09-08):
  - every tool message clipped to SINGLE_LIMIT chars (~4k tokens), keep head
  - tool messages after the LAST assistant message share COMBINED_LIMIT chars
    (~8k tokens), allocated FRONT-to-BACK: once the running total crosses the
    budget, that message is clipped and every later one becomes marker-only
  - max_tokens / max_completion_tokens capped at OUTPUT_LIMIT
Streaming is passed through untouched (request body is buffered, response is
relayed chunk-by-chunk).
"""
import json
import os
import socket
import threading
import time
import sys
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN_PORT = int(sys.argv[1])
UPSTREAM_PORT = int(sys.argv[2])
SINGLE_LIMIT = int(os.environ.get("TRUNC_SINGLE_CHARS", "16000"))
COMBINED_LIMIT = int(os.environ.get("TRUNC_COMBINED_CHARS", "32000"))
OUTPUT_LIMIT = int(os.environ.get("TRUNC_OUTPUT_TOKENS", "6144"))

# --- compaction recording (2026-09-21) ---------------------------------------------------------
# opencode's `run --format json` stream emits NO compaction event, so result.json could never see
# one.  A compaction is a separate model request whose first user message starts with COMPACT_MARK;
# the rebuilt context afterwards starts with POST_MARK (same markers as teacher_proxy.py).  We log
# one line per compaction request keyed by client IP (= container IP); run_repo_sweep.py matches
# the lines to its container by IP + time window.
COMPACT_MARK = "Here is the conversation so far"
POST_MARK = "What did we do so far?"
COMPACT_DIR = os.environ.get("TRUNC_COMPACT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "compact_events"))
_compact_lock = threading.Lock()


def record_compaction(raw, client_ip):
    try:
        messages = json.loads(raw.decode("utf-8")).get("messages") or []
        first = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                first = _get_text(m).lstrip()
                break
        if not first.startswith(COMPACT_MARK):
            return
        row = {"ts": time.time(), "ip": client_ip, "port": LISTEN_PORT,
               "n_messages": len(messages), "body_chars": len(raw)}
        with _compact_lock:
            os.makedirs(COMPACT_DIR, exist_ok=True)
            with open(os.path.join(COMPACT_DIR, f"{LISTEN_PORT}.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
    except Exception:
        pass


# --- step budget reminders (2026-09-21, user decision: tell the model its step budget, remind near the end) ----
# One step = one non-compaction chat request of a container (client IP).  The counter restarts on the first request of
# a session (task prompt, no assistant message yet).  From step BUDGET-WARN on, a reminder is appended to the LAST
# message of the request only (opencode resends its own history, so earlier reminders never accumulate).
STEP_BUDGET = int(os.environ.get("TRUNC_STEP_BUDGET", "0") or 0)
STEP_WARN = int(os.environ.get("TRUNC_STEP_WARN", "10"))
# TRUNC_STEP_HARD=1 (2026-09-22, user decision: step cap instead of wall clock): the request that would be step
# BUDGET+1 is NOT forwarded; the proxy answers with a synthetic assistant reply (finish_reason=stop, no tool call) so
# opencode ends the episode. Compaction requests are never counted or blocked.
STEP_HARD = os.environ.get("TRUNC_STEP_HARD", "0") == "1"
_steps = {}
_steps_lock = threading.Lock()


def apply_step_budget(raw, client_ip):
    if STEP_BUDGET <= 0:
        return raw
    try:
        payload = json.loads(raw.decode("utf-8"))
        messages = payload.get("messages") or []
        first = next((_get_text(m).lstrip() for m in messages if isinstance(m, dict) and m.get("role") == "user"), "")
        if first.startswith(COMPACT_MARK) or not messages:
            return raw
        fresh = "<pr_description>" in first and not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages)
        with _steps_lock:
            n = 1 if fresh else _steps.get(client_ip, 0) + 1
            _steps[client_ip] = n
        if STEP_HARD and n > STEP_BUDGET:
            raise _StepCapExceeded(n, bool(payload.get("stream")), payload.get("model") or "")
        if n < STEP_BUDGET - STEP_WARN:
            return raw
        if n < STEP_BUDGET:
            note = (f"[step budget] This is step {n} of {STEP_BUDGET}. Only {STEP_BUDGET - n} steps remain after this one: "
                    "finish your edit now and then stop with a short summary of the files you changed.")
        else:
            note = (f"[step budget] The budget of {STEP_BUDGET} steps is used up. Do not call any more tools. "
                    "Reply now with a short summary of the files you changed.")
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") in ("tool", "user"):
            _set_text(last, _get_text(last) + "\n\n" + note)
        else:
            messages.append({"role": "user", "content": note})
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")
    except _StepCapExceeded:
        raise
    except Exception:
        return raw


class _StepCapExceeded(Exception):
    def __init__(self, n, stream, model):
        self.n, self.stream, self.model = n, stream, model


STEP_CAP_TEXT = "Step budget exhausted: stopping here. The changes made so far are left in the working tree."


def _step_cap_reply(stream, model):
    """Synthetic OpenAI chat reply (SSE when the client streams) carrying finish_reason=stop and no tool call."""
    cid = f"chatcmpl-stepcap-{int(time.time()*1000)}"
    if not stream:
        body = {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": STEP_CAP_TEXT}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        return "application/json", json.dumps(body).encode()
    def chunk(delta, fin):
        return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                                      "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}) + "\n\n"
    sse = chunk({"role": "assistant", "content": ""}, None) + chunk({"content": STEP_CAP_TEXT}, None) + chunk({}, "stop") + "data: [DONE]\n\n"
    return "text/event-stream", sse.encode()


def _is_tool_message(m):
    return isinstance(m, dict) and (m.get("role") == "tool" or m.get("tool_call_id"))


def _get_text(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _set_text(m, text):
    m["content"] = text


def _clip(text, limit):
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n...[truncated {omitted} chars: tool output limit]"


def truncate_body(raw):
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw
    messages = payload.get("messages")
    if isinstance(messages, list):
        for m in messages:
            if _is_tool_message(m):
                _set_text(m, _clip(_get_text(m), SINGLE_LIMIT))
        last_assistant = -1
        for i, m in enumerate(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                last_assistant = i
        last_tools = [m for m in messages[last_assistant + 1:] if _is_tool_message(m)]
        if sum(len(_get_text(m)) for m in last_tools) > COMBINED_LIMIT:
            budget = COMBINED_LIMIT
            for m in last_tools:
                text = _get_text(m)
                if len(text) <= budget:
                    budget -= len(text)
                else:
                    _set_text(m, _clip(text, max(budget, 0)))
                    budget = 0
    if payload.get("max_tokens"):
        payload["max_tokens"] = min(int(payload["max_tokens"]), OUTPUT_LIMIT)
    if payload.get("max_completion_tokens"):
        payload["max_completion_tokens"] = min(int(payload["max_completion_tokens"]), OUTPUT_LIMIT)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _relay(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.command == "POST" and "/chat/completions" in self.path:
            record_compaction(body, self.client_address[0])
            body = truncate_body(body)
            try:
                body = apply_step_budget(body, self.client_address[0])
            except _StepCapExceeded as cap:
                ctype, payload = _step_cap_reply(cap.stream, cap.model)
                try:
                    with open(os.path.join(COMPACT_DIR, f"{LISTEN_PORT}.stepcap.jsonl"), "a") as f:
                        f.write(json.dumps({"ts": time.time(), "ip": self.client_address[0], "step": cap.n}) + "\n")
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
                self.close_connection = True
                return
        conn = http.client.HTTPConnection("127.0.0.1", UPSTREAM_PORT, timeout=900)
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "transfer-encoding", "connection")}
        headers["Content-Length"] = str(len(body))
        try:
            conn.request(self.command, self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in ("transfer-encoding", "connection", "content-length"):
                    continue
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception as exc:
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": f"proxy: {exc}"}}).encode())
            except Exception:
                pass
        finally:
            conn.close()
            self.close_connection = True

    do_GET = _relay
    do_POST = _relay


if __name__ == "__main__":
    os.makedirs(COMPACT_DIR, exist_ok=True)
    open(os.path.join(COMPACT_DIR, f"{LISTEN_PORT}.jsonl"), "a").close()   # file present == this proxy records compactions
    srv = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    print(f"trunc-proxy :{LISTEN_PORT} -> 127.0.0.1:{UPSTREAM_PORT} "
          f"single={SINGLE_LIMIT} combined={COMBINED_LIMIT} out={OUTPUT_LIMIT}", flush=True)
    srv.serve_forever()
