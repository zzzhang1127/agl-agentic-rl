#!/usr/bin/env python3
"""Teacher-model proxy for SFT trajectory collection (2026-09-20).

opencode inside the SWE-smith containers talks OpenAI chat/completions to
host.docker.internal:LISTEN_PORT.  This proxy

  * applies the SAME tool-output truncation rules as tool_trunc_proxy.py
    (so the teacher sees exactly what the MiniCPM student would see),
  * swaps the dummy Authorization for the real gateway key and forwards to
    the HTTPS gateway (TEACHER_BASE_URL, TEACHER_API_KEY, TEACHER_MODEL),
  * rejects any request whose `tools` contain a non-`function` tool
    (no server-side web search / code interpreter / built-in browser),
  * logs every request + the assembled response (streaming SSE is
    re-assembled) under LOG_DIR: `events.jsonl` gets one small row per
    request (latency/usage/response only) and `<session>.json` is
    OVERWRITTEN with the latest full request+response, so the final file of
    a session is the complete trajectory (each request carries the whole
    history).  session = md5 of the first user message, which is the
    prompt_text(instance) string, so it joins back to instance_id.
  * (21:30 patch, run "ra") optionally REWRITES the `usage` block that opencode
    sees with MiniCPM-tokenizer counts (env MINICPM_TOKENIZER=<hf dir>), so
    opencode's compaction trigger (tokens.total >= context - output) fires by
    the student's own counting; the gateway's raw usage stays in events.jsonl
    (`response.usage`, for cost) and the rewritten one is `response.usage_minicpm`.
  * (21:30 patch) optional per-run turn cap (env TEACHER_MAX_TURNS, 0 = off):
    like the student's EpisodeGuard, the (cap+1)-th model call of a run is
    answered with a synthetic no-tool "episode ended" reply so opencode stops;
    events.jsonl gets an {"event": "turn_cap"} row for the assembler.

usage: teacher_proxy.py LISTEN_PORT LOG_DIR
"""
import copy
import hashlib
import json
import os
import ssl
import sys
import threading
import time
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

LISTEN_PORT = int(sys.argv[1])
LOG_DIR = sys.argv[2]
os.makedirs(LOG_DIR, exist_ok=True)
EVENTS_PATH = os.path.join(LOG_DIR, "events.jsonl")
BASE = os.environ["TEACHER_BASE_URL"].rstrip("/")
KEY = os.environ["TEACHER_API_KEY"]
MODEL = os.environ.get("TEACHER_MODEL", "gpt-5.5")
SINGLE_LIMIT = int(os.environ.get("TRUNC_SINGLE_CHARS", "16000"))
COMBINED_LIMIT = int(os.environ.get("TRUNC_COMBINED_CHARS", "32000"))
OUTPUT_LIMIT = int(os.environ.get("TRUNC_OUTPUT_TOKENS", "16384"))
UPSTREAM_TIMEOUT = int(os.environ.get("TEACHER_UPSTREAM_TIMEOUT", "900"))
MAX_TURNS = int(os.environ.get("TEACHER_MAX_TURNS", "0"))
TOKENIZER_DIR = os.environ.get("MINICPM_TOKENIZER", "")

_u = urlparse(BASE)
UP_HOST, UP_PORT, UP_PREFIX = _u.hostname, _u.port or (443 if _u.scheme == "https" else 80), _u.path.rstrip("/")
UP_TLS = _u.scheme == "https"
_log_lock = threading.Lock()


def _is_tool_message(m):
    return isinstance(m, dict) and (m.get("role") == "tool" or m.get("tool_call_id"))


def _get_text(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _clip(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars: tool output limit]"


def truncate_messages(messages):
    for m in messages:
        if _is_tool_message(m):
            m["content"] = _clip(_get_text(m), SINGLE_LIMIT)
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
                m["content"] = _clip(text, max(budget, 0))
                budget = 0


def norm_prompt(t):
    """opencode re-quotes the CLI message as "..." with inner quotes escaped; undo that so
    md5(norm_prompt(first user msg)) == md5(prompt_text(instance).strip())."""
    t = t.strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1].replace('\\"', '"')
    return t.strip()


def session_key(messages):
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return hashlib.md5(norm_prompt(_get_text(m)).encode("utf-8")).hexdigest()
    return "nouser"


# --- MiniCPM token counting (usage rewrite) ---------------------------------------------------
# vLLM renders the request with the model's chat template (tool_calls.function.arguments parsed
# from JSON string to dict first, exactly like vllm.entrypoints.chat_utils does) and reports
# len(tokens) as prompt_tokens; we do the same so the teacher run is counted the student's way.
_tok = None
_tok_lock = threading.Lock()
if TOKENIZER_DIR:
    from transformers import AutoTokenizer  # only imported when the feature is on
    _tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)


def _norm_for_template(messages):
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        m = copy.deepcopy(m)
        if m.get("content") is None:
            m["content"] = ""
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    fn["arguments"] = json.loads(a)
                except Exception:
                    fn["arguments"] = {"_raw": a}
        out.append(m)
    return out


def count_prompt_tokens(messages, tools):
    """(n_tokens, method) with the MiniCPM chat template; falls back to a JSON-text count."""
    with _tok_lock:
        try:
            ids = _tok.apply_chat_template(_norm_for_template(messages), tools=tools or None,
                                           add_generation_prompt=True, tokenize=True)
            return len(ids), "chat_template"
        except Exception:
            text = json.dumps(messages, ensure_ascii=False) + (json.dumps(tools, ensure_ascii=False) if tools else "")
            return len(_tok.encode(text, add_special_tokens=False)), "json_fallback"


def count_completion_tokens(msg):
    """MiniCPM count of what the assistant produced: reasoning + content + tool calls."""
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    calls = "".join(
        (tc.get("function") or {}).get("name", "") + (tc.get("function") or {}).get("arguments", "")
        for tc in msg.get("tool_calls") or []
    )
    with _tok_lock:
        n_r = len(_tok.encode(reasoning, add_special_tokens=False)) if reasoning else 0
        n_c = len(_tok.encode(content + calls, add_special_tokens=False)) if (content or calls) else 0
    return n_r + n_c, n_r


def minicpm_usage(prompt_tokens, msg, method):
    n_comp, n_reason = count_completion_tokens(msg)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": n_comp,
        "total_tokens": prompt_tokens + n_comp,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": n_reason},
    }


# --- per-client run tracking (fix 2026-09-20) -------------------------------------------------
# opencode rebuilds the context after a compaction as  user "What did we do so far?" + assistant
# <summary> + continuation, so md5(first user msg) is the SAME for every post-compaction request of
# every problem and the session files collided.  One opencode run == one container == one client
# IP, so we key by client IP: the task-prompt request registers the run, later requests from the
# same IP are filed as segments of that run:
#   <task>.json            last full pre-compaction request (unchanged format)
#   <task>.compactK.json   K-th compaction summarization request + summary  (K = 1, 2, ...)
#   <task>.postK.json      last full request of the continuation after compaction K
#   <task>.otherKEY.json   anything else from that client (e.g. task-tool sub-sessions)
# run["n_turns"] counts every answered model call of the run (task/compact/post/other alike, like
# the student's EpisodeGuard which sits in front of every call) for the TEACHER_MAX_TURNS cap.
COMPACT_MARK = "Here is the conversation so far"
POST_MARK = "What did we do so far?"
_runs = {}          # client ip -> {"task": key, "n_compact": int, "n_turns": int}
_runs_lock = threading.Lock()


def classify_request(client_ip, messages):
    """Return (file_stem, segment) for this request."""
    first = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            first = norm_prompt(_get_text(m))
            break
    key = hashlib.md5(first.encode("utf-8")).hexdigest()
    # the 1st compaction request renders the whole conversation (task text included) inside the
    # summarizer prompt -> test the compaction/post markers BEFORE the task marker (bug 09-20 18:55)
    is_compact = first.startswith(COMPACT_MARK)
    is_post = first.startswith(POST_MARK)
    is_task = "<pr_description>" in first and not is_compact and not is_post
    with _runs_lock:
        run = _runs.get(client_ip)
        if is_task:
            if run is None or run["task"] != key:
                run = {"task": key, "n_compact": 0, "n_turns": 0}
                _runs[client_ip] = run
            return key, "task"
        if run is None:
            return key, "unknown"
        if is_compact:
            run["n_compact"] += 1
            return f"{run['task']}.compact{run['n_compact']}", "compact"
        if is_post:
            return f"{run['task']}.post{max(run['n_compact'], 1)}", "post"
        return f"{run['task']}.other{key[:8]}", "other"


def turn_cap_reached(client_ip):
    """Turn count of the run before this call, or None when the call may go through."""
    if MAX_TURNS <= 0:
        return None
    with _runs_lock:
        run = _runs.get(client_ip)
        if run is not None and run.get("n_turns", 0) >= MAX_TURNS:
            return run["n_turns"]
    return None


def count_turn(client_ip):
    with _runs_lock:
        run = _runs.get(client_ip)
        if run is not None:
            run["n_turns"] = run.get("n_turns", 0) + 1
            return run["n_turns"]
    return None


class StreamAssembler:
    """Incrementally re-assemble an OpenAI SSE stream into one message dict."""

    def __init__(self):
        self.content, self.reasoning, self.tool_calls, self.finish, self.usage = [], [], {}, None, None

    def feed(self, obj):
        if obj.get("usage"):
            self.usage = obj["usage"]
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                self.content.append(d["content"])
            if d.get("reasoning_content"):
                self.reasoning.append(d["reasoning_content"])
            for tc in d.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = self.tool_calls.setdefault(idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                self.finish = ch["finish_reason"]

    def message(self):
        msg = {"role": "assistant", "content": "".join(self.content) or None}
        if self.reasoning:
            msg["reasoning_content"] = "".join(self.reasoning)
        if self.tool_calls:
            msg["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        return msg

    def result(self):
        return {"message": self.message(), "finish_reason": self.finish, "usage": self.usage}


def assemble_stream(raw: bytes):
    """Re-assemble an OpenAI SSE stream into one message dict."""
    asm = StreamAssembler()
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]" or not data:
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        asm.feed(obj)
    return asm.result()


def assemble_json(raw: bytes):
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return {"message": None, "finish_reason": None, "usage": None, "raw": raw[:2000].decode("utf-8", "replace")}
    ch = (obj.get("choices") or [{}])[0]
    return {"message": ch.get("message"), "finish_reason": ch.get("finish_reason"), "usage": obj.get("usage"), "error": obj.get("error")}


def synthetic_stop(text, stream):
    """(body, content_type): a no-tool assistant reply so opencode ends the session."""
    now = int(time.time())
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not stream:
        comp = {"id": "teacher-episode-stop", "object": "chat.completion", "created": now, "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": usage}
        return json.dumps(comp, ensure_ascii=False).encode("utf-8"), "application/json"
    head = {"id": "teacher-episode-stop", "object": "chat.completion.chunk", "created": now, "model": MODEL}
    chunks = [
        dict(head, choices=[{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]),
        dict(head, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
        dict(head, choices=[], usage=usage),
    ]
    body = "".join("data: " + json.dumps(c, ensure_ascii=False) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode("utf-8"), "text/event-stream; charset=utf-8"


def write_log(row):
    with _log_lock:
        with open(EVENTS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_session(skey, row):
    path = os.path.join(LOG_DIR, f"{skey}.json")
    tmp = f"{path}.{threading.get_ident()}.tmp"   # unique per thread: no rename race between concurrent writers
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(row, fh, ensure_ascii=False)
    os.replace(tmp, path)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send_body(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _reject(self, code, msg):
        self._send_body(code, json.dumps({"error": {"message": msg}}).encode(), "application/json")

    def _relay(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        is_chat = self.command == "POST" and "/chat/completions" in self.path
        payload = None
        skey = None
        stem = segment = None
        prompt_count = None
        wants_stream = False
        t0 = time.time()
        client_ip = self.client_address[0]
        if is_chat:
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return self._reject(400, "proxy: bad json")
            tools = payload.get("tools") or []
            bad = [t for t in tools if not (isinstance(t, dict) and t.get("type") == "function")]
            if bad:
                write_log({"ts": t0, "event": "rejected_tools", "tools": bad})
                return self._reject(400, "proxy: only function tools are allowed")
            if payload.get("web_search_options") or payload.get("tool_resources"):
                return self._reject(400, "proxy: server-side tools are not allowed")
            payload["model"] = MODEL
            wants_stream = bool(payload.get("stream"))
            messages = payload.get("messages")
            if isinstance(messages, list):
                truncate_messages(messages)
                skey = session_key(messages)
                stem, segment = classify_request(client_ip, messages)
            else:
                stem, segment = skey, "nomsg"
            n_before = turn_cap_reached(client_ip)
            if n_before is not None:
                reason = f"turn cap {MAX_TURNS} reached"
                write_log({"ts": t0, "event": "turn_cap", "session": skey, "stem": stem, "segment": segment,
                           "client": client_ip, "n_turns": n_before, "max_turns": MAX_TURNS})
                out, ctype = synthetic_stop(f"[episode ended by trainer: {reason}]", wants_stream)
                return self._send_body(200, out, ctype)
            if payload.get("max_tokens"):
                payload["max_tokens"] = min(int(payload["max_tokens"]), OUTPUT_LIMIT)
            if payload.get("max_completion_tokens"):
                payload["max_completion_tokens"] = min(int(payload["max_completion_tokens"]), OUTPUT_LIMIT)
            if wants_stream:
                so = payload.get("stream_options") or {}
                so["include_usage"] = True
                payload["stream_options"] = so
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "transfer-encoding", "connection", "authorization", "x-api-key")}
        headers["Authorization"] = f"Bearer {KEY}"
        headers["Content-Length"] = str(len(body))
        headers["Host"] = UP_HOST
        if UP_TLS:
            conn = http.client.HTTPSConnection(UP_HOST, UP_PORT, timeout=UPSTREAM_TIMEOUT, context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=UPSTREAM_TIMEOUT)
        collected = bytearray()      # what the CLIENT received (usage possibly rewritten)
        status = None
        asm = StreamAssembler()
        rewritten_usage = gateway_usage = None
        try:
            conn.request(self.command, UP_PREFIX + self.path, body=body or None, headers=headers)
            if is_chat and _tok is not None and isinstance(payload.get("messages"), list):
                # counted while the gateway works on the request (~0.1 s at 40k tokens)
                prompt_count = count_prompt_tokens(payload["messages"], payload.get("tools"))
            resp = conn.getresponse()
            status = resp.status
            cost_hdr = {k: resp.getheader(k) for k in ("x-litellm-response-cost", "x-litellm-key-spend", "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens")}
            ctype = resp.getheader("Content-Type") or ""
            is_sse = "text/event-stream" in ctype
            rewrite = is_chat and status == 200 and _tok is not None and prompt_count is not None
            if rewrite and not is_sse:
                # non-streaming: buffer, rewrite usage, send with Content-Length
                raw = resp.read()
                try:
                    obj = json.loads(raw.decode("utf-8"))
                    ch = (obj.get("choices") or [{}])[0]
                    if obj.get("usage") and ch.get("message"):
                        gateway_usage = obj["usage_gateway"] = obj["usage"]
                        rewritten_usage = obj["usage"] = minicpm_usage(prompt_count[0], ch["message"], prompt_count[1])
                        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                except Exception:
                    pass
                collected.extend(raw)
                self.send_response(status)
                for k, v in resp.getheaders():
                    if k.lower() in ("transfer-encoding", "connection", "content-length"):
                        continue
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)
                self.wfile.flush()
            else:
                self.send_response(status)
                for k, v in resp.getheaders():
                    if k.lower() in ("transfer-encoding", "connection", "content-length"):
                        continue
                    self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                pending = b""
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    if not (rewrite and is_sse):
                        collected.extend(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        continue
                    pending += chunk
                    out_lines = []
                    while True:
                        nl = pending.find(b"\n")
                        if nl < 0:
                            break
                        line, pending = pending[:nl + 1], pending[nl + 1:]
                        s = line.decode("utf-8", errors="replace")
                        if s.startswith("data:"):
                            data = s[5:].strip()
                            if data and data != "[DONE]":
                                try:
                                    obj = json.loads(data)
                                except json.JSONDecodeError:
                                    obj = None
                                if isinstance(obj, dict):
                                    asm.feed(obj)
                                    if obj.get("usage"):
                                        gateway_usage = obj["usage_gateway"] = obj["usage"]
                                        rewritten_usage = obj["usage"] = minicpm_usage(prompt_count[0], asm.message(), prompt_count[1])
                                        line = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
                        out_lines.append(line)
                    if out_lines:
                        blob = b"".join(out_lines)
                        collected.extend(blob)
                        self.wfile.write(blob)
                        self.wfile.flush()
                if pending:
                    collected.extend(pending)
                    self.wfile.write(pending)
                    self.wfile.flush()
        except Exception as exc:
            try:
                self._reject(502, f"proxy: {exc}")
            except Exception:
                pass
            if is_chat:
                write_log({"ts": t0, "event": "upstream_error", "session": skey, "error": str(exc)[:500]})
            return
        finally:
            conn.close()
            self.close_connection = True
        if is_chat:
            out = assemble_stream(bytes(collected)) if (is_sse or wants_stream) else assemble_json(bytes(collected))
            if rewritten_usage is not None:
                # events.jsonl keeps the GATEWAY usage in `usage` (cost accounting); what opencode saw is `usage_minicpm`
                out["usage_minicpm"] = rewritten_usage
                out["usage"] = gateway_usage
            n_turn = count_turn(client_ip) if status == 200 and out.get("message") else None
            n_msgs = len(payload.get("messages") or [])
            write_log({
                "ts": t0,
                "latency_s": round(time.time() - t0, 2),
                "session": skey,
                "stem": stem,
                "segment": segment,
                "client": client_ip,
                "status": status,
                "n_messages": n_msgs,
                "n_turn": n_turn,
                "tools": [t.get("function", {}).get("name") for t in payload.get("tools") or []],
                "cost_usd": float(cost_hdr.get("x-litellm-response-cost") or 0) if cost_hdr.get("x-litellm-response-cost") else None,
                "key_spend_usd": float(cost_hdr.get("x-litellm-key-spend") or 0) if cost_hdr.get("x-litellm-key-spend") else None,
                "rl_remaining_req": cost_hdr.get("x-ratelimit-remaining-requests"),
                "rl_remaining_tok": cost_hdr.get("x-ratelimit-remaining-tokens"),
                "response": out,
            })
            if status == 200 and out.get("message"):
                write_session(stem, {
                    "ts": t0,
                    "session": skey,
                    "stem": stem,
                    "segment": segment,
                    "client": client_ip,
                    "model": MODEL,
                    "n_messages": n_msgs,
                    "n_turn": n_turn,
                    "request": {
                        "messages": payload.get("messages"),
                        "tool_schemas": payload.get("tools"),
                        "temperature": payload.get("temperature"),
                        "max_tokens": payload.get("max_tokens") or payload.get("max_completion_tokens"),
                    },
                    "response": out,
                })

    do_GET = _relay
    do_POST = _relay


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    print(f"teacher-proxy :{LISTEN_PORT} -> {BASE} model={MODEL} single={SINGLE_LIMIT} combined={COMBINED_LIMIT} "
          f"out={OUTPUT_LIMIT} max_turns={MAX_TURNS} tokenizer={TOKENIZER_DIR or '-'} log={LOG_DIR}", flush=True)
    srv.serve_forever()
