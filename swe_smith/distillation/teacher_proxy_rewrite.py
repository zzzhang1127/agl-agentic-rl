#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Teacher-only OpenAI-compatible proxy that rewrites replies into smith bash fences.

MiniCPM eval must keep talking to raw vLLM. Point Qwen / flash teacher sweeps at
this proxy instead::

    python teacher_proxy_rewrite.py --upstream http://127.0.0.1:18008 --port 18010

Then ``AGL_VLLM_URL=http://127.0.0.1:18010``. Failed rewrites pass the upstream
body through so the harness still records format errors.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urljoin

from teacher_rewrite import rewrite_choice


def _load_json(raw: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


class TeacherProxyHandler(BaseHTTPRequestHandler):
    upstream = "http://127.0.0.1:18008"
    timeout_s = 300.0

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        url = urljoin(self.upstream.rstrip("/") + "/", self.path.lstrip("/"))
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length"}
        }
        if body and "Content-Type" not in {k.title() for k in headers}:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body or None, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                raw = resp.read()
                status = resp.status
                out_headers = dict(resp.headers.items())
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = int(exc.code)
            out_headers = dict(exc.headers.items()) if exc.headers else {}
        except Exception as exc:
            payload = json.dumps({"error": {"message": str(exc), "type": type(exc).__name__}}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path.rstrip("/").endswith("/chat/completions"):
            parsed = _load_json(raw)
            if parsed is not None:
                for choice in parsed.get("choices") or []:
                    if isinstance(choice, dict):
                        rewrite_choice(choice)
                raw = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                out_headers.pop("Content-Length", None)
                out_headers.pop("content-length", None)

        self.send_response(status)
        skip = {"transfer-encoding", "content-encoding", "content-length"}
        for key, value in out_headers.items():
            if key.lower() in skip:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._proxy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:18008")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18010)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    TeacherProxyHandler.upstream = args.upstream.rstrip("/")
    TeacherProxyHandler.timeout_s = args.timeout
    server = ThreadingHTTPServer((args.host, args.port), TeacherProxyHandler)
    print(f"teacher proxy {args.host}:{args.port} -> {TeacherProxyHandler.upstream}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
