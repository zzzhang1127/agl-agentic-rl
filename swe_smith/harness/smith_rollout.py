#!/usr/bin/env python3
"""Run official smith_agent loop only (no pytest). Used inside the SWE image.

A stdlib OpenAI-compatible client talks to host vLLM so the testbed Python
does not need the openai package. Evaluation is done afterwards by eval_inside.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smith_agent as sa  # noqa: E402


class _HttpError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens


class _Completion:
    def __init__(self, content: str, finish_reason: str, prompt_tokens: int) -> None:
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage(prompt_tokens)


class _Completions:
    def __init__(self, base_url: str, api_key: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        extra_body: dict[str, Any] | None = None,
    ) -> _Completion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_body:
            payload.update(extra_body)
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            raise _HttpError(int(exc.code), raw or str(exc)) from exc
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        return _Completion(
            msg.get("content") or "",
            choice.get("finish_reason") or "stop",
            int(usage.get("prompt_tokens") or 0),
        )


class OpenAI:
    def __init__(self, base_url: str, api_key: str = "dummy", max_retries: int = 6) -> None:
        del max_retries
        self.chat = type("Chat", (), {"completions": _Completions(base_url, api_key, 300.0)})()


def _query(client: Any, messages: list[dict[str, Any]], max_tokens: int) -> tuple[str, str, int]:
    model = os.environ.get("AGL_MODEL", "Qwen3-8B")
    # 2026-10-02: upstream hard-codes temperature=1.0, which overrides the server's
    # --override-generation-config. The OpenCode val-474 numbers we compare against
    # were all measured at 0.6, so make it settable and pass 0.6 for those runs.
    temperature = float(os.environ.get("SMITH_TEMPERATURE", "1.0"))
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            # 2026-10-05: must match smith_agent._query's extra_body exactly — this
            # function replaces it (see sa._query = _query below), so a fix made
            # there does not reach the eval path unless it is repeated here.
            #   skip_special_tokens=False: MiniCPM5's <function>/<param> tool-call
            #     tags are special tokens, so the default True deletes them and the
            #     harness scores a well-formed reply as a format error.
            #   stop: harmless insurance. The earlier claim here ("the server
            #     does not turn generation_config.json's eos_token_id into a stop
            #     condition") is RETRACTED 10-05 — it came from an n>1 measurement
            #     read as one reply. At n=1 the server stops at <|im_end|> itself.
            # See docs/pitfalls-and-lessons.md §54, §55.
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "skip_special_tokens": False,
                "stop": sa._STOP_STRINGS,
            },
        )
        choice = completion.choices[0]
        prompt_tokens = getattr(getattr(completion, "usage", None), "prompt_tokens", 0) or 0
        content = sa.strip_turn_delims(choice.message.content or "")
        return content, (choice.finish_reason or "stop"), int(prompt_tokens)
    except Exception as exc:
        if sa._is_context_overflow(exc):
            raise sa._ContextOverflow(str(exc)) from exc
        if sa._is_gateway_paused(exc):
            raise sa._GatewayPaused(str(exc)) from exc
        sa.log.error("LLM call failed: %s", exc)
        return "", "error", 0


def eval_meta_from_env() -> dict[str, Any]:
    raw = os.environ.get("AGL_EVAL_META", "")
    if raw.strip():
        return json.loads(raw)
    path = os.environ.get("AGL_INSTANCE_JSON", "/opt/agl_eval/instance.json")
    return json.loads(open(path, encoding="utf-8").read())


HINT_PATCH_CHAR_CAP = int(os.environ.get("SMITH_HINT_PATCH_CHARS", "8000"))


def build_hint(mode: str, meta: dict[str, Any]) -> str:
    """Privileged-information hint for the OPSD go/no-go diagnostic (10-06).

    Called AFTER checkout_bug_commit (HEAD = `Remove F2P Tests`, HEAD~1 = `Bug
    Patch`, HEAD~2 = clean main) and BEFORE relocate_git, so plain git works.
    `git diff HEAD~1 HEAD~2` is the reverse of the bug patch = the gold fix on
    source files only (the test deletion lives in HEAD, which is not in that range).
    Modes: none | tests (F2P node ids from the dataset, no git) | files (paths the
    fix touches) | patch (the gold diff itself, capped). Returns "" on failure so a
    broken hint degrades to the un-hinted condition instead of crashing the rollout.
    """
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return ""
    if mode == "tests":
        nodes = meta.get("FAIL_TO_PASS") or []
        if isinstance(nodes, str):
            try:
                nodes = json.loads(nodes)
            except Exception:
                nodes = [nodes]
        nodes = [str(n) for n in nodes][:20]
        if not nodes:
            return ""
        return (
            "<hint>\nThe following tests must pass after your fix (they are currently absent from the "
            "repository and will be restored at grading time; use their names to locate the relevant module):\n"
            + "\n".join(f"- {n}" for n in nodes)
            + "\n</hint>"
        )
    subj = sa._git_retry(["log", "-1", "--format=%s", "HEAD~1"], timeout=60, attempts=2)
    if subj is None or subj.returncode != 0 or "Bug Patch" not in (subj.stdout or ""):
        sa.log.error("hint: HEAD~1 is not the Bug Patch commit (%s); no hint", (subj.stdout if subj else "timeout"))
        return ""
    if mode == "files":
        proc = sa._git_retry(["diff", "--name-only", "HEAD~1", "HEAD~2"], timeout=60, attempts=2)
        if proc is None or proc.returncode != 0:
            return ""
        paths = [p for p in (proc.stdout or "").splitlines() if p.strip()]
        if not paths:
            return ""
        return "<hint>\nThe bug is located in the following file(s):\n" + "\n".join(f"- {p}" for p in paths) + "\n</hint>"
    if mode == "patch":
        proc = sa._git_retry(["diff", "HEAD~1", "HEAD~2"], timeout=60, attempts=2)
        if proc is None or proc.returncode != 0 or not (proc.stdout or "").strip():
            return ""
        diff = proc.stdout
        if len(diff) > HINT_PATCH_CHAR_CAP:
            diff = diff[:HINT_PATCH_CHAR_CAP] + "\n... [patch truncated]\n"
        return (
            "<hint>\nA reference patch that fixes the issue is given below. Apply the equivalent change to the "
            "files under /testbed by editing them directly (git is not available), verify, then submit:\n"
            "```diff\n" + diff.rstrip("\n") + "\n```\n</hint>"
        )
    sa.log.error("hint: unknown SMITH_HINT_MODE=%r; no hint", mode)
    return ""


def main() -> int:
    sa.logging.basicConfig(
        level=sa.logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[sa.logging.StreamHandler(sys.stdout)],
    )
    meta = eval_meta_from_env()
    instance_id = str(meta.get("instance_id") or "")
    problem = (os.environ.get("AGL_TASK_INPUT") or meta.get("problem_statement") or "").strip()
    if not instance_id or not problem:
        raise SystemExit("instance_id or problem_statement missing")
    base_url = os.environ["AGL_OPENAI_BASE_URL"]
    api_key = os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
    # Match OpenCode val: no turn cap (10 min wall binds), output 8192, context 40960 on vLLM.
    max_turns = int(os.environ.get("SMITH_MAX_TURNS", "10000"))
    cmd_timeout = int(os.environ.get("SMITH_CMD_TIMEOUT", "120"))
    obs_cap = int(os.environ.get("SMITH_OBS_CHAR_CAP", "6000"))
    max_tokens = int(os.environ.get("AGL_MAX_TOKENS", "8192"))
    max_format_errors = int(os.environ.get("SMITH_MAX_FORMAT_ERRORS", "3"))
    gateway_wait_s = float(os.environ.get("SMITH_GATEWAY_WAIT_S", "600"))
    out_path = os.environ.get("AGL_STATUS_OUT", "/opt/agl_eval/status.json")
    hint_mode = os.environ.get("SMITH_HINT_MODE", "none").strip().lower()

    sa._query = _query  # type: ignore[method-assign]
    client = OpenAI(base_url=base_url, api_key=api_key)
    t0 = time.time()
    status: dict[str, Any] = {
        "instance_id": instance_id,
        "submitted": False,
        "n_turns": 0,
        "max_prompt_tokens": 0,
        "overflowed": False,
        "error": None,
        "hint_mode": hint_mode,
        "hint_chars": 0,
    }
    try:
        sa.log.info("SmithAgent rollout start: instance=%s max_turns=%d hint=%s", instance_id, max_turns, hint_mode)
        sa.checkout_bug_commit(instance_id)
        if hint_mode != "none":
            hint = build_hint(hint_mode, meta)
            status["hint_chars"] = len(hint)
            if hint:
                problem = problem + "\n\n" + hint
                sa.log.info("hint injected: mode=%s chars=%d head=%r", hint_mode, len(hint), hint[:160])
            else:
                sa.log.error("hint requested (%s) but empty; running un-hinted", hint_mode)
        sa.relocate_git()
        submitted, n_turns, max_prompt_tokens = sa.run_agent_loop(
            client,
            problem,
            max_turns=max_turns,
            cmd_timeout=cmd_timeout,
            obs_cap=obs_cap,
            max_tokens=max_tokens,
            max_format_errors=max_format_errors,
            gateway_wait_s=gateway_wait_s,
        )
        status.update(
            {
                "submitted": submitted,
                "n_turns": n_turns,
                "max_prompt_tokens": max_prompt_tokens,
            }
        )
    except sa._ContextOverflow as exc:
        status["overflowed"] = True
        status["error"] = f"context overflow: {exc}"[:500]
        sa.log.warning("%s", status["error"])
    except SystemExit as exc:
        status["error"] = str(exc)[:800]
        sa.log.error("%s", status["error"])
        return 2
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"[:800]
        status["traceback"] = traceback.format_exc()[-2000:]
        sa.log.exception("rollout failed")
        return 1
    finally:
        status["wall_seconds"] = round(time.time() - t0, 1)
        open(out_path, "w", encoding="utf-8").write(json.dumps(status, ensure_ascii=False, indent=2))
        sa.log.info("wrote %s %s", out_path, json.dumps({k: status[k] for k in status if k != "traceback"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
