#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.


from __future__ import annotations

import http.client
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

TESTBED = "/testbed"
CHECKOUT_JITTER_SECONDS = 30.0

# /testbed ships full git history (the fix + deleted F2P tests are recoverable),
# a reward-hack route. Relocate .git out of the worktree (O(1) rename); the
# harness still reaches it via --git-dir/--work-tree. See also _forbidden_action.
_HIDDEN_GIT_DIR = os.environ.get("SMITH_HIDDEN_GIT_DIR", "/opt/agl_tmp")

_STATUS_RE = re.compile(
    r"(?:^|\s)(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)|(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s|$)"
)

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# Accept both the canonical ```bash fence and mini-swe-agent's ```mswea_bash_command.
_ACTION_RE = re.compile(r"```(?:bash|mswea_bash_command)[^\S\n]*\n(.*?)\n?```", re.DOTALL)

# MiniCPM5's *native* tool-call syntax, which our SFT policy emits instead of a
# ```bash fence. This is not model noise, it is what we trained: the teacher
# trajectories keep tool calls in the OpenAI ``tool_calls`` field, and
# swe_smith_smoke/sft/swe_sft_dataset.py:35-44,64-67 renders them through
# chat_template.jinja, which turns every one into
#   <function name="bash"><param name="command">…</param></function>
# Measured over the 858 training rows: 13254 native calls inside the 12528
# loss-masked assistant turns, against **0** ```bash fences. Accepting the syntax
# here keeps the action space identical (still exactly one shell command) — it
# only stops a syntax difference from being scored as a failed episode.
# chat_template.jinja:8,75 also tells the model to wrap a value containing "<",
# "&" or a newline in CDATA, and 43.2% of the native calls we recorded do.
_NATIVE_CALL_RE = re.compile(r"<function\s+name=\"([^\"]*)\"\s*>(.*?)</function\s*>", re.DOTALL)
_NATIVE_PARAM_RE = re.compile(r"<param\s+name=\"([^\"]*)\"\s*>(.*?)</param\s*>", re.DOTALL)
_CDATA_RE = re.compile(r"\A\s*<!\[CDATA\[(.*?)\]\]>\s*\Z", re.DOTALL)
# Names that mean "run this shell command". The teacher only ever used "bash";
# the rest are what the policy drifts to once it is improvising.
_NATIVE_BASH_NAMES = frozenset(
    {"bash", "sh", "shell", "exec", "execute", "execute_bash", "code_exec", "run", "run_bash", "command", "terminal"}
)
_NATIVE_CMD_PARAMS = ("command", "cmd", "script", "command_line", "commands", "input")
# Tokens that only ever delimit a chat turn. With skip_special_tokens=False (see
# _query) the server hands back the stop token inside ``content``, and that text
# is appended to ``messages`` verbatim, so it has to come off first.
_TURN_DELIMS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")

# The turn boundary, passed explicitly as a stop string.
#
# 2026-10-05 RETRACTION: this used to claim vLLM 0.8.5's OpenAI server does not
# turn generation_config.json's ``eos_token_id: [1, 130073]`` into a stop
# condition, citing "16/16 replies ran to max_tokens with 9-239 <|im_end|> in
# them, median 2236 chars". That measurement was taken with n>1 and read as if
# each result were one reply — the samples had been concatenated. Re-measured at
# n=1, the server DOES stop at <|im_end|> by itself.
#
# So this is harmless insurance, not a repair: it costs nothing and pins the
# boundary explicitly however the server is configured. Do not cite it as the
# reason §54's 2.4x reward drop was fixed — that was skip_special_tokens=False
# alone. See docs/pitfalls-and-lessons.md, the correction box in §54.
_STOP_STRINGS = [s for s in os.environ.get("SMITH_STOP", "<|im_end|>").split(",") if s]

# Block git use/metadata access. `git` matched only in command position (after
# ; && || | ( ` $( and VAR=val prefixes) so args/prose containing "git" pass.
_GIT_INVOKE_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:[\w./-]*/)?git(?:-[a-z]+)?(?=\s|$|;|&|\|)",
    re.I,
)
_GIT_ACCESS_RE = re.compile(r'--git-dir|--work-tree|(?:^|[\s=:"\'/])\.git(?:/|\b)', re.I)

# With git blocked, the model pivots to fetching upstream source over the network
# (curl/wget, pip install, python urllib). Block these; same command-position
# anchoring as _GIT_INVOKE_RE. Code-level backstop for the egress NetworkPolicy.
_NET_FETCH_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:[\w./-]*/)?"
    r"(?:curl|wget|httpie|http|https|aria2c|scp|sftp|rsync|nc|ncat|netcat|telnet)"
    r"(?=\s|$|;|&|\|)",
    re.I,
)
# Any install can pull the target package's correct source. Match
# pip/pip3/conda/mamba/uv/easy_install install and `python -m pip install`.
_PKG_INSTALL_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:"
    r"(?:[\w./-]*/)?(?:pip|pip3|conda|mamba|easy_install|uv)\b[^\n;&|]*?\binstall\b"
    r"|(?:[\w./-]*/)?python[0-9.]*\s+-m\s+pip\b[^\n;&|]*?\binstall\b)",
    re.I,
)
# Python one-liners that reach the network -- a route around curl/wget.
_PY_NET_RE = re.compile(
    r"urllib\.request|\burlopen\b|\brequests\.(?:get|post|put|head|Session)\b|"
    r"\bhttpx\.|\bsocket\.(?:socket|create_connection)\b|\burllib3\b",
    re.I,
)
# Writing test-harness/config files (conftest, .pth, sitecustomize) can force
# PASS or patch imports, bypassing evaluate(). Match writes TO these files only.
_TEST_TAMPER_RE = re.compile(
    r'(?:>>?|\btee\b(?:\s+-a)?\s+)\s*[\'"]?[^\s\'"|;&<>]*'
    r"(?:conftest\.py|pytest\.ini|tox\.ini|sitecustomize\.py|usercustomize\.py|"
    r'setup\.cfg|pyproject\.toml|\.pth)(?=[\'"\s;&|]|$)',
    re.I,
)

# Quiet pagers/progress bars so one command can't flood the observation.
_CMD_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

SYSTEM_PROMPT = """\
You are a helpful assistant that can interact multiple times with a computer \
shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or \
commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning. \
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""

INSTANCE_PROMPT = """\
<pr_description>
Consider the following PR description:
{problem_statement}
</pr_description>

<instructions>
# Task Instructions

## Overview
You're a software engineer interacting continuously with a computer by submitting \
commands. You'll be helping implement the changes needed to satisfy the PR \
description. Your task is specifically to make changes to non-test files in the \
/testbed directory in order to fix the issue in a way that is general and \
consistent with the codebase.

<IMPORTANT>This is an interactive process: you think and issue ONE command, see \
its result, then think and issue your next command.</IMPORTANT>

For each response:
1. Include a THOUGHT section explaining your reasoning and goal.
2. Provide exactly ONE bash command to execute.

## Important Boundaries
- MODIFY: regular source files in /testbed (the working directory for every command).
- DO NOT MODIFY: tests or configuration files (pyproject.toml, setup.cfg, etc.).

## Recommended Workflow
1. Find and read the files relevant to the issue.
2. Create a script to reproduce the issue.
3. Edit the source code to resolve the issue.
4. Re-run your script to verify the fix.
5. Test edge cases to make sure the fix is robust.

## Command Execution Rules
You operate in a loop:
1. You write a single command.
2. The system executes it in a fresh subshell.
3. You see the result.
4. You write your next command.

**CRITICAL REQUIREMENTS:**
- Your response MUST contain EXACTLY ONE bash code block.
- That block MUST contain EXACTLY ONE command (or several joined with && or ||).
- If you include zero or multiple bash blocks, YOUR RESPONSE WILL FAIL.
- Do NOT put several independent commands in separate blocks in one response.
- Directory and environment changes are NOT persistent: every action runs in a new
  subshell. Prefix an action with `MY_VAR=value cd /testbed && ...`, or read/write
  state from files, if you need persistence.

Example of a CORRECT response:
<example_response>
THOUGHT: I need to understand the repository layout first, so let me list the files.

```bash
ls -la
```
</example_response>

## Environment Details
- You have a full Linux shell environment.
- Always use non-interactive flags (-y, -f).
- Avoid interactive tools (vi, nano, ...) that wait for input.
- The NETWORK is DISABLED. Do NOT fetch anything from the internet -- no curl,
  wget, httpie, scp, nc, and no Python urllib/requests/httpx/socket. Every
  dependency you need is ALREADY installed.
- Do NOT install packages (no pip / pip3 / conda / uv install). Everything
  required to run the code and its tests is already present.
- Do NOT modify test-harness or config files: conftest.py, pytest.ini, tox.ini,
  setup.cfg, pyproject.toml, sitecustomize.py, *.pth. Fix the bug in the source.
- `git` is DISABLED here and the `.git` directory is unavailable. Do not use git
  or try to read git metadata; read and edit the source files directly.

## Submission
When your fix is complete, submit with this EXACT command on its own. The system
captures your changes to /testbed automatically -- you do NOT need git:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

<CRITICAL>
- git is disabled; do not use it for any purpose.
- The network is disabled; do not curl/wget/pip-install or otherwise fetch code.
  Solve the bug ONLY from the source files already under /testbed.
- You CANNOT continue working on this task after submitting.
</CRITICAL>
</instructions>"""

_OVERLONG_WARNING = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can use head, tail or sed to view a smaller "
    "number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, use a more "
    "selective search pattern.\n"
    "If you really need the full output, redirect it to a file and search within it."
)


class FormatError(Exception):
    """Model reply did not contain exactly one bash action."""

    def __init__(self, n_actions: int) -> None:
        super().__init__(f"expected exactly one action, found {n_actions}")
        self.n_actions = n_actions


class UnavailableToolError(FormatError):
    """Reply held a well-formed native call for a tool this harness does not have.

    Subclasses ``FormatError`` on purpose: the caller's consecutive-error counting
    and episode termination stay exactly as they were, only the message the model
    gets back is more specific. SFT taught six tools we do not provide (read 3270,
    edit 1178, grep 975, glob 81, write 20, webfetch 12 calls), so this is the
    common case, not a corner case. They are deliberately **not** translated into
    shell equivalents — that would widen the action space and make the score
    incomparable with the mini-swe-agent baselines.
    """

    def __init__(self, tools: list[str]) -> None:
        super().__init__(len(tools))
        self.tools = tools


def _unwrap_cdata(value: str) -> str:
    """Strip one CDATA wrapper, which chat_template.jinja:8 tells the model to add."""
    m = _CDATA_RE.match(value)
    return m.group(1) if m else value


def strip_turn_delims(content: str) -> str:
    """Cut the reply at the first chat-delimiter special token, if any.

    Every token in ``_TURN_DELIMS`` ends the assistant turn, so nothing after the
    earliest one belongs to this reply.
    """
    cut = len(content)
    for tok in _TURN_DELIMS:
        idx = content.find(tok)
        if idx != -1:
            cut = min(cut, idx)
    return content[:cut]


def _native_actions(content: str) -> tuple[list[str], list[str]]:
    """Split native ``<function>`` calls into shell commands and unavailable tools."""
    commands: list[str] = []
    others: list[str] = []
    for name, body in _NATIVE_CALL_RE.findall(content):
        name = name.strip()
        if name not in _NATIVE_BASH_NAMES:
            others.append(name)
            continue
        params = {k.strip(): _unwrap_cdata(v) for k, v in _NATIVE_PARAM_RE.findall(body)}
        for key in _NATIVE_CMD_PARAMS:
            if key in params:
                commands.append(params[key])
                break
        else:
            # <function name="bash"><![CDATA[cmd]]></function> — the param wrapper
            # dropped but the value unambiguous. Anything else is a real malformation.
            cdata = _CDATA_RE.match(body)
            if cdata is not None and cdata.group(1).strip():
                commands.append(cdata.group(1))
            else:
                others.append(name)
    return commands, others


def parse_action(content: str) -> str:
    """Return the single shell action from a model reply.

    Accepts the ```bash fence and MiniCPM5's native ``<function name="bash">``
    syntax as two spellings of the same action. Raises ``UnavailableToolError``
    when the only thing found is a native call for a tool we do not provide, and
    ``FormatError`` whenever the count of actions is not exactly one.
    """
    actions = _ACTION_RE.findall(content)
    commands, others = _native_actions(content)
    actions = actions + commands
    if len(actions) == 1 and not others:
        return actions[0].strip()
    if not actions and others:
        raise UnavailableToolError(others)
    raise FormatError(len(actions) + len(others))


def is_submission(output: str) -> bool:
    """True when command output begins with the submit marker."""
    stripped = output.lstrip()
    if not stripped:
        return False
    return stripped.split("\n", 1)[0].strip() == SUBMIT_MARKER


def render_observation(returncode: int, output: str, obs_cap: int) -> str:
    """Format a command result the way mini-swe-agent does.

    Output shorter than ``obs_cap`` is shown verbatim; longer output is elided to
    a head+tail window with a warning so one noisy command cannot blow up context.
    """
    if len(output) < obs_cap:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}</output>"
    half = obs_cap // 2
    elided = len(output) - obs_cap
    return (
        f"<returncode>{returncode}</returncode>\n"
        f"<warning>\n{_OVERLONG_WARNING}\n</warning>\n"
        f"<output_head>\n{output[:half]}\n</output_head>\n"
        f"<elided_chars>\n{elided} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{output[-half:]}\n</output_tail>"
    )


def _forbidden_action(action: str) -> str | None:
    """Return a rejection reason if the action cheats instead of solving the bug.

    Blocks four reward-hacking routes so the agent must fix the bug from the
    local /testbed source only:
      - git (relocate_git() moved the repo; history holds the fix + deleted tests)
      - network fetches (curl/wget/python-http) that download the upstream code
      - package installs (pip/conda) that pull the target package's correct source
      - writing test-harness files (conftest/pytest.ini/sitecustomize) to force PASS
    Returns None for allowed actions. Network/install/tamper blocks are a code
    backstop; the authoritative fix is a default-deny egress NetworkPolicy.
    """
    if _GIT_INVOKE_RE.search(action):
        return (
            "git is disabled in this environment; do not use it for any "
            "purpose. Inspect and edit the source files under /testbed "
            "directly (cat, grep, sed, python) to fix the bug."
        )
    if _GIT_ACCESS_RE.search(action) or _HIDDEN_GIT_DIR in action:
        return (
            "Accessing the git metadata directory is not allowed. Work only "
            "with the source files under /testbed; do not read .git."
        )
    if _NET_FETCH_RE.search(action) or _PY_NET_RE.search(action):
        return (
            "Network access is disabled. Do not fetch code from the internet "
            "(curl, wget, urllib, requests, etc.); all dependencies are "
            "already installed. Solve the bug using only the source files "
            "already present under /testbed."
        )
    if _PKG_INSTALL_RE.search(action):
        return (
            "Installing packages is not allowed. Everything needed to run the "
            "code and its tests is already installed. Fix the bug by editing "
            "the source under /testbed; do not install anything."
        )
    if _TEST_TAMPER_RE.search(action):
        return (
            "Modifying test-harness or config files (conftest.py, pytest.ini, "
            "tox.ini, setup.cfg, pyproject.toml, sitecustomize.py, .pth) is "
            "not allowed. Fix the bug in the source under /testbed instead."
        )
    return None


def format_error_message(n_actions: int, finish_reason: str) -> str:
    """Nudge the model back to the one-bash-block format after a parse failure."""
    if finish_reason in ("length", "tool_calls"):
        return (
            f"Your previous response reached the output token limit "
            f"(finish_reason={finish_reason}) before you produced a complete action, "
            "so it was cut off. Respond more concisely and provide exactly one action "
            "in the required format. If you need to think more, do so briefly."
        )
    return (
        "Format error:\n\n"
        f"<error>\nExpected EXACTLY ONE bash code block, found {n_actions}.\n</error>\n\n"
        "Please always provide EXACTLY ONE action in a single triple-backtick bash "
        "block, as shown in <response_example>.\n\n"
        "<response_example>\n"
        "THOUGHT: brief reasoning about the action you want to perform.\n\n"
        "```bash\n<your_command>\n```\n"
        "</response_example>\n\n"
        "If you have completed your assignment, consult the first message about how "
        "to submit your solution."
    )


def unavailable_tool_message(tools: list[str]) -> str:
    """Tell the model the tool it called does not exist here, and name what does.

    The generic format error above is actively misleading for this case: the reply
    *was* a well-formed tool call, just for a tool (``read``, ``edit``, ``grep``,
    ``glob``, ``write``, ``webfetch``) that only existed in the teacher's
    environment. Naming the tool and the one available action costs a turn instead
    of three, and does not change what the agent is able to do.
    """
    named = ", ".join(f"`{t}`" for t in dict.fromkeys(tools)) or "that tool"
    return (
        "Tool error:\n\n"
        f"<error>\nNo such tool: {named}. This environment provides exactly one "
        "action: running a single shell command inside /testbed.\n</error>\n\n"
        "Everything you need is reachable from the shell — read a file with `sed -n`, "
        "search with `grep -rn`, edit with `python - <<'PY'`. Re-issue your action as "
        "ONE shell command, in either of these two accepted spellings:\n\n"
        "```bash\n<your_command>\n```\n\n"
        '<function name="bash"><param name="command"><your_command></param></function>\n'
    )


def _agent_env() -> dict[str, str]:
    """Environment for agent shell commands.

    Hides the relocated git directory from the agent: drops SMITH_HIDDEN_GIT_DIR
    and any variable whose value leaks the path (so ``env`` reveals nothing about
    where .git went), then applies the quiet-pager/progress overrides.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "SMITH_HIDDEN_GIT_DIR" and _HIDDEN_GIT_DIR not in value
    }
    env.update(_CMD_ENV)
    return env


def _run(command: str, timeout: int) -> tuple[str, int]:
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=TESTBED,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_agent_env(),
        )
        return proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]", 124
    except Exception as exc:
        return f"[failed to start: {exc}]", 1


def post_event(event_type: str, data: dict[str, Any], retry: bool = False) -> None:
    event_url = os.environ.get("AGL_EVENT_URL")
    if not event_url:
        log.warning("AGL_EVENT_URL not set; skip %s event", event_type)
        return
    body = json.dumps({"event_type": event_type, "data": data}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    agl_key = os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY")
    if agl_key:
        headers["Authorization"] = f"Bearer {agl_key}"
    request = urllib.request.Request(event_url, data=body, headers=headers, method="POST")
    attempt = 0
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                response.read()
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if not retry:
                log.warning("failed to post %s event: %s", event_type, exc)
                return
            backoff = min(2 ** min(attempt, 5), 30)
            log.warning(
                "failed to post %s event (attempt %d): %s; retrying in %ds",
                event_type,
                attempt,
                exc,
                backoff,
            )
            time.sleep(backoff)


def fetch_eval_meta() -> dict[str, Any]:
    event_url = os.environ.get("AGL_EVENT_URL", "")
    match = re.match(r"^(?P<base>.*)/api/rollouts/(?P<rid>[^/]+)/attempt/", event_url)
    if not match:
        log.error("cannot derive rollout id from AGL_EVENT_URL=%r; eval meta unavailable", event_url)
        return {}
    url = f"{match.group('base')}/api/rollouts/{match.group('rid')}"
    headers = {}
    agl_key = os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY")
    if agl_key:
        headers["Authorization"] = f"Bearer {agl_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    payload = None
    attempts = 4
    for i in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
            json.JSONDecodeError,
        ) as exc:
            log.warning("fetch eval meta failed (attempt %d/%d): %s", i + 1, attempts, exc)
            if i < attempts - 1:
                time.sleep(min(2**i, 8))
    if payload is None:
        log.error("failed to fetch eval meta from server after %d attempts", attempts)
        return {}
    inp = (payload.get("rollout") or {}).get("input") or {}
    return {
        "instance_id": inp.get("instance_id", ""),
        "repo": inp.get("repo", ""),
        "FAIL_TO_PASS": inp.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": inp.get("PASS_TO_PASS", []),
    }


def _git_dir() -> str:
    """Path to the live git directory (relocated once the agent loop is set up)."""
    return _HIDDEN_GIT_DIR if os.path.isdir(_HIDDEN_GIT_DIR) else os.path.join(TESTBED, ".git")


def _git_base() -> list[str]:
    """Base argv for harness git calls.

    Before relocation .git is in-tree, so a bare ``git`` (cwd=/testbed) works.
    After relocate_git() moves it out of the worktree, every harness git call
    must target it explicitly with --git-dir/--work-tree; safe.directory avoids
    git's dubious-ownership refusal now that git-dir and work-tree differ.
    """
    if os.path.isdir(_HIDDEN_GIT_DIR):
        return ["git", "--git-dir", _HIDDEN_GIT_DIR, "--work-tree", TESTBED, "-c", "safe.directory=*"]
    return ["git"]


def _remove_stale_git_index_lock() -> None:
    lock_path = os.path.join(_git_dir(), "index.lock")
    if not os.path.exists(lock_path):
        return
    try:
        os.unlink(lock_path)
        log.warning("removed stale git index lock: %s", lock_path)
    except OSError as exc:
        log.warning("failed to remove stale git index lock %s: %s", lock_path, exc)


def _proc_output(proc: subprocess.CompletedProcess) -> str:
    return "\n".join(part for part in (proc.stdout, proc.stderr) if part).strip()


def _git_retry(args: list[str], *, timeout: int = 120, attempts: int = 3) -> subprocess.CompletedProcess | None:
    """Run a git command in /testbed with retries on timeout/failure.

    Under high pod concurrency the node is heavily oversubscribed and IO-bound
    git operations intermittently time out; a few retries with backoff clears
    the transient failures. Returns the CompletedProcess of the last attempt
    (caller checks returncode), or None if every attempt raised.
    """
    last: subprocess.CompletedProcess | None = None
    for i in range(attempts):
        _remove_stale_git_index_lock()
        if args and args[0] == "checkout":
            delay = random.uniform(0.0, CHECKOUT_JITTER_SECONDS)
            log.info("sleeping %.2fs before git checkout", delay)
            time.sleep(delay)
        try:
            proc = subprocess.run([*_git_base(), *args], cwd=TESTBED, capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 0:
                return proc
            last = proc
            if "index.lock" in _proc_output(proc):
                _remove_stale_git_index_lock()
            log.warning(
                "git %s rc=%d (attempt %d/%d): %s", args[0], proc.returncode, i + 1, attempts, _proc_output(proc)[:200]
            )
        except subprocess.TimeoutExpired:
            last = None
            _remove_stale_git_index_lock()
            log.warning("git %s timed out after %ds (attempt %d/%d)", args[0], timeout, i + 1, attempts)
        if i < attempts - 1:
            time.sleep(min(2**i, 8))
    return last


def checkout_bug_commit(instance_id: str) -> None:
    """Bring the injected bug into /testbed using SWE-smith's official checkout.

    SWE-smith builds each instance as a branch `<instance_id>` of two
    commits on top of `main`: `Bug Patch` (injects the bug) then `Remove F2P
    Tests` (deletes the FAIL_TO_PASS test files). The image defaults to clean
    `main`, so without this the agent sees fixed code and evaluate() false-passes.
    """
    if not instance_id:
        raise SystemExit("cannot checkout bug commit: instance_id is empty")
    proc = _git_retry(["checkout", instance_id], timeout=120)
    if proc is None or proc.returncode != 0:
        reason = "timeout" if proc is None else _proc_output(proc)
        raise SystemExit(f"git checkout {instance_id} failed: {reason}")
    log.info("checked out SWE-smith bug branch in /testbed: %s", instance_id)


def relocate_git() -> None:
    """Move /testbed/.git out of the agent's working tree (call after checkout).

    This closes the reward-hacking leak where the agent recovers the injected-bug
    fix (and the deleted FAIL_TO_PASS tests) from git history via
    `git checkout <pre-bug-sha> -- <src>` / `git show`/`git log -p`. Once .git is
    gone from /testbed the worktree is no longer a repo, so the agent's git calls
    fail; the harness still reaches history through _git_base() (--git-dir). A
    same-filesystem move is an O(1) rename (no blob IO); a cross-fs move falls
    back to a copy. The agent is also forbidden from invoking git or reading the
    relocated directory (see _forbidden_action).
    """
    src = os.path.join(TESTBED, ".git")
    if not os.path.isdir(src):
        log.warning("relocate_git: %s missing; git-history leak NOT closed", src)
        return
    if os.path.exists(_HIDDEN_GIT_DIR):
        shutil.rmtree(_HIDDEN_GIT_DIR, ignore_errors=True)
    parent = os.path.dirname(_HIDDEN_GIT_DIR) or "/"
    os.makedirs(parent, exist_ok=True)
    try:
        same_fs = os.stat(src).st_dev == os.stat(parent).st_dev
    except OSError:
        same_fs = False
    shutil.move(src, _HIDDEN_GIT_DIR)
    log.info("relocated .git -> %s (%s)", _HIDDEN_GIT_DIR, "rename O(1)" if same_fs else "cross-fs copy")


def restore_f2p_tests(instance_id: str, test_nodes: list[str]) -> None:
    """Restore the test files deleted by `Remove F2P Tests`, keeping the agent's fix.

    The branch HEAD has the FAIL_TO_PASS test files deleted; the parent commit
    (`HEAD~1`, the `Bug Patch` commit) still has them. We restore
    just those test paths from `~1` into the working tree -- this overwrites only
    the test files, leaving the agent's source edits intact -- so evaluate() can
    run the real FAIL_TO_PASS/PASS_TO_PASS suite against the agent's fix.
    """
    paths = sorted({node.split("::", 1)[0] for node in test_nodes if node})
    if not paths or not instance_id:
        return
    proc = _git_retry(["checkout", "HEAD~1", "--", *paths], timeout=120)
    if proc is None or proc.returncode != 0:
        log.warning("restore_f2p_tests failed: %s", getattr(proc, "stderr", "timeout"))


def capture_patch() -> str:
    try:
        proc = subprocess.run(
            [*_git_base(), "-c", "core.fileMode=false", "diff", "HEAD"],
            cwd=TESTBED,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.stdout
    except Exception as exc:
        log.error("patch capture failed: %s", exc)
        return ""


def parse_test_statuses(test_output: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in test_output.splitlines():
        m = _STATUS_RE.search(line)
        if not m:
            continue
        if m.group(1):
            status, node = m.group(1), m.group(2)
        else:
            node, status = m.group(3), m.group(4)
        statuses[node] = status
    return statuses


def evaluate(
    eval_meta: dict[str, Any], timeout: int, f2p_only: bool = True
) -> tuple[float, bool, str, bool, float]:
    """Grade a rollout. Returns ``(binary, resolved, reason, timed_out, f2p_ratio)``.

    ``binary`` is the historical solved-or-not score and stays the VALIDATION
    metric, so every val number ever reported (base 123/474, SFT 136/474) remains
    comparable. ``f2p_ratio`` is the dense training signal: the fraction of
    FAIL_TO_PASS tests that pass, forced to 0.0 whenever the rollout timed out or
    broke a PASS_TO_PASS, so ``f2p_ratio == 1.0`` iff ``resolved``. The caller
    picks which one to use (see ``SMITH_F2P_PARTIAL``).
    """
    fail_to_pass = list(eval_meta.get("FAIL_TO_PASS", []))
    pass_to_pass = list(eval_meta.get("PASS_TO_PASS", []))
    if not fail_to_pass:
        return 0.0, False, "no FAIL_TO_PASS in eval meta", False, 0.0

    # f2p_only (SWE-smith --f2p_only): grade only on the test FILES holding the
    # F2P tests. F2P kept whole; P2P filtered to those same files (dropping the
    # unrelated cross-file P2P that cause eval timeouts). Resolve = all F2P + P2P pass.
    if f2p_only:
        f2p_files = sorted({t.split("::", 1)[0] for t in fail_to_pass})
        pass_to_pass = [t for t in pass_to_pass if any(t.startswith(f) for f in f2p_files)]

    nodes = fail_to_pass + pass_to_pass
    restore_f2p_tests(eval_meta.get("instance_id", ""), nodes)
    # Cap pytest at 4 workers: the project's own `-n auto` reads HOST cores (dozens),
    # ignores the pod cgroup, and spawns ~70 workers that OOMKill the 4Gi pod. `-n`
    # needs xdist, so probe first and fall back to serial `-p no:xdist` when absent.
    xdist_flag = "-n4" if _run("python -c 'import xdist'", 30)[1] == 0 else "-p no:xdist"
    output, rc = _run(
        "python -m pytest -rA -p no:cacheprovider " + xdist_flag + " " + " ".join(map(_shq, nodes)),
        timeout,
    )
    # rc 124 == subprocess.TimeoutExpired (see _run); pytest itself never exits 124.
    timed_out = rc == 124 or "[timed out after" in output
    statuses = parse_test_statuses(output)
    f2p_pass = [t for t in fail_to_pass if statuses.get(t) in ("PASSED", "XFAIL")]
    # only count P2P reporting PASSED/XFAIL — a missing status means it never ran.
    p2p_ok = [t for t in pass_to_pass if statuses.get(t) in ("PASSED", "XFAIL")]
    resolved = not timed_out and len(f2p_pass) == len(fail_to_pass) and len(p2p_ok) == len(pass_to_pass)
    prefix = f"EVAL TIMEOUT after {timeout}s — " if timed_out else ""
    suffix = " (f2p_only)" if f2p_only else ""
    reason = (
        f"{prefix}FAIL_TO_PASS {len(f2p_pass)}/{len(fail_to_pass)} passed, "
        f"PASS_TO_PASS {len(p2p_ok)}/{len(pass_to_pass)} ok{suffix}"
    )
    # Dense signal. Gated on the same two hard failures as `resolved` (timeout,
    # P2P regression) so a partial F2P score can never be earned by breaking the
    # rest of the suite, and so ratio 1.0 and `resolved` cannot disagree.
    f2p_ratio = 0.0 if (timed_out or len(p2p_ok) != len(pass_to_pass)) else len(f2p_pass) / len(fail_to_pass)
    return (1.0 if resolved else 0.0), resolved, reason, timed_out, f2p_ratio


def _shq(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


class _ContextOverflow(Exception):
    pass


class _GatewayPaused(Exception):
    pass


_OVERFLOW_MARKERS = ("maximum context length", "'max_tokens' is too large")


def _is_context_overflow(exc: Exception) -> bool:

    if getattr(exc, "status_code", None) != 400:
        return False
    message = str(getattr(exc, "message", None) or exc).lower()
    return any(marker in message for marker in _OVERFLOW_MARKERS)


def _is_gateway_paused(exc: Exception) -> bool:
    # async weight-sync pauses the proxy (429 gateway paused). Transient — callers
    # wait it out instead of burning a turn.
    if getattr(exc, "status_code", None) != 429:
        return False
    return "gateway paused" in str(getattr(exc, "message", None) or exc).lower()


def length_penalized_reward(
    reward: float, n_turns: int, max_turns: int, *, t0: int, lam: float, is_train: bool
) -> float:
    """Apply the long-turn penalty (plan A) to a SOLVED *training* trajectory's reward.

    The penalty applies **only** when ``is_train`` is True and ``reward >= 1.0``
    (solved). This keeps VALIDATION reward the true, unshaped metric (validation
    drives checkpoint selection, so it must never be reshaped), and leaves
    unsolved rollouts full exploration room. The penalty is a linear soft ramp
    over the turn budget::

        reward = 1.0 - lam * clip((n_turns - t0) / (max_turns - t0), 0, 1)

    A train solve within ``t0`` turns keeps reward ``1.0``; one that runs to the
    cap gets ``1 - lam``. Validation rewards and non-solved rewards are returned
    unchanged.
    """
    if not is_train or reward < 1.0 or n_turns <= t0:
        return reward
    span = max(1, max_turns - t0)
    frac = min((n_turns - t0) / span, 1.0)
    return 1.0 - lam * frac


def prompt_length_penalty(
    reward: float,
    max_prompt_tokens: int,
    *,
    soft_start: int,
    hard_cap: int,
    max_pen: float,
    is_train: bool,
    solved: bool,
) -> float:
    """Penalizes context bloat: the longest single-turn prompt of a rollout is its
    true upper bound on context pressure (each turn's prompt embeds all prior
    history). Gated identically to the turn penalty — only ``is_train`` and
    ``solved`` — so VALIDATION reward stays the unshaped metric and unsolved
    rollouts are untouched. ``solved`` is the *raw* solved status (not the running
    reward), so this composes with the turn penalty even after it lowered the
    reward below 1.0. Soft linear ramp over the context band::

        penalty = max_pen * clip((max_prompt_tokens - soft_start) /
                                 (hard_cap - soft_start), 0, 1)
        reward -= penalty

    A solve whose longest prompt is <= ``soft_start`` keeps its reward; at or
    above ``hard_cap`` it loses the full ``max_pen``. Stacks with (subtracts on
    top of) the turn penalty.
    """
    if not is_train or not solved or max_prompt_tokens <= soft_start:
        return reward
    if max_prompt_tokens >= hard_cap:
        return reward - max_pen
    span = max(1, hard_cap - soft_start)
    frac = min((max_prompt_tokens - soft_start) / span, 1.0)
    return reward - max_pen * frac


def run_agent_loop(
    client: Any,
    problem: str,
    *,
    max_turns: int,
    cmd_timeout: int,
    obs_cap: int,
    max_tokens: int,
    max_format_errors: int = 3,
    gateway_wait_s: float = 600.0,
    gateway_poll_s: float = 5.0,
) -> tuple[bool, int, int]:
    """Drive the agent until it submits, errors out, or hits ``max_turns``.

    Mirrors mini-swe-agent's control flow: each turn we query the model, parse a
    single bash action from its free-text reply, run it in a fresh /testbed
    subshell, and append a templated observation.

    Returns ``(submitted, turns_used, max_prompt_tokens)`` where ``submitted`` is
    ``True`` iff the model emitted the submit marker, ``turns_used`` is the number
    of model completions consumed (one per turn; matches ``len(rollout.triplets)``
    / wandb ``training/n_turns``) and equals ``max_turns`` when the loop runs to
    the cap, and ``max_prompt_tokens`` is the largest ``usage.prompt_tokens`` over
    all successful requests (0 if none succeeded). ``turns_used`` feeds the turn
    penalty and ``max_prompt_tokens`` the prompt-length penalty in ``main``.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTANCE_PROMPT.format(problem_statement=problem)},
    ]
    n_format_errors = 0
    turns_used = 0
    max_prompt_tokens = 0

    for turn in range(max_turns):
        # Query the model; retry in place (no turn consumed) while the gateway is
        # paused for weight sync.
        paused_for = 0.0
        while True:
            try:
                content, finish_reason, prompt_tokens = _query(client, messages, max_tokens)
                break
            except _ContextOverflow as exc:
                # Prompt no longer fits: end without appending a turn so the
                # trajectory stays a clean token-level prefix chain.
                log.warning("turn=%d context overflow, ending episode: %s", turn, exc)
                return False, turns_used, max_prompt_tokens
            except _GatewayPaused as exc:
                if paused_for >= gateway_wait_s:
                    log.warning("turn=%d gateway paused >%.0fs, giving up: %s", turn, gateway_wait_s, exc)
                    content, finish_reason, prompt_tokens = "", "error", 0
                    break
                log.info(
                    "turn=%d gateway paused, waiting %.0fs (waited %.0fs): %s", turn, gateway_poll_s, paused_for, exc
                )
                time.sleep(gateway_poll_s)
                paused_for += gateway_poll_s

        # A completion was consumed this turn (including burned error turns).
        turns_used += 1

        if finish_reason == "error":
            # Transient API failure (already retried): burn the turn and continue.
            log.warning("turn=%d model query failed; burning turn", turn)
            messages.append({"role": "assistant", "content": ""})
            continue

        # Successful request: track the largest prompt (last turn is usually it).
        if prompt_tokens > max_prompt_tokens:
            max_prompt_tokens = prompt_tokens

        messages.append({"role": "assistant", "content": content})

        try:
            action = parse_action(content)
        except FormatError as exc:
            n_format_errors += 1
            if 0 < max_format_errors <= n_format_errors:
                log.warning("turn=%d %d consecutive format errors, ending episode", turn, n_format_errors)
                return False, turns_used, max_prompt_tokens
            log.info("turn=%d format error (%d/%d)", turn, n_format_errors, max_format_errors)
            if isinstance(exc, UnavailableToolError):
                reply = unavailable_tool_message(exc.tools)
            else:
                reply = format_error_message(exc.n_actions, finish_reason)
            messages.append({"role": "user", "content": reply})
            continue
        n_format_errors = 0

        reason = _forbidden_action(action)
        if reason is not None:
            log.info("turn=%d blocked disallowed action: %r", turn, action[:200])
            messages.append({"role": "user", "content": render_observation(1, reason, obs_cap)})
            continue

        output, rc = _run(action, cmd_timeout)
        log.info("turn=%d cmd=%r rc=%d", turn, action[:200], rc)
        if is_submission(output):
            log.info("turn=%d submit marker detected", turn)
            return True, turns_used, max_prompt_tokens
        messages.append({"role": "user", "content": render_observation(rc, output, obs_cap)})
    return False, turns_used, max_prompt_tokens


class _StdlibHttpError(Exception):
    """Mirrors openai.APIStatusError's surface for _is_context_overflow/_is_gateway_paused."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class _StdlibCompletions:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        # Never route the gateway call through an inherited http_proxy: urllib honours
        # those env vars and a corporate proxy answers 502 for an internal host, which
        # would look like a model failure. An empty ProxyHandler pins it to direct.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_body:
            payload.update(extra_body)
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            raise _StdlibHttpError(int(exc.code), raw or str(exc)) from exc
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        ns = type("_NS", (), {})
        out, ch = ns(), ns()
        ch.message, ch.finish_reason = ns(), choice.get("finish_reason") or "stop"
        ch.message.content = msg.get("content") or ""
        out.choices, out.usage = [ch], ns()
        out.usage.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        return out


class _StdlibOpenAI:
    """Drop-in stand-in for openai.OpenAI covering only what _query uses.

    2026-10-02: our SWE-smith rollout images are the plain ``:latest`` ones and carry
    no ``openai`` package (the official job template uses ``:openai`` images built by
    pull_images.py; building 132 of them to import one class is not worth it, and a
    runtime ``pip install`` inside a network-restricted pod is worse). This same shim
    already drove the 474-instance smith eval (swe_smith_smoke/smith_rollout.py), and
    it raises _StdlibHttpError with .status_code/.message so the context-overflow and
    gateway-paused detectors above keep working unchanged.
    """

    def __init__(self, base_url: str, api_key: str = "dummy", max_retries: int = 6) -> None:
        del max_retries  # urllib has no retry layer; the agent loop burns a turn instead
        self.chat = type("_Chat", (), {"completions": _StdlibCompletions(base_url, api_key)})()


def _query(client: Any, messages: list[dict[str, Any]], max_tokens: int) -> tuple[str, str, int]:
    """Return ``(content, finish_reason, prompt_tokens)`` for one model call.

    ``prompt_tokens`` is the server-reported ``usage.prompt_tokens`` (0 if the
    gateway omits usage), used for the prompt-length penalty — no local tokenizer.
    Raises ``_ContextOverflow`` when the prompt exceeds the model's context.
    Any other failure yields ``("", "error", 0)`` so the caller can burn the turn.
    """
    try:
        completion = client.chat.completions.create(
            model="auto",
            messages=messages,
            max_tokens=max_tokens,
            # 2026-10-02: was hard-coded 1.0, which also overrides the server's
            # --override-generation-config. 1.0 stays the default (= official training
            # recipe), but probes/val need 0.6 to be comparable with the val-474 baselines.
            temperature=float(os.environ.get("SMITH_TEMPERATURE", "1.0")),
            # skip_special_tokens=False: MiniCPM5's tool-call tags <function>,
            # <param>, </param>, </function> ARE special tokens (ids 18/20/21/19 in
            # tokenizer_config.json), so vLLM's default skip_special_tokens=True
            # deletes them from the detokenized text and hands back
            #   name="bash"> name="command">grep -rn "x" /testbed
            # for a reply the model emitted perfectly well. The harness then scored
            # it as a format error AND appended the mangled text to the
            # conversation, so the model read its own garbled turn and copied it.
            # Measured on s3: 1145 native calls over 6 steps, 1015 of them wrongly
            # judged, and both SFT-started runs collapsed on the step this rate
            # jumped. See docs/pitfalls-and-lessons.md §54.
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "skip_special_tokens": False,
                "stop": _STOP_STRINGS,
            },
        )
        choice = completion.choices[0]
        prompt_tokens = getattr(getattr(completion, "usage", None), "prompt_tokens", 0) or 0
        content = strip_turn_delims(choice.message.content or "")
        return content, (choice.finish_reason or "stop"), int(prompt_tokens)
    except Exception as exc:
        if _is_context_overflow(exc):
            raise _ContextOverflow(str(exc)) from exc
        if _is_gateway_paused(exc):
            raise _GatewayPaused(str(exc)) from exc
        log.error("LLM call failed: %s", exc)
        return "", "error", 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
    )

    problem = os.environ.get("AGL_TASK_INPUT", "").strip()
    if not problem:
        raise SystemExit("AGL_TASK_INPUT (problem statement) not set")
    eval_meta = json.loads(os.environ.get("AGL_EVAL_META", "") or "{}")
    if not eval_meta:
        eval_meta = fetch_eval_meta()
    instance_id = eval_meta.get("instance_id", "")
    if not instance_id:
        raise SystemExit("eval meta unavailable: no instance_id (server fetch failed); aborting before rollout")

    base_url = os.environ["AGL_OPENAI_BASE_URL"]
    api_key = os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
    # Train vs val is the /mode/train/ marker in the base URL (fail-safe to val); the
    # reward penalties below key off the same flag. Decided here, before the loop, so
    # validation can sample at a different temperature than training: our val-474
    # baselines (base 123/474) were all measured at 0.6, and an in-training val at the
    # training temperature would not be comparable to them.
    is_train = "/mode/train/" in base_url
    if not is_train and os.environ.get("SMITH_VAL_TEMPERATURE"):
        os.environ["SMITH_TEMPERATURE"] = os.environ["SMITH_VAL_TEMPERATURE"]
    max_turns = int(os.environ.get("SMITH_MAX_TURNS", "40"))
    cmd_timeout = int(os.environ.get("SMITH_CMD_TIMEOUT", "120"))
    eval_timeout = int(os.environ.get("SMITH_EVAL_TIMEOUT", "600"))
    obs_cap = int(os.environ.get("SMITH_OBS_CHAR_CAP", "10000"))
    max_tokens = int(os.environ.get("AGL_MAX_TOKENS", "16384"))
    max_format_errors = int(os.environ.get("SMITH_MAX_FORMAT_ERRORS", "3"))
    gateway_wait_s = float(os.environ.get("SMITH_GATEWAY_WAIT_S", "600"))
    f2p_only = os.environ.get("SMITH_F2P_ONLY", "1").lower() not in ("0", "false", "no")

    try:
        from openai import OpenAI  # type: ignore[assignment]
    except ModuleNotFoundError:
        OpenAI = _StdlibOpenAI  # type: ignore[assignment,misc]
        log.info("openai package absent; using stdlib urllib client")

    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=6)

    log.info("SmithAgent start: instance=%s max_turns=%d", instance_id, max_turns)
    checkout_bug_commit(instance_id)
    relocate_git()
    submitted, n_turns, max_prompt_tokens = run_agent_loop(
        client,
        problem,
        max_turns=max_turns,
        cmd_timeout=cmd_timeout,
        obs_cap=obs_cap,
        max_tokens=max_tokens,
        max_format_errors=max_format_errors,
        gateway_wait_s=gateway_wait_s,
    )

    patch = capture_patch()
    binary, resolved, reason, timed_out, f2p_ratio = evaluate(eval_meta, eval_timeout, f2p_only=f2p_only)

    # Long-turn penalty (plan A): penalize only SOLVED *training* rollouts for
    # burning turns; validation reward stays unshaped (drives ckpt selection).
    # Train mode is the /mode/train/ marker in the base URL (fail-safe to val).
    # SMITH_HMPO=1 skips this global t0 ramp AND the prompt-length penalty so
    # the trainer sees raw 0/1 + n_turns; group-median HMPO is applied in
    # agl_rollout_manager.apply_hmpo_to_completed. Default off = eval unchanged.
    is_train = "/mode/train/" in base_url

    # Dense TRAINING reward: the F2P pass ratio instead of solved-or-not. With a
    # binary reward ~40% of GRPO groups had every rollout at the same value (zero
    # advantage, no gradient at all), and the one scalar was spread over ~5400
    # response tokens of mostly navigation that looks the same in a success and a
    # failure, so those tokens cancelled across the batch. Measured offline on
    # s1's 5735 rollouts: zero-variance groups 40.4% -> 30.3% (x1.17 more groups
    # producing gradient), 34.9% of the all-fail groups regain variance, and a
    # near-miss now scores strictly above doing nothing -- the direct counter to
    # s2's collapse-to-empty. VALIDATION stays binary so every previously
    # reported val number remains comparable.
    f2p_partial = os.environ.get("SMITH_F2P_PARTIAL", "1") == "1"
    reward = f2p_ratio if (is_train and f2p_partial) else binary
    raw_reward = reward
    use_hmpo = os.environ.get("SMITH_HMPO", "0").lower() in ("1", "true", "yes")
    len_pen_t0 = int(os.environ.get("SMITH_LEN_PEN_T0", "80"))
    len_pen_lambda = float(os.environ.get("SMITH_LEN_PEN_LAMBDA", "0.1"))
    prompt_pen_soft = int(os.environ.get("SMITH_PROMPT_PEN_SOFT_START", "50000"))
    prompt_pen_hard = int(os.environ.get("SMITH_PROMPT_PEN_HARD_CAP", "64000"))
    prompt_pen_max = float(os.environ.get("SMITH_PROMPT_PEN_MAX", "0.1"))
    if is_train and max_prompt_tokens >= prompt_pen_hard:
        log.warning("max_prompt_tokens=%d >= hard_cap=%d (context near budget)", max_prompt_tokens, prompt_pen_hard)
    if not use_hmpo:
        reward = length_penalized_reward(reward, n_turns, max_turns, t0=len_pen_t0, lam=len_pen_lambda, is_train=is_train)
        reward = prompt_length_penalty(
            reward,
            max_prompt_tokens,
            soft_start=prompt_pen_soft,
            hard_cap=prompt_pen_hard,
            max_pen=prompt_pen_max,
            is_train=is_train,
            solved=resolved,
        )

    log.info(
        "done: mode=%s submitted=%s patch=%dB reward=%.3f raw_reward=%.3f binary=%.1f f2p_ratio=%.4f "
        "n_turns=%d max_prompt_tokens=%d reason=%s",
        "train" if is_train else "val",
        submitted,
        len(patch),
        reward,
        raw_reward,
        binary,
        f2p_ratio,
        n_turns,
        max_prompt_tokens,
        reason,
    )

    event_base = {"instance_id": instance_id, "repo": eval_meta.get("repo", "")}
    post_event("agent_output", {**event_base, "patch": patch, "patch_size": len(patch), "submitted": submitted})
    post_event(
        "reward",
        {
            **event_base,
            "value": reward,
            "raw_value": raw_reward,
            "resolved": resolved,
            "reason": reason,
            "n_turns": n_turns,
            "max_prompt_tokens": max_prompt_tokens,
            "eval_timeout": timed_out,
            "source": "agent",
        },
        retry=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
