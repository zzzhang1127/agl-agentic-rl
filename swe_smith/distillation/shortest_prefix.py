#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Find the shortest successful action prefix of a resolved smith trajectory.

A solved rollout may wander for many turns after the fix is already in the
tree. Truncating by a global turn cap (short15) throws those traces away.
This module replays prefixes and keeps the smallest k whose pytest still
resolves.

Binary search is **not** used: later actions can break a working tree, so
``eval(k)`` is not monotonic. The search is a linear scan over edit-like
candidate turns, after a sanity eval of the full trajectory.

Environment drift (full replay no longer resolves) drops the trajectory
instead of falling back to the original long trace.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import sys

_AGENTS = Path(__file__).resolve().parents[1] / "agents"
if str(_AGENTS) not in sys.path:
    sys.path.insert(0, str(_AGENTS))
from smith_agent import (  # noqa: E402
    SUBMIT_MARKER,
    FormatError,
    _forbidden_action,
    is_submission,
    parse_action,
)

EvalPrefix = Callable[[int], bool]

_EDIT_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?)\s*(?:"
    r"sed\b|perl\s+-i|ruby\s+-i|python[0-9.]*\b[^\n]*\b(?:open|write|Path)\b|"
    r"tee\b|patch\b|ed\b|ex\b|printf\b|cat\s+(?:>>|>)|"
    r">>|install\b"
    r")",
    re.I,
)


def actions_from_smith_log(text: str) -> list[str]:
    """Parse executed bash commands from a smith.log (status.json may omit messages)."""
    actions: list[str] = []
    for line in text.splitlines():
        if "turn=" not in line or " cmd=" not in line or " rc=" not in line:
            continue
        blob = line.split(" cmd=", 1)[1].rsplit(" rc=", 1)[0]
        try:
            action = ast.literal_eval(blob)
        except (SyntaxError, ValueError):
            action = blob.strip("'\"")
        if isinstance(action, str) and action:
            actions.append(action)
    return actions


def assistant_actions(messages: Sequence[dict[str, Any]]) -> list[str]:
    """Return parsed bash actions, skipping format-error assistant turns."""
    actions: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        if not isinstance(content, str):
            continue
        try:
            action = parse_action(content)
        except FormatError:
            continue
        actions.append(action)
    return actions


def is_edit_action(action: str) -> bool:
    if is_submission(action) or SUBMIT_MARKER in action:
        return True
    return bool(_EDIT_RE.search(action))


def candidate_ks(actions: Sequence[str]) -> list[int]:
    """1-based prefix lengths worth evaluating, plus the full trajectory."""
    if not actions:
        return []
    found = [index + 1 for index, action in enumerate(actions) if is_edit_action(action)]
    full = len(actions)
    if full not in found:
        found.append(full)
    return sorted(set(found))


def filter_reason(actions: Sequence[str], *, submitted: bool, resolved: bool) -> str | None:
    """Qwen3-Coder-style drop reasons. None means the trace may be shortened."""
    if not resolved:
        return "not_resolved"
    if not submitted:
        return "missing_termination"
    if not actions:
        return "no_actions"
    if any(_forbidden_action(action) for action in actions):
        return "forbidden_action"
    if not any(SUBMIT_MARKER in action or is_submission(action) for action in actions):
        if not submitted:
            return "missing_termination"
    return None


def shortest_success_k(candidate_indices: Sequence[int], eval_prefix: EvalPrefix, *, full_n: int) -> int | None:
    """Return the smallest candidate k that still resolves.

    Evaluates ``full_n`` first. If that fails, drop the trajectory (env drift).
    """
    if full_n <= 0:
        return None
    if not eval_prefix(full_n):
        return None
    ordered = [k for k in candidate_indices if 1 <= k <= full_n]
    if full_n not in ordered:
        ordered.append(full_n)
    for k in sorted(set(ordered)):
        if k == full_n:
            return k
        if eval_prefix(k):
            return k
    return full_n


def truncate_messages(messages: Sequence[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Keep conversation turns through the k-th parsed assistant action."""
    kept: list[dict[str, Any]] = []
    seen = 0
    for index, message in enumerate(messages):
        kept.append(dict(message))
        if message.get("role") != "assistant":
            continue
        try:
            parse_action(message.get("content") or "")
        except FormatError:
            continue
        seen += 1
        if seen == k:
            # Drop trailing user observations after the last kept action.
            return kept
    return kept


def load_status(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_status(
    status_path: Path,
    result: dict[str, Any],
    *,
    eval_prefix: EvalPrefix,
) -> dict[str, Any] | None:
    status = load_status(status_path)
    messages = status.get("messages") or []
    if not isinstance(messages, list):
        return None
    actions = assistant_actions(messages)
    submitted = bool(status.get("submitted") or result.get("submitted"))
    resolved = bool(result.get("resolved") or (result.get("eval") or {}).get("resolved"))
    reason = filter_reason(actions, submitted=submitted, resolved=resolved)
    if reason:
        return {"instance_id": result.get("instance_id") or status.get("instance_id"), "drop": reason}
    k = shortest_success_k(candidate_ks(actions), eval_prefix, full_n=len(actions))
    if k is None:
        return {"instance_id": result.get("instance_id") or status.get("instance_id"), "drop": "replay_drift"}
    truncated = truncate_messages(messages, k)
    return {
        "instance_id": result.get("instance_id") or status.get("instance_id"),
        "n_turns_original": len(actions),
        "n_turns": k,
        "submitted": True,
        "resolved": True,
        "messages": truncated,
    }


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, help="One status.json to inspect (no docker replay)")
    parser.add_argument("--print-actions", action="store_true")
    args = parser.parse_args(argv)
    if args.status:
        status = load_status(args.status)
        actions = assistant_actions(status.get("messages") or [])
        print(json.dumps({"n": len(actions), "candidates": candidate_ks(actions), "actions": actions if args.print_actions else [a[:80] for a in actions]}, ensure_ascii=False, indent=2))
        return 0
    parser.error("pass --status PATH (docker replay driver lives in shortest_prefix_replay.py)")
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
