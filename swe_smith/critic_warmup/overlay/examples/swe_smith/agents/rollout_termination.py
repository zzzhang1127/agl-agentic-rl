# Copyright (c) Microsoft. All rights reserved.

"""Conservative terminal evidence for fixed-horizon, non-bootstrapped GAE."""
from __future__ import annotations

import json
from pathlib import Path


def validate_opencode_terminal(path: Path, returncode: int) -> dict:
    """Require a clean single-session log ending in an explicit stop event.

    This rejects truncation instead of pretending the missing bootstrap is zero.
    It does not prove token-prefix or behavior-logprob consistency.
    """
    result = {"trajectory_terminal": False, "termination_reason": "unknown"}
    if returncode != 0:
        result["termination_reason"] = "runner_timeout" if returncode == 124 else "runner_error"
        return result
    sessions = set()
    last_type, last_reason = None, None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise ValueError("invalid event")
            session = event.get("sessionID")
            if not isinstance(session, str) or not session:
                raise ValueError("missing session")
            sessions.add(session)
            part = event.get("part", {})
            if not isinstance(part, dict):
                raise ValueError("invalid part")
            if part.get("sessionID", session) != session:
                raise ValueError("session mismatch")
            if event["type"] == "error":
                raise ValueError("runner error event")
            if event["type"] == "step_finish" and part.get("reason") not in {"stop", "tool-calls"}:
                raise ValueError("truncated or unknown finish")
            last_type, last_reason = event["type"], part.get("reason")
    except (OSError, UnicodeError, ValueError):
        result["termination_reason"] = "incomplete_or_invalid_log"
        return result
    if len(sessions) == 1 and last_type == "step_finish" and last_reason == "stop":
        return {"trajectory_terminal": True, "termination_reason": "stop"}
    result["termination_reason"] = "missing_final_stop_or_mixed_sessions"
    return result
