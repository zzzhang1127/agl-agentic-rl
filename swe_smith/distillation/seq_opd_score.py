#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Sequence-level on-policy score from a cross-vocab teacher.

The student samples a smith trajectory. We detokenize to text, let the teacher
tokenizer / vLLM score ``log P_T(y)``, and use that scalar as an advantage or
filter. This is not token-aligned MOPD: MiniCPM and Qwen vocabs do not match.

Never insert gold patches into the teacher context.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from typing import Any


def assistant_text(messages: list[dict[str, Any]]) -> str:
    parts = [str(m.get("content") or "") for m in messages if m.get("role") == "assistant"]
    return "\n".join(parts)


def sum_token_logprobs(logprobs: list[dict[str, Any]] | None) -> float | None:
    """Sum chosen-token logprobs from an OpenAI-style logprobs payload."""
    if not logprobs:
        return None
    total = 0.0
    n = 0
    for row in logprobs:
        token_logprob = row.get("logprob")
        if token_logprob is None and isinstance(row.get("top_logprobs"), list) and row["top_logprobs"]:
            token_logprob = row["top_logprobs"][0].get("logprob")
        if token_logprob is None:
            continue
        total += float(token_logprob)
        n += 1
    if n == 0:
        return None
    return total


def sequence_score_from_choice(choice: dict[str, Any]) -> float | None:
    logprobs = (choice.get("logprobs") or {}).get("content") or (choice.get("logprobs") or {}).get("token_logprobs")
    if isinstance(logprobs, list) and logprobs and isinstance(logprobs[0], dict):
        return sum_token_logprobs(logprobs)
    if isinstance(logprobs, list) and logprobs and isinstance(logprobs[0], (int, float)):
        return float(sum(logprobs))
    return None


def request_teacher_logprobs(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Score the student's own messages under the teacher (echo / max_tokens=1).

    Uses chat completions with logprobs. The teacher must not receive extra
    gold context — ``messages`` is the student trajectory as-is.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "logprobs": True,
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", required=True, help="Student status.json with messages")
    parser.add_argument("--base-url", default="http://127.0.0.1:18010/v1")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    args = parser.parse_args()
    status = json.loads(open(args.status, encoding="utf-8").read())
    messages = status.get("messages") or []
    data = request_teacher_logprobs(base_url=args.base_url, model=args.model, messages=messages)
    choice = (data.get("choices") or [{}])[0]
    score = sequence_score_from_choice(choice)
    print(json.dumps({"instance_id": status.get("instance_id"), "seq_logprob": score, "n_assistant": assistant_text(messages).count("\n")}, ensure_ascii=False))
    return 0 if score is not None and math.isfinite(score) else 2


if __name__ == "__main__":
    raise SystemExit(main())
