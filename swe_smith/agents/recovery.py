# Copyright (c) Microsoft. All rights reserved.

"""Failure-conditioned second attempt (FC-SWE) for smith rollouts.

After attempt 1 fails, the worktree is reset and the failed patch plus verifier
text become the next user message. Each attempt keeps its own reward
(trajectory-local): a later success must not credit the failed first try.

Default-off via ``SMITH_RECOVERY_ATTEMPTS``. MiniCPM val eval leaves it unset.
"""

from __future__ import annotations

from typing import Any

RECOVERY_PROMPT = (
    "The previous attempt did not resolve the issue. The repository has been "
    "reset to the original buggy state.\n\n"
    "Failed patch:\n```\n{patch}\n```\n\n"
    "Verifier feedback:\n{reason}\n\n"
    "Start a new attempt. Use exactly one ```bash block per turn. "
    "When done, echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.\n"
)


def recovery_user_message(patch: str, reason: str, *, patch_cap: int = 8000) -> str:
    clipped = patch if len(patch) <= patch_cap else patch[:patch_cap] + "\n...[truncated]...\n"
    return RECOVERY_PROMPT.format(patch=clipped or "(empty diff)", reason=reason or "(no verifier output)")


def trajectory_local_rewards(attempt_resolved: list[bool]) -> list[float]:
    """One 0/1 reward per attempt; later success does not rewrite earlier zeros."""
    return [1.0 if flag else 0.0 for flag in attempt_resolved]


def should_recover(*, resolved: bool, attempts_allowed: int, attempt_index: int) -> bool:
    """attempt_index is 0-based. Recover only after a failure with budget left."""
    if resolved:
        return False
    return attempt_index + 1 < attempts_allowed
