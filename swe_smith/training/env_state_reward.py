# Copyright (c) Microsoft. All rights reserved.

"""Process rewards from environment state changes, not gold-diff overlap.

A step is useful when more FAIL_TO_PASS tests pass than before, or when a
reproduce command flips from fail to pass. Token-level 'is this line correct'
is intentionally out of scope.
"""

from __future__ import annotations


def f2p_delta_reward(prev_pass: int, now_pass: int, n_f2p: int) -> float:
    """Dense signal in [-1, 1] from a change in passing F2P count."""
    if n_f2p <= 0:
        return 0.0
    return (int(now_pass) - int(prev_pass)) / float(n_f2p)


def reproduce_flip_reward(*, prev_ok: bool, now_ok: bool) -> float:
    """+1 when a reproduce script starts passing, -1 when it regresses."""
    if now_ok and not prev_ok:
        return 1.0
    if prev_ok and not now_ok:
        return -1.0
    return 0.0
