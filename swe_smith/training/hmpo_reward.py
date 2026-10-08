# Copyright (c) Microsoft. All rights reserved.

"""Hybrid Median-length Policy Optimization (HMPO) group reward.

Length budget is the median of *correct* rollouts in the group. Wrong rollouts
always score 0, so a short-but-wrong reply cannot outrank a correct one.
An all-wrong group stays all zeros (no length hacking).
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median


def hmpo_rewards(
    resolved: Sequence[bool],
    lengths: Sequence[int],
    *,
    floor: float = 0.1,
) -> list[float]:
    """Return one multiplicative reward per group member.

    ``reward = 1_{resolved} * clip(budget / max(length, 1), floor, 1)``.
    ``budget`` is the median length among resolved members. Unresolved members
    are 0. If nobody resolved, every reward is 0.
    """
    if len(resolved) != len(lengths):
        raise ValueError(f"resolved ({len(resolved)}) and lengths ({len(lengths)}) size mismatch")
    if not resolved:
        return []
    correct_lengths = [max(int(length), 1) for flag, length in zip(resolved, lengths, strict=True) if flag]
    if not correct_lengths:
        return [0.0] * len(resolved)
    budget = float(median(correct_lengths))
    out: list[float] = []
    for flag, length in zip(resolved, lengths, strict=True):
        if not flag:
            out.append(0.0)
            continue
        scale = budget / max(int(length), 1)
        out.append(min(1.0, max(floor, scale)))
    return out


def hmpo_reshape_group(rewards: Sequence[float], lengths: Sequence[int], *, floor: float = 0.1) -> list[float]:
    """Treat a positive terminal reward as resolved, then apply HMPO."""
    resolved = [float(reward) > 0.0 for reward in rewards]
    return hmpo_rewards(resolved, lengths, floor=floor)


def hmpo_reshape_by_uid(
    uids: Sequence[str],
    rewards: Sequence[float],
    lengths: Sequence[int],
    *,
    floor: float = 0.1,
) -> list[float]:
    """Reshape rewards inside each uid group independently."""
    if not (len(uids) == len(rewards) == len(lengths)):
        raise ValueError("uids, rewards, and lengths must be the same length")
    groups: dict[str, list[int]] = {}
    for index, uid in enumerate(uids):
        groups.setdefault(str(uid), []).append(index)
    out = [0.0] * len(rewards)
    for indices in groups.values():
        reshaped = hmpo_reshape_group(
            [rewards[i] for i in indices],
            [lengths[i] for i in indices],
            floor=floor,
        )
        for i, value in zip(indices, reshaped, strict=True):
            out[i] = value
    return out
