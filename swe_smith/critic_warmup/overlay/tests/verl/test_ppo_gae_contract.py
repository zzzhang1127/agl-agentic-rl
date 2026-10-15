# Copyright (c) Microsoft. All rights reserved.

"""Pin installed verl GAE semantics before connecting agentic critic training.

These are PPO contract tests, not a claim of reproducing the SAO paper. Known
adapter hazards are demonstrated explicitly so a future integration cannot
mistake a finite loss for correct credit assignment.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("verl")

import torch
from verl.trainer.ppo.core_algos import compute_gae_advantage_return


def gae(rewards, values, mask, *, gamma=1.0, lam=1.0):
    return compute_gae_advantage_return(
        token_level_rewards=torch.tensor(rewards, dtype=torch.float32),
        values=torch.tensor(values, dtype=torch.float32),
        response_mask=torch.tensor(mask, dtype=torch.float32),
        gamma=gamma,
        lam=lam,
    )


def test_observations_do_not_decay_advantage_or_supply_values():
    # Two actions separated by three observation tokens. Deliberately huge
    # observation values must not affect action advantages or returns.
    advantages, returns = gae(
        [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
        [[0.2, 100, -200, 300, 0.4], [0.3, -100, 200, -300, 0.1]],
        [[1, 0, 0, 0, 1], [1, 0, 0, 0, 1]], gamma=0.9, lam=0.8,
    )
    compact_advantages, compact_returns = gae(
        [[0, 1], [0, 0]], [[0.2, 0.4], [0.3, 0.1]], [[1, 1], [1, 1]],
        gamma=0.9, lam=0.8,
    )
    torch.testing.assert_close(returns[:, [0, 4]], compact_returns)
    torch.testing.assert_close(advantages[:, [0, 4]], compact_advantages)
    # delta0=.9*.4-.2=.16; delta1=1-.4=.6;
    # return0=.2+.16+.9*.8*.6=.792 (no observation-length discount).
    assert returns[0, 0].item() == pytest.approx(0.792)


def test_lambda_one_terminal_returns_equal_observed_outcome():
    _, returns = gae(
        [[0, 0, 1], [0, 0, 0]], [[0.2, 0.5, 0.4], [0.8, 0.1, 0.3]],
        [[1, 1, 1], [1, 1, 1]],
    )
    torch.testing.assert_close(returns, torch.tensor([[1., 1., 1.], [0., 0., 0.]]))


def test_reward_on_final_observation_is_lost_by_skip_observation_gae():
    # Current trajectory adapter can emit this when the final response is
    # empty. The test documents a real blocker, not desired reward placement.
    _, wrong_returns = gae([[0, 0, 1]], [[0, 0, 0]], [[1, 1, 0]])
    _, action_returns = gae([[0, 1, 0]], [[0, 0, 0]], [[1, 1, 0]])
    torch.testing.assert_close(wrong_returns[:, :2], torch.zeros(1, 2))
    torch.testing.assert_close(action_returns[:, :2], torch.ones(1, 2))


def test_default_gae_assumes_terminal_not_time_limit_bootstrap():
    # No next-state value is passed to this API. At an artificial truncation,
    # V(next)=.7 would yield a different target; treating it as terminal is not
    # a valid substitute for an explicit environment termination decision.
    _, returns = gae([[0, 0]], [[0.2, 0.4]], [[1, 1]], gamma=0.9, lam=1.0)
    assert returns[0, -1].item() == pytest.approx(0.0, abs=1e-7)
    assert returns[0, -1].item() != pytest.approx(0.9 * 0.7)
