"""Dependency-free checks for the optional critic/PPO reward transport guard."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[3] / "agentlightning/verl/reward_guard.py"
spec = importlib.util.spec_from_file_location("reward_guard_tests", PATH)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def rollout(reward=0.0, *, error=None, events=None):
    return SimpleNamespace(rollout_id="test", final_reward=reward, error_message=error, triplet_events=events or [])


@pytest.mark.parametrize("reward", [None, float("nan"), float("inf"), -float("inf"), True, "0"])
def test_unavailable_reward_rejected(reward):
    with pytest.raises(ValueError, match="missing or non-finite"):
        guard.require_explicit_training_rewards([rollout(reward)])


@pytest.mark.parametrize("reward", [0.0, 1.0, -0.2, 0.5])
def test_explicit_finite_reward_accepted(reward):
    guard.require_explicit_training_rewards([rollout(reward)])


def test_runner_failure_blocks_even_with_scalar_reward():
    with pytest.raises(ValueError, match="runner error"):
        guard.require_explicit_training_rewards([rollout(error="container unavailable")])


@pytest.mark.parametrize("eligible", [False, None, "true", 1])
def test_explicit_evaluator_rejection_blocks(eligible):
    events = [{"event_type": "reward", "data": {"value": 0.0, "reward_eligible": eligible}}]
    with pytest.raises(ValueError, match="evaluator rejected"):
        guard.require_explicit_training_rewards([rollout(events=events)])


def test_later_event_cannot_hide_rejected_reward():
    events = [{"event_type": "reward", "data": {"reward_eligible": False}},
              {"event_type": "reward", "data": {"reward_eligible": True}}]
    with pytest.raises(ValueError):
        guard.require_explicit_training_rewards([rollout(events=events)])


@pytest.mark.parametrize("data", [
    {"value": 0.0},
    {"value": 0.0, "reward_eligible": True},
    {"value": 0.0, "reward_eligible": True, "evaluation_state": "timeout"},
])
def test_gae_requires_complete_approval(data):
    with pytest.raises(ValueError, match="complete evaluator approval"):
        guard.require_explicit_training_rewards(
            [rollout(events=[{"event_type": "reward", "data": data}])], require_evaluator_approval=True)


def test_gae_rejects_scalar_without_event():
    with pytest.raises(ValueError, match="complete evaluator approval"):
        guard.require_explicit_training_rewards([rollout()], require_evaluator_approval=True)


@pytest.mark.parametrize("value", [None, True, "0", float("nan"), float("inf"), 1.0])
def test_gae_approval_must_match_final_reward(value):
    data = {"value": value, "reward_eligible": True, "evaluation_state": "complete"}
    with pytest.raises(ValueError, match="disagrees"):
        guard.require_explicit_training_rewards(
            [rollout(events=[{"event_type": "reward", "data": data}])], require_evaluator_approval=True)


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_gae_approved_binary_reward_accepted(reward):
    data = {"value": reward, "reward_eligible": True, "evaluation_state": "complete"}
    guard.require_explicit_training_rewards(
        [rollout(reward, events=[{"event_type": "reward", "data": data}])], require_evaluator_approval=True)


def test_malformed_event_rejected():
    with pytest.raises(ValueError, match="malformed"):
        guard.require_explicit_training_rewards([rollout(events=[{"event_type": "reward", "data": None}])])


def test_split_explicit_training_rewards_keeps_clean_rollouts_and_reports_rejections():
    good = rollout(1.0, events=[{"event_type": "reward", "data": {
        "value": 1.0, "reward_eligible": True, "evaluation_state": "complete",
    }}])
    bad = rollout(0.0, events=[{"event_type": "reward", "data": {
        "value": 0.0, "reward_eligible": False, "evaluation_state": "unavailable",
    }}])

    accepted, rejected = guard.split_explicit_training_rewards(
        [good, bad],
        require_evaluator_approval=True,
    )

    assert accepted == [good]
    assert len(rejected) == 1
    assert rejected[0][0] == "test"
    assert "evaluator rejected" in rejected[0][1]


def test_token_logprob_length_mismatch_rejected():
    triplet = SimpleNamespace(response={"token_ids": [1, 2], "log_probs": [-0.1]})
    bad = SimpleNamespace(rollout_id="bad", triplets=[triplet])
    with pytest.raises(ValueError, match="length mismatch"):
        guard.require_token_logprob_alignment(bad)


def test_split_token_logprob_aligned_rollouts():
    good = SimpleNamespace(
        rollout_id="good",
        triplets=[SimpleNamespace(response={"token_ids": [1, 2], "log_probs": [-0.1, -0.2]})],
    )
    bad = SimpleNamespace(
        rollout_id="bad",
        triplets=[SimpleNamespace(response={"token_ids": [1], "log_probs": [-0.1, -0.2]})],
    )
    accepted, rejected = guard.split_token_logprob_aligned_rollouts([good, bad])
    assert accepted == [good]
    assert rejected[0][0] == "bad"
