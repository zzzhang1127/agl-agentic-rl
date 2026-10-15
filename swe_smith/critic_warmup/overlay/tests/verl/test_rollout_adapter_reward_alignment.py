# Copyright (c) Microsoft. All rights reserved.

"""Regression tests for batch-width alignment in RolloutAdapter.

These tests pin the contract introduced by ``_repad_to_batch_max``: the batch is
padded only to the longest real row (``response_width``), not the configured
``max_response_length``. Every score / mask / log-prob tensor whose meaning is
"per response token" must share that dynamic width, most importantly
``token_level_scores`` (previously sliced with the configured ceiling and thus
out of sync with ``responses``).

They also document where the scalar reward is placed: on the last real token of
the merged response, which may be an observation token (mask 0) rather than an
eligible action token. See ``test_trajectory_reward_lands_on_final_observation``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

import torch

from agentlightning.verl.agl_rollout_manager import CompletedRollout, Triplet
from agentlightning.verl.rollout_adapter import RolloutAdapter


class FakeTokenizer:
    all_special_ids: list[int] = []

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)


_SENTINEL = object()


def _triplet(
    prompt_ids: list[int],
    response_ids: list[int],
    log_probs: object = _SENTINEL,
) -> Triplet:
    if log_probs is _SENTINEL:
        log_probs = [-0.1] * len(response_ids)
    return Triplet(
        prompt={"token_ids": prompt_ids},
        response={"token_ids": response_ids, "log_probs": log_probs},
    )


def _rollout(
    rollout_id: str,
    triplets: list[Triplet],
    *,
    final_reward: float = 1.0,
) -> CompletedRollout:
    return CompletedRollout(
        rollout_id=rollout_id,
        data_id=f"data-{rollout_id}",
        step=1,
        sample_idx_in_step=0,
        enqueue_time=0.0,
        final_reward=final_reward,
        triplets=triplets,
        triplet_events=[{"event_type": "reward", "data": {
            "value": final_reward, "reward_eligible": True, "evaluation_state": "complete",
        }}],
    )


def _adapter(
    *,
    max_prompt_length: int,
    max_response_length: int,
    level: str,
) -> RolloutAdapter:
    return RolloutAdapter(
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        device=torch.device("cpu"),
        pad_token_id=0,
        trace_aggregator_level=level,
        tokenizer=FakeTokenizer(),
    )


def test_strict_reward_guard_runs_before_batch_construction() -> None:
    adapter = _adapter(max_prompt_length=8, max_response_length=8, level="transition")
    adapter.require_explicit_reward = True
    rollout = _rollout("missing", [_triplet([1], [2])]).model_copy(update={"final_reward": None})
    with pytest.raises(ValueError, match="missing or non-finite training reward"):
        adapter.get_train_data_batch([rollout], global_steps=1)


def test_gae_rejects_legacy_scalar_even_without_explicit_guard_flag() -> None:
    adapter = _adapter(max_prompt_length=8, max_response_length=8, level="trajectory")
    adapter.gae_trajectory = True
    assert adapter.require_explicit_reward is False
    rollout = _rollout("legacy", [_triplet([1], [2])]).model_copy(update={"triplet_events": []})
    with pytest.raises(ValueError, match="complete evaluator approval"):
        adapter.get_train_data_batch([rollout], global_steps=1)


def test_token_level_scores_match_dynamic_response_width_transition() -> None:
    adapter = _adapter(max_prompt_length=8, max_response_length=8, level="transition")
    rollouts = [
        _rollout("r1", [_triplet([1], [2, 3])], final_reward=1.0),
        _rollout("r2", [_triplet([4], [5, 6, 7, 8])], final_reward=2.0),
    ]

    batch, _ = adapter.get_train_data_batch(rollouts, global_steps=1)

    responses = batch.batch["responses"]
    scores = batch.batch["token_level_scores"]
    assert responses.shape == (2, 4)  # dynamic width = max(2, 4)
    assert scores.shape == (2, 4)  # aligned to responses, not 8 (max_response_length)
    # Reward lands on the last real response token of each row.
    assert scores[0, 1].item() == 1.0
    assert scores[0, 0].item() == 0.0
    assert scores[1, 3].item() == 2.0
    assert scores[1, 0].item() == 0.0


def test_token_level_scores_match_when_response_hits_max_length() -> None:
    adapter = _adapter(max_prompt_length=8, max_response_length=4, level="transition")
    rollouts = [_rollout("r1", [_triplet([1, 2], [3, 4, 5, 6])], final_reward=1.5)]

    batch, _ = adapter.get_train_data_batch(rollouts, global_steps=1)

    responses = batch.batch["responses"]
    scores = batch.batch["token_level_scores"]
    assert responses.shape == (1, 4)
    assert scores.shape == (1, 4)  # full width coincides with max_response_length
    assert scores[0, 3].item() == 1.5


def test_left_padded_long_prompt_short_response_no_prompt_leakage() -> None:
    adapter = _adapter(max_prompt_length=16, max_response_length=16, level="transition")
    rollouts = [_rollout("r1", [_triplet(list(range(10)), [100, 101])], final_reward=3.0)]

    batch, _ = adapter.get_train_data_batch(rollouts, global_steps=1)

    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    scores = batch.batch["token_level_scores"]
    # Prompt is left-padded to the longest real prompt (no padding here).
    assert prompts.shape == (1, 10)
    assert prompts[0].tolist() == list(range(10))
    # Response width is dynamic (2), not prompt+response width (12) and not 16.
    assert responses.shape == (1, 2)
    assert scores.shape == (1, 2)
    assert scores[0, 1].item() == 3.0
    assert scores[0, 0].item() == 0.0


def test_trajectory_observation_mask_and_reward_on_final_action() -> None:
    adapter = _adapter(max_prompt_length=16, max_response_length=16, level="trajectory")
    rollout = _rollout(
        "r1",
        [
            _triplet([1], [2]),
            _triplet([1, 2, 3], [4]),  # observation token 3, action token 4
        ],
        final_reward=0.75,
    )

    batch, _ = adapter.get_train_data_batch([rollout], global_steps=1)

    responses = batch.batch["responses"]
    scores = batch.batch["token_level_scores"]
    mask = batch.batch["response_mask"]
    log_probs = batch.batch["rollout_log_probs"]
    assert responses.shape == (1, 3)
    assert scores.shape == (1, 3)
    assert mask.shape == (1, 3)
    assert mask.tolist() == [[1, 0, 1]]
    # Reward sits on the final action token (mask 1), index 2.
    assert scores[0, 2].item() == 0.75
    assert scores[0, 0].item() == 0.0
    assert scores[0, 1].item() == 0.0
    # Log probs share the dynamic response width; observation slot is zero-filled.
    assert log_probs.shape == (1, 3)
    assert log_probs[0].tolist() == pytest.approx([-0.1, 0.0, -0.1])


def test_trajectory_reward_lands_on_final_observation() -> None:
    # Documents current reward placement semantics: when the last merged turn
    # contributes only an observation (empty action), the scalar reward is still
    # placed on the last real token, which is masked out of the policy loss.
    adapter = _adapter(max_prompt_length=16, max_response_length=16, level="trajectory")
    rollout = _rollout(
        "r1",
        [
            _triplet([1], [2]),
            _triplet([1, 2, 9], []),  # observation-only final turn
        ],
        final_reward=0.5,
    )

    batch, _ = adapter.get_train_data_batch([rollout], global_steps=1)

    responses = batch.batch["responses"]
    scores = batch.batch["token_level_scores"]
    mask = batch.batch["response_mask"]
    assert responses.shape == (1, 2)
    assert scores.shape == (1, 2)
    assert mask.tolist() == [[1, 0]]
    # Reward lands on the observation token (mask 0), not an eligible action.
    assert scores[0, 1].item() == 0.5
    assert mask[0, 1].item() == 0
    # This placement is incompatible with skip-observation GAE: the terminal
    # reward disappears from the action's return. Keep an executable blocker
    # until the critic path has an explicit action-aligned reward policy.
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return

    # Two identical rows supply two action elements for unbiased whitening.
    _, returns = compute_gae_advantage_return(
        token_level_rewards=scores.float().repeat(2, 1),
        values=torch.zeros_like(scores, dtype=torch.float32).repeat(2, 1),
        response_mask=mask.float().repeat(2, 1), gamma=1.0, lam=1.0,
    )
    assert returns[0, 0].item() == 0.0


def test_gae_trajectory_places_outcome_on_last_action() -> None:
    adapter = _adapter(max_prompt_length=16, max_response_length=16, level="trajectory")
    adapter.gae_trajectory = True
    rollout = _rollout("r1", [_triplet([1], [2, 3]), _triplet([1, 2, 3, 9], [])], final_reward=0.5)
    batch, _ = adapter.get_train_data_batch([rollout], global_steps=1)
    scores, mask = batch.batch["token_level_scores"], batch.batch["response_mask"]
    assert scores.tolist() == [[0.0, 0.5, 0.0]]
    assert mask.tolist() == [[1, 1, 0]]
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return

    _, returns = compute_gae_advantage_return(
        token_level_rewards=scores.float(), values=torch.zeros_like(scores, dtype=torch.float32),
        response_mask=mask.float(), gamma=1.0, lam=1.0,
    )
    torch.testing.assert_close(returns[:, :2], torch.full((1, 2), 0.5))


@pytest.mark.parametrize("case", ["transition", "truncation"])
def test_gae_trajectory_rejects_unlinked_or_truncated_rows(case) -> None:
    adapter = _adapter(max_prompt_length=8, max_response_length=4,
                       level="transition" if case == "transition" else "trajectory")
    adapter.gae_trajectory = True
    if case == "truncation":
        triplets = [_triplet([1], [2, 3, 4, 5, 6])]
    else:
        triplets = [_triplet([1], [2])]
    with pytest.raises(ValueError, match="GAE"):
        adapter.get_train_data_batch([_rollout("r1", triplets)], global_steps=1)


def test_gae_trajectory_skips_prefix_mismatch_rollouts() -> None:
    adapter = _adapter(max_prompt_length=8, max_response_length=8, level="trajectory")
    adapter.gae_trajectory = True
    bad = _rollout("bad", [_triplet([1], [2]), _triplet([9], [3])])
    good = _rollout("good", [_triplet([4], [5]), _triplet([4, 5, 6], [7])])

    batch, metrics = adapter.get_train_data_batch([bad, good], global_steps=1)

    assert batch.non_tensor_batch["rollout_id_list"].tolist() == ["good"]
    assert metrics["training/n_rollouts_skipped_prefix_mismatch"] == 1
    assert batch.batch["response_mask"].tolist() == [[1, 0, 1]]


def test_gae_trajectory_reconstructs_missing_later_prompts() -> None:
    adapter = _adapter(max_prompt_length=8, max_response_length=8, level="trajectory")
    adapter.gae_trajectory = True
    rollout = _rollout("r1", [_triplet([1], [2]), _triplet([], [3])])

    batch, metrics = adapter.get_train_data_batch([rollout], global_steps=1)

    assert batch.non_tensor_batch["rollout_id_list"].tolist() == ["r1"]
    assert metrics["training/n_rollouts_skipped_prefix_mismatch"] == 0
    assert batch.batch["response_mask"].tolist() == [[1, 1]]
    assert batch.batch["responses"].tolist() == [[2, 3]]


def test_gae_trajectory_raises_when_all_prefix_mismatch_rollouts_skip() -> None:
    adapter = _adapter(max_prompt_length=8, max_response_length=8, level="trajectory")
    adapter.gae_trajectory = True
    bad = _rollout("bad", [_triplet([1], [2]), _triplet([9], [3])])

    with pytest.raises(RuntimeError, match="zero training rows"):
        adapter.get_train_data_batch([bad], global_steps=1)


def test_gae_can_keep_terminal_segment_on_prefix_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGL_GAE_KEEP_TERMINAL_SEGMENT_ON_PREFIX_MISMATCH", "1")
    adapter = _adapter(max_prompt_length=8, max_response_length=8, level="trajectory")
    adapter.gae_trajectory = True
    rollout = _rollout("r1", [_triplet([1], [2]), _triplet([9], [3])], final_reward=0.5)

    batch, metrics = adapter.get_train_data_batch([rollout], global_steps=1)

    assert batch.non_tensor_batch["rollout_id_list"].tolist() == ["r1"]
    assert metrics["training/n_rollouts_skipped_prefix_mismatch"] == 0
    assert metrics["training/n_trace_merge_mismatch_rows"] == 1
    assert batch.batch["responses"].tolist() == [[3]]
    assert batch.batch["token_level_scores"].tolist() == [[0.5]]


def test_rollout_log_probs_omitted_when_missing_and_aligned_when_present() -> None:
    adapter = _adapter(max_prompt_length=8, max_response_length=8, level="transition")

    no_log = _rollout("r1", [_triplet([1], [2, 3], log_probs=None)], final_reward=1.0)
    batch, _ = adapter.get_train_data_batch([no_log], global_steps=1)
    assert "rollout_log_probs" not in batch.batch

    with_log = _rollout("r2", [_triplet([4], [5, 6, 7, 8])], final_reward=2.0)
    batch, _ = adapter.get_train_data_batch([with_log], global_steps=1)
    assert batch.batch["rollout_log_probs"].shape == (1, 4)


def test_empty_action_rows_skipped_and_all_empty_raises() -> None:
    adapter = _adapter(max_prompt_length=8, max_response_length=8, level="transition")
    empty = _rollout("r1", [_triplet([1], [])], final_reward=1.0)

    with pytest.raises(RuntimeError, match="zero training rows"):
        adapter.get_train_data_batch([empty], global_steps=1)

    valid = _rollout("r2", [_triplet([2], [3, 4])], final_reward=2.0)
    batch, metrics = adapter.get_train_data_batch([empty, valid], global_steps=1)
    assert batch.batch["responses"].shape == (1, 2)
    assert batch.batch["token_level_scores"].shape == (1, 2)
    assert metrics["training/n_skipped_empty_rows"] == 1


def test_truncation_metric_counts_response_tokens_over_length() -> None:
    # n_truncated_sample counts merged rows whose response tokens exceed
    # max_response_length; it is a token-length counter, not a turn counter.
    adapter = _adapter(max_prompt_length=8, max_response_length=4, level="transition")
    rollout = _rollout("r1", [_triplet([1], [2, 3, 4, 5, 6, 7])], final_reward=1.0)

    batch, metrics = adapter.get_train_data_batch([rollout], global_steps=1)

    assert metrics["training/n_truncated_sample"] == 1
    assert batch.batch["responses"].shape == (1, 4)  # truncated to max_response_length
    assert batch.batch["token_level_scores"].shape == (1, 4)
