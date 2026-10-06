import numpy as np
import pytest
import torch

pytest.importorskip("verl")

from verl import DataProto  # noqa: E402

from agentlightning.verl.trainer import _mini_batch_fit, _pad_with_neutral_duplicates  # noqa: E402


def test_remainder_is_padded_not_dropped() -> None:
    # 66 rows, mini-batch 32: the old floor kept 64 and threw 2 away.
    assert _mini_batch_fit(66, 32, None) == (0, 30)
    assert _mini_batch_fit(64, 32, None) == (0, 0)
    assert _mini_batch_fit(5, 32, None) == (0, 27)
    assert _mini_batch_fit(0, 32, None) == (0, 0)


def test_update_cap_still_drops() -> None:
    assert _mini_batch_fit(227, 32, 7) == (3, 0)
    assert _mini_batch_fit(227, 32, 16) == (0, 29)


def _batch(scores: list[float], uids: list[str]) -> DataProto:
    n = len(scores)
    proto = DataProto.from_single_dict(
        {
            "token_level_scores": torch.tensor([[0.0, s] for s in scores]),
            "responses": torch.arange(n * 2).reshape(n, 2),
            "response_mask": torch.ones(n, 2, dtype=torch.long),
        }
    )
    proto.non_tensor_batch["uid"] = np.array(uids, dtype=object)
    proto.non_tensor_batch["rollout_id_list"] = np.array([f"r{i}" for i in range(n)])
    return proto


def test_pad_rows_are_invisible_to_grpo() -> None:
    batch = _batch([1.0, 0.0, 1.0], ["g1", "g1", "g2"])
    padded = _pad_with_neutral_duplicates(batch, 5)

    assert len(padded) == 8
    # Originals untouched: same scores, same groups.
    assert padded.batch["token_level_scores"][:3].sum(-1).tolist() == [1.0, 0.0, 1.0]
    assert padded.non_tensor_batch["uid"][:3].tolist() == ["g1", "g1", "g2"]
    # Pads: zero score, and each in a group of its own so no source group's
    # mean moves.  GRPO on a singleton with score 0 yields advantage 0.
    assert padded.batch["token_level_scores"][3:].abs().sum().item() == 0.0
    pad_uids = set(padded.non_tensor_batch["uid"][3:].tolist())
    assert len(pad_uids) == 5
    assert pad_uids.isdisjoint({"g1", "g2"})
    # And a rollout of its own: rollout-level advantage groups rows by
    # rollout_id and insists they share a uid, so a pad cannot keep its source's.
    assert padded.non_tensor_batch["rollout_id_list"][:3].tolist() == ["r0", "r1", "r2"]
    pad_rollouts = set(padded.non_tensor_batch["rollout_id_list"][3:].tolist())
    assert len(pad_rollouts) == 5
    assert pad_rollouts.isdisjoint({"r0", "r1", "r2"})
    # Pads are real rows (copied tokens), so the forward pass has something to chew on.
    assert padded.batch["responses"][3:].shape == (5, 2)


def test_padded_batch_passes_rollout_level_advantage_validation() -> None:
    from agentlightning.verl.rollout_level_advantage import compute_rollout_level_advantage

    # r0 spans two rows (a budget-split trajectory): same reward, same group.
    batch = _batch([1.0, 1.0, 0.0], ["g1", "g1", "g2"])
    batch.non_tensor_batch["rollout_id_list"] = np.array(["r0", "r0", "r2"])
    padded = _pad_with_neutral_duplicates(batch, 5)
    padded.batch["token_level_rewards"] = padded.batch["token_level_scores"]

    def fake_compute_advantage(rollout_batch: DataProto, **_: object) -> DataProto:
        # Outcome-style advantage: the row's own score, constant over the response.
        score = rollout_batch.batch["token_level_rewards"].sum(-1, keepdim=True)
        rollout_batch.batch["advantages"] = score * rollout_batch.batch["response_mask"]
        return rollout_batch

    out, metrics = compute_rollout_level_advantage(
        padded, adv_estimator="grpo", gamma=1.0, lam=1.0, num_repeat=4, compute_advantage_fn=fake_compute_advantage
    )
    assert metrics["training/rollout_level_advantage/n_rollouts"] == 2 + 5
    assert out.batch["advantages"][3:].abs().sum().item() == 0.0
    assert out.batch["advantages"][:3].sum(-1).tolist() == [2.0, 2.0, 0.0]
