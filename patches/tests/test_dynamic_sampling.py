# Copyright (c) Microsoft. All rights reserved.

"""Dynamic sampling (DAPO): groups whose rollouts all scored the same carry no
advantage, so the trainer drops them and rolls out more batches instead of
burning old_log_prob/ref/actor compute on zero gradients."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("verl")

from verl import DataProto  # noqa: E402

from agentlightning.verl.trainer import (  # noqa: E402
    AgentLightningRayPPOTrainer,
    _merge_round_metrics,
    _split_zero_adv_groups,
)


def _batch(scores: list[float], uids: list[str]) -> DataProto:
    n = len(scores)
    return DataProto.from_dict(
        tensors={
            "token_level_scores": torch.tensor([[0.0, s] for s in scores]),
            "responses": torch.zeros(n, 2, dtype=torch.long),
            "response_mask": torch.ones(n, 2, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        },
        non_tensors={
            "uid": np.array(uids, dtype=object),
            "data_id_list": np.array(uids, dtype=object),
            "rollout_id_list": np.array([f"r{i}" for i in range(n)], dtype=object),
        },
    )


def test_split_keeps_only_groups_with_reward_variance() -> None:
    batch = _batch([1.0, 0.0, 1.0, 1.0, 0.0, 0.0], ["a", "a", "b", "b", "c", "c"])
    keep, n_groups, n_zero = _split_zero_adv_groups(batch)
    assert keep == [0, 1]
    assert (n_groups, n_zero) == (3, 2)


def test_split_all_groups_valid_keeps_everything() -> None:
    batch = _batch([1.0, 0.0, 0.0, -0.2], ["a", "a", "b", "b"])
    keep, n_groups, n_zero = _split_zero_adv_groups(batch)
    assert keep == [0, 1, 2, 3]
    assert (n_groups, n_zero) == (2, 0)


def test_split_empty_batch() -> None:
    batch = _batch([], [])
    assert _split_zero_adv_groups(batch) == ([], 0, 0)


def test_merge_round_metrics_sums_counts_and_keeps_latest_rest() -> None:
    acc = _merge_round_metrics({}, {"training/n_rollouts": 32, "training/reward": 0.1, "timing/x": 1.0})
    acc = _merge_round_metrics(acc, {"training/n_rollouts": 32, "training/reward": 0.3, "timing/x": 2.0})
    assert acc == {"training/n_rollouts": 64, "training/reward": 0.3, "timing/x": 2.0}


class _FakeTrainer(AgentLightningRayPPOTrainer):
    """Only the pieces _collect_train_batch touches; no Ray, no workers."""

    def __init__(self, rounds: list[DataProto], ds_cfg: dict[str, Any] | None, train_batch_size: int = 2):
        agl: dict[str, Any] = {"async_rollout": {"enabled": False}}
        if ds_cfg is not None:
            agl["dynamic_sampling"] = ds_cfg
        self.config = OmegaConf.create(
            {
                "agentlightning": agl,
                "data": {"train_batch_size": train_batch_size},
            }
        )
        self.is_async = False
        self.global_steps = 1
        self._rounds = list(rounds)
        self.n_rollout_calls = 0

    def _rollout_one_round(self, curr_step_profile: bool) -> tuple[DataProto, dict[str, Any]]:  # type: ignore[override]
        self.n_rollout_calls += 1
        if not self._rounds:
            raise AssertionError("more rollout rounds requested than the test provided")
        out = self._rounds.pop(0)
        return out, {"training/n_rollouts": len(out), "training/reward": float(self.n_rollout_calls)}


def _uids(batch: DataProto) -> list[str]:
    return list(batch.non_tensor_batch["uid"])


def test_disabled_is_single_round_and_keeps_zero_adv_groups() -> None:
    trainer = _FakeTrainer([_batch([1.0, 1.0, 1.0, 0.0], ["a", "a", "b", "b"])], ds_cfg=None)
    metrics: dict[str, Any] = {}
    out = trainer._collect_train_batch(metrics, curr_step_profile=False)
    assert out is not None
    assert len(out) == 4
    assert trainer.n_rollout_calls == 1
    assert metrics["training/dynamic_sampling/enabled"] == 0
    assert metrics["training/dynamic_sampling/gen_rounds"] == 1
    assert metrics["training/n_rollouts"] == 4


def test_enabled_resamples_until_enough_valid_groups() -> None:
    rounds = [
        _batch([1.0, 1.0, 1.0, 0.0], ["a", "a", "b", "b"]),  # only b valid -> 1/2
        _batch([0.0, 0.0, 0.0, 0.0], ["c", "c", "d", "d"]),  # nothing valid -> still 1/2
        _batch([0.0, 1.0, 1.0, 1.0], ["e", "e", "f", "f"]),  # e valid -> 2/2, stop
        _batch([0.0, 1.0, 0.0, 1.0], ["g", "g", "h", "h"]),  # must not be drawn
    ]
    trainer = _FakeTrainer(rounds, ds_cfg={"enabled": True, "max_gen_batches": 5, "min_valid_groups": None})
    metrics: dict[str, Any] = {}
    out = trainer._collect_train_batch(metrics, curr_step_profile=False)
    assert out is not None
    assert trainer.n_rollout_calls == 3
    assert _uids(out) == ["b", "b", "e", "e"]
    assert out.batch["token_level_scores"].sum(-1).tolist() == [1.0, 0.0, 0.0, 1.0]
    assert metrics["training/dynamic_sampling/gen_rounds"] == 3
    assert metrics["training/dynamic_sampling/n_rows_seen"] == 12
    assert metrics["training/dynamic_sampling/n_rows_discarded_zero_adv"] == 8
    assert metrics["training/dynamic_sampling/n_groups_seen"] == 6
    assert metrics["training/dynamic_sampling/n_groups_zero_adv"] == 4
    assert metrics["training/dynamic_sampling/n_groups_valid"] == 2
    # counts add up across rounds, other agent metrics keep the latest round
    assert metrics["training/n_rollouts"] == 12
    assert metrics["training/reward"] == 3.0


def test_enabled_stops_at_max_gen_batches_with_partial_batch() -> None:
    rounds = [
        _batch([1.0, 1.0, 1.0, 0.0], ["a", "a", "b", "b"]),
        _batch([0.0, 0.0, 0.0, 0.0], ["c", "c", "d", "d"]),
        _batch([0.0, 1.0, 1.0, 1.0], ["e", "e", "f", "f"]),
    ]
    trainer = _FakeTrainer(rounds, ds_cfg={"enabled": True, "max_gen_batches": 2, "min_valid_groups": 2})
    metrics: dict[str, Any] = {}
    out = trainer._collect_train_batch(metrics, curr_step_profile=False)
    assert out is not None
    assert trainer.n_rollout_calls == 2
    assert _uids(out) == ["b", "b"]
    assert metrics["training/dynamic_sampling/n_groups_valid"] == 1


def test_enabled_returns_none_when_nothing_survives() -> None:
    rounds = [_batch([0.0, 0.0], ["a", "a"]), _batch([1.0, 1.0], ["b", "b"])]
    trainer = _FakeTrainer(rounds, ds_cfg={"enabled": True, "max_gen_batches": 2, "min_valid_groups": 1})
    metrics: dict[str, Any] = {}
    assert trainer._collect_train_batch(metrics, curr_step_profile=False) is None
    assert trainer.n_rollout_calls == 2
    assert metrics["training/dynamic_sampling/n_rows_discarded_zero_adv"] == 4


def test_enabled_single_round_still_filters() -> None:
    trainer = _FakeTrainer(
        [_batch([1.0, 1.0, 1.0, 0.0], ["a", "a", "b", "b"])],
        ds_cfg={"enabled": True, "max_gen_batches": 1, "min_valid_groups": 2},
    )
    out = trainer._collect_train_batch({}, curr_step_profile=False)
    assert out is not None
    assert _uids(out) == ["b", "b"]


def test_config_validation() -> None:
    trainer = _FakeTrainer([], ds_cfg={"enabled": True, "max_gen_batches": 0})
    with pytest.raises(ValueError, match="max_gen_batches"):
        trainer._dynamic_sampling_config()
    trainer = _FakeTrainer([], ds_cfg={"enabled": True, "max_gen_batches": 2, "min_valid_groups": 0})
    with pytest.raises(ValueError, match="min_valid_groups"):
        trainer._dynamic_sampling_config()
    trainer = _FakeTrainer([], ds_cfg={"enabled": True, "max_gen_batches": 2})
    trainer.is_async = True
    with pytest.raises(ValueError, match="async"):
        trainer._dynamic_sampling_config()
    trainer = _FakeTrainer([], ds_cfg={"enabled": True, "max_gen_batches": 2}, train_batch_size=8)
    assert trainer._dynamic_sampling_config() == (True, 2, 8)
