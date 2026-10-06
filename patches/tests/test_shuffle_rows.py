"""Row shuffling before mini-batch splitting: a permutation of the batch that keeps
every tensor and non-tensor field aligned."""

from __future__ import annotations

import numpy as np
import torch
from verl import DataProto

from agentlightning.verl.trainer import _shuffle_rows


def _batch(n: int) -> DataProto:
    ids = torch.arange(n)
    return DataProto.from_dict(
        tensors={"row_id": ids, "responses": ids.unsqueeze(1).repeat(1, 3)},
        non_tensors={
            "uid": np.array([f"g{i // 4}" for i in range(n)], dtype=object),
            "rollout_id_list": np.array([f"r{i // 2}" for i in range(n)], dtype=object),
        },
    )


def test_shuffle_is_a_permutation_with_aligned_fields() -> None:
    torch.manual_seed(0)
    n = 64
    batch = _shuffle_rows(_batch(n))
    row_id = batch.batch["row_id"].tolist()
    assert sorted(row_id) == list(range(n))
    assert row_id != list(range(n))  # actually moved (seeded, so deterministic)
    # non-tensor fields moved with their rows
    for pos, rid in enumerate(row_id):
        assert batch.non_tensor_batch["uid"][pos] == f"g{rid // 4}"
        assert batch.non_tensor_batch["rollout_id_list"][pos] == f"r{rid // 2}"
        assert batch.batch["responses"][pos].tolist() == [rid, rid, rid]


def test_group_membership_is_order_independent() -> None:
    torch.manual_seed(1)
    batch = _shuffle_rows(_batch(32))
    uids = batch.non_tensor_batch["uid"].tolist()
    assert sorted(set(uids)) == [f"g{i}" for i in range(8)]
    assert all(uids.count(u) == 4 for u in set(uids))


def test_tiny_batches_are_left_alone() -> None:
    b = _batch(1)
    assert _shuffle_rows(b) is b
