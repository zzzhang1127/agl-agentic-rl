# Copyright (c) Microsoft. All rights reserved.

"""Two-rank CPU FSDP1 contract; no pretrained weights or production workers."""
import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from transformers import LlamaConfig, LlamaForTokenClassification

LOCAL = Path(__file__).with_name("critic_freeze.py")
SOURCE = LOCAL if LOCAL.exists() else Path(__file__).resolve().parents[2] / "agentlightning/verl/critic_freeze.py"
spec = importlib.util.spec_from_file_location("critic_freeze_fsdp_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_model():
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         num_labels=1, pad_token_id=0, classifier_dropout=0.0)
    config._attn_implementation = "eager"
    critic = LlamaForTokenClassification(config)
    module.freeze_llama_critic_attention(critic)
    return critic


def wrap_model(auto_wrap):
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    critic = make_model()
    policy = get_fsdp_wrap_policy(critic) if auto_wrap else None
    wrapped = FSDP(critic, device_id=torch.device("cpu"), use_orig_params=True,
                   auto_wrap_policy=policy, sharding_strategy=ShardingStrategy.FULL_SHARD,
                   sync_module_states=False)
    assert len(FSDP.fsdp_modules(wrapped)) == (3 if auto_wrap else 1)
    return wrapped


def run_rank(rank, rendezvous, checkpoint, auto_wrap):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer

        # Negative control: mixed requires_grad must not silently flatten together.
        with pytest.raises(ValueError, match="uniform.*requires_grad"):
            FSDP(make_model(), device_id=torch.device("cpu"), use_orig_params=False,
                 sharding_strategy=ShardingStrategy.FULL_SHARD, sync_module_states=False)
        reference = make_model()
        initial = {n: p.detach().clone() for n, p in reference.named_parameters()}
        critic = wrap_model(auto_wrap)
        assert critic.sharding_strategy == ShardingStrategy.FULL_SHARD
        config = FSDPOptimizerConfig(lr=0.01, weight_decay=0.1)
        optimizer = build_optimizer(critic.parameters(), config)
        reference_optimizer = build_optimizer(reference.parameters(), config)
        # Distinct rank inputs test gradient averaging, not just identical replicas.
        all_inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=1, gamma=0.5)
        manager = FSDPCheckpointManager(critic, optimizer, lr_scheduler=scheduler)
        for step in range(2):
            reference_optimizer.zero_grad(set_to_none=True)
            (reference(input_ids=all_inputs).logits - 1).square().mean().backward()
            reference_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            (critic(input_ids=all_inputs[rank:rank + 1]).logits - 1).square().mean().backward()
            assert all(p.grad is None for n, p in critic.named_parameters() if ".self_attn." in n)
            optimizer.step()
            with FSDP.summon_full_params(critic, writeback=False):
                for name, parameter in critic.module.named_parameters():
                    name = name.replace("_fsdp_wrapped_module.", "")
                    if ".self_attn." in name:
                        assert not parameter.requires_grad
                        torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
                    # Reduction order differs between sharded and unsharded execution.
                    torch.testing.assert_close(parameter, dict(reference.named_parameters())[name],
                                               rtol=1e-4, atol=1e-6)
            scheduler.step()
            reference_scheduler.step()
            if step == 0:
                manager.save_checkpoint(checkpoint, global_step=1)
                expected_rng = torch.rand(4)
                # Recreate and freeze before wrapping, just as a fresh worker must.
                critic = wrap_model(auto_wrap)
                optimizer = build_optimizer(critic.parameters(), config)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
                manager = FSDPCheckpointManager(critic, optimizer, lr_scheduler=scheduler)
                manager.load_checkpoint(checkpoint, del_local_after_load=False)
                torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)
                assert scheduler.state_dict() == reference_scheduler.state_dict()
                assert optimizer.param_groups[0]["lr"] == reference_optimizer.param_groups[0]["lr"]
                for other_rank in range(2):
                    for prefix in ("model", "optim", "extra_state"):
                        assert (Path(checkpoint) / f"{prefix}_world_size_2_rank_{other_rank}.pt").is_file()
        with FSDP.summon_full_params(critic, writeback=False):
            assert not torch.equal(critic.module.model.layers[0].mlp.down_proj.weight,
                                   initial["model.layers.0.mlp.down_proj.weight"])
            assert not torch.equal(critic.module.score.weight, initial["score.weight"])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("auto_wrap", [False, True])
def test_two_rank_fsdp_frozen_attention(tmp_path, auto_wrap):
    pytest.importorskip("verl")
    mp.spawn(run_rank, args=((tmp_path / "rendezvous").resolve().as_uri(), str(tmp_path / "checkpoint"), auto_wrap),
             nprocs=2, join=True)
