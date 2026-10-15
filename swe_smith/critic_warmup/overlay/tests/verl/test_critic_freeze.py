# Copyright (c) Microsoft. All rights reserved.

"""CPU-only freeze tests using real Transformers Llama value-head models."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch
from transformers import LlamaConfig, LlamaForTokenClassification

# Can run directly in a server test directory without changing installed code.
LOCAL = Path(__file__).with_name("critic_freeze.py")
SOURCE = LOCAL if LOCAL.exists() else Path(__file__).resolve().parents[2] / "agentlightning/verl/critic_freeze.py"
spec = importlib.util.spec_from_file_location("critic_freeze_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def model():
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, num_labels=1, pad_token_id=0,
                         classifier_dropout=0.0)
    config._attn_implementation = "eager"
    return LlamaForTokenClassification(config)


def test_frozen_attention_unchanged_mlp_and_value_head_update():
    critic = model()
    audit = module.freeze_llama_critic_attention(critic)
    assert audit["attention_layers"] == 2
    assert len(audit["frozen_attention_names"]) == 8
    before = {n: p.detach().clone() for n, p in critic.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in critic.parameters() if p.requires_grad], lr=0.01)
    critic.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = critic(input_ids=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])).logits
        loss = (logits - 1).square().mean()
        loss.backward()
        assert all(p.grad is None for n, p in critic.named_parameters() if ".self_attn." in n)
        # Earlier MLPs must retain gradients through later frozen attention.
        grad = critic.model.layers[0].mlp.down_proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        optimizer.step()
    changed = []
    for name, parameter in critic.named_parameters():
        if ".self_attn." in name:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        elif not torch.equal(parameter, before[name]):
            changed.append(name)
    assert any(".mlp." in n for n in changed)
    assert any(n.startswith("score.") for n in changed)


def test_verl_optimizer_resume_preserves_attention_freeze(tmp_path):
    pytest.importorskip("verl")
    from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer

    critic = model()
    module.freeze_llama_critic_attention(critic)
    # Match CriticWorker: pass all parameters, not a filtered trainable list.
    config = FSDPOptimizerConfig(lr=0.01, weight_decay=0.1)
    optimizer = build_optimizer(critic.parameters(), config)
    inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    initial_attention = {n: p.detach().clone() for n, p in critic.named_parameters() if ".self_attn." in n}

    def step(current, current_optimizer):
        current_optimizer.zero_grad(set_to_none=True)
        (current(input_ids=inputs).logits - 1).square().mean().backward()
        current_optimizer.step()

    step(critic, optimizer)
    checkpoint = tmp_path / "critic.pt"
    torch.save({"model": critic.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    restored = model()
    # requires_grad is not checkpoint state: freeze every newly built worker.
    module.freeze_llama_critic_attention(restored)
    restored_optimizer = build_optimizer(restored.parameters(), config)
    saved = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(saved["model"])
    restored_optimizer.load_state_dict(saved["optimizer"])
    step(critic, optimizer)
    step(restored, restored_optimizer)
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(parameter, dict(critic.named_parameters())[name], rtol=0, atol=0)
        if ".self_attn." in name:
            torch.testing.assert_close(parameter, initial_attention[name], rtol=0, atol=0)
            assert not parameter.requires_grad and parameter.grad is None
            assert parameter not in restored_optimizer.state
    assert restored.model.layers[0].mlp.down_proj.weight in restored_optimizer.state
    assert restored.score.weight in restored_optimizer.state


def test_checkpoint_alone_does_not_restore_freeze():
    critic = model()
    module.freeze_llama_critic_attention(critic)
    restored = model()
    restored.load_state_dict(critic.state_dict())
    assert all(p.requires_grad for n, p in restored.named_parameters() if ".self_attn." in n)
    module.freeze_llama_critic_attention(restored)
    assert all(not p.requires_grad for n, p in restored.named_parameters() if ".self_attn." in n)


def test_meta_materialization_preserves_freeze_and_audit():
    reference = model()
    state = reference.state_dict()
    with torch.device("meta"):
        critic = LlamaForTokenClassification(reference.config)
    meta_audit = module.freeze_llama_critic_attention(critic)
    assert all(p.is_meta for p in critic.parameters())
    # Match verl's non-primary rank materialization: recurse=False per module.
    for submodule in critic.modules():
        submodule.to_empty(device="cpu", recurse=False)
    critic.load_state_dict(state)
    assert not any(p.is_meta for p in critic.parameters())
    assert all(not p.requires_grad for n, p in critic.named_parameters() if ".self_attn." in n)
    materialized_audit = module.freeze_llama_critic_attention(critic)
    assert materialized_audit == meta_audit
    logits = critic(input_ids=torch.tensor([[1, 2, 3]])).logits
    logits.square().mean().backward()
    assert critic.model.layers[0].mlp.down_proj.weight.grad is not None
    assert critic.model.layers[0].self_attn.q_proj.weight.grad is None


def test_flattened_parameter_rejected_before_mutation():
    critic = model()
    parameter = next(critic.parameters())
    parameter._is_flat_param = True
    with pytest.raises(ValueError, match="before FSDP"):
        module.freeze_llama_critic_attention(critic)
    assert all(p.requires_grad for p in critic.parameters())


def test_validation_failure_does_not_partially_freeze():
    critic = model()
    critic.model.layers[-1].self_attn.o_proj = None
    before = {n: p.requires_grad for n, p in critic.named_parameters()}
    with pytest.raises(ValueError, match="missing o_proj"):
        module.freeze_llama_critic_attention(critic)
    assert {n: p.requires_grad for n, p in critic.named_parameters()} == before


def test_unknown_architecture_rejected():
    critic = model()
    critic.config.model_type = "unknown"
    with pytest.raises(ValueError, match="only verified"):
        module.freeze_llama_critic_attention(critic)


def test_shared_attention_mlp_parameter_rejected():
    critic = model()
    critic.model.layers[0].mlp.down_proj.weight = critic.model.layers[0].self_attn.o_proj.weight
    with pytest.raises(ValueError, match="shared"):
        module.freeze_llama_critic_attention(critic)
