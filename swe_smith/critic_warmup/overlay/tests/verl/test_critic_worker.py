# Copyright (c) Microsoft. All rights reserved.

"""Test the real worker override without initializing Ray or a GPU process group."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("verl")
pytest.importorskip("torch")
from omegaconf import OmegaConf
from transformers import LlamaConfig, LlamaForTokenClassification
from verl.utils import model as model_utils
from verl.workers.fsdp_workers import CriticWorker

ROOT = Path(__file__).resolve().parents[2] / "agentlightning/verl"
for name in ("critic_freeze", "critic_worker"):
    local = Path(__file__).with_name(name + ".py")
    path = local if local.exists() else ROOT / (name + ".py")
    spec = importlib.util.spec_from_file_location("agentlightning.verl." + name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
worker_module = sys.modules["agentlightning.verl.critic_worker"]


def config():
    return OmegaConf.create({"strategy": "fsdp", "model": {"fsdp_config": {"use_orig_params": True}}})


@pytest.mark.parametrize("fail", [False, True])
def test_freezing_happens_inside_loader_and_patch_is_restored(fail):
    model = LlamaForTokenClassification(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                                                  num_hidden_layers=1, num_attention_heads=2,
                                                  num_key_value_heads=1, num_labels=1))
    worker = object.__new__(worker_module.AttentionFrozenCriticWorker)
    worker._rank = 0
    original = lambda *a, **k: model

    def parent_builder(self, cfg):
        loaded_model = model_utils.load_valuehead_model("unused")
        assert not loaded_model.model.layers[0].self_attn.q_proj.weight.requires_grad
        assert loaded_model.model.layers[0].mlp.down_proj.weight.requires_grad
        if fail:
            raise RuntimeError("synthetic builder failure")
        return loaded_model, "optimizer", "scheduler"

    with patch.object(model_utils, "load_valuehead_model", original), patch.object(
        CriticWorker, "_build_critic_model_optimizer", parent_builder
    ):
        if fail:
            with pytest.raises(RuntimeError, match="synthetic builder failure"):
                worker._build_critic_model_optimizer(config())
        else:
            assert worker._build_critic_model_optimizer(config())[0] is model
            assert worker.attention_freeze_audit["attention_layers"] == 1
        assert model_utils.load_valuehead_model is original


@pytest.mark.parametrize("key,value", [("strategy", "fsdp2"),
                                       ("model.fsdp_config.use_orig_params", False),
                                       ("model.lora_rank", 8)])
def test_unsupported_worker_configuration_rejected(key, value):
    cfg = config()
    OmegaConf.update(cfg, key, value)
    worker = object.__new__(worker_module.AttentionFrozenCriticWorker)
    with patch.object(CriticWorker, "_build_critic_model_optimizer", side_effect=AssertionError("must not build")):
        with pytest.raises(ValueError):
            worker._build_critic_model_optimizer(cfg)


def test_installed_parent_uses_loader_before_fsdp_and_optimizer():
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(CriticWorker._build_critic_model_optimizer)))
    calls = [(node.func.id, node.lineno) for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    lines = {name: [line for call, line in calls if call == name]
             for name in ("load_valuehead_model", "FSDP", "build_optimizer")}
    assert all(len(items) == 1 for items in lines.values())
    assert lines["load_valuehead_model"][0] < lines["FSDP"][0] < lines["build_optimizer"][0]
