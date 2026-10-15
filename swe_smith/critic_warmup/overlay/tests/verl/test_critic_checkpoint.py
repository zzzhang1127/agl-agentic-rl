"""CPU contract checks for resuming only the critic, never the rollout actor."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checkpoint = load(ROOT / "agentlightning/verl/critic_checkpoint.py")
phase = load(ROOT / "agentlightning/verl/training_phase.py")


@pytest.fixture
def saved(tmp_path):
    folder = tmp_path / "global_step_5"
    critic = folder / "critic"
    (critic / "huggingface").mkdir(parents=True)
    model = tmp_path / "base_model"
    model.mkdir()
    config = dict(model_type="llama", hidden_size=32, num_hidden_layers=2,
                  vocab_size=128, num_attention_heads=4, num_key_value_heads=2)
    (model / "config.json").write_text(json.dumps(config))
    (critic / "huggingface/config.json").write_text(json.dumps(config))
    (folder / "data.pt").write_bytes(b"state")
    for kind in ("model", "optim", "extra_state"):
        for rank in (0, 1):
            (critic / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"state")
    return folder, model


def test_rejects_partial_checkpoint_and_incompatible_model(saved):
    folder, model = saved
    assert checkpoint.validate_critic_checkpoint(str(folder), world_size=2, model_path=str(model))[0] == 5
    shard = folder / "critic/optim_world_size_2_rank_1.pt"
    shard.unlink()
    with pytest.raises(ValueError, match="missing"):
        checkpoint.validate_critic_checkpoint(str(folder), world_size=2, model_path=str(model))
    shard.write_bytes(b"state")
    config = json.loads((model / "config.json").read_text())
    config["hidden_size"] = 64
    (model / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="model mismatch: hidden_size"):
        checkpoint.validate_critic_checkpoint(str(folder), world_size=2, model_path=str(model))


@pytest.mark.parametrize("name", ["global_step_5.tmp", "global_step_5.old", "step5"])
def test_rejects_unpublished_directories(saved, name):
    folder, model = saved
    renamed = folder.with_name(name)
    folder.rename(renamed)
    with pytest.raises(ValueError, match="published"):
        checkpoint.validate_critic_checkpoint(str(renamed), world_size=2, model_path=str(model))


def trainer_with_actual_load_method():
    tree = ast.parse((ROOT / "agentlightning/verl/trainer.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentLightningRayPPOTrainer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_load_checkpoint")
    # Compile the real method in a tiny fake base class; avoid importing GPU/verl.
    fake = ast.ClassDef(name="FakeTrainer", bases=[ast.Name(id="Base", ctx=ast.Load())],
                        keywords=[], body=[method], decorator_list=[])
    namespace = {"Base": type("Base", (), {"_load_checkpoint": lambda self: "native"}),
                 "validate_critic_checkpoint": checkpoint.validate_critic_checkpoint,
                 "actor_update_allowed": phase.actor_update_allowed,
                 "torch": SimpleNamespace(load=Mock(return_value={"batch_position": 5}))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fake], type_ignores=[])), "actual_load_method", "exec"), namespace)
    return namespace["FakeTrainer"](), namespace["torch"]


def test_resume_restores_critic_optimizer_rng_and_dataloader_without_actor(saved):
    folder, model = saved
    trainer, torch = trainer_with_actual_load_method()
    trainer.config = SimpleNamespace(
        agentlightning={"critic_only": True},
        trainer=SimpleNamespace(resume_mode="resume_path", resume_from_path=str(folder),
                                del_local_ckpt_after_load=False, n_gpus_per_node=2, nnodes=1, critic_warmup=0),
        critic=SimpleNamespace(model=SimpleNamespace(path=str(model))),
    )
    trainer.use_critic = True
    trainer.critic_wg = SimpleNamespace(load_checkpoint=Mock())
    trainer.actor_rollout_wg = SimpleNamespace(load_checkpoint=Mock(side_effect=AssertionError("actor must stay fixed")))
    trainer.train_dataloader = SimpleNamespace(load_state_dict=Mock())
    trainer._load_checkpoint()
    trainer.critic_wg.load_checkpoint.assert_called_once_with(str(folder / "critic"), del_local_after_load=False)
    trainer.actor_rollout_wg.load_checkpoint.assert_not_called()
    trainer.train_dataloader.load_state_dict.assert_called_once_with({"batch_position": 5})
    torch.load.assert_called_once_with(str(folder / "data.pt"), weights_only=False)
    assert trainer.global_steps == 5
    assert len(list((folder / "critic").glob("*.pt"))) == 6


def test_noncritic_resume_keeps_native_implementation():
    trainer, _ = trainer_with_actual_load_method()
    trainer.config = SimpleNamespace(agentlightning={"critic_only": False})
    assert trainer._load_checkpoint() == "native"
