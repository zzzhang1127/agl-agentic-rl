"""Exercise actual save code with failed shard and tracker writes, without GPUs."""
import ast
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("failure", ["partial", "tracker", None])
def test_previous_checkpoint_survives_failed_publication(tmp_path, monkeypatch, failure):
    source = Path(__file__).resolve().parents[2] / "agentlightning/verl/trainer.py"
    tree = ast.parse(source.read_text())
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
               {"_atomic_replace_dir", "_atomic_write_text", "_prune_old_step_dirs"}]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AgentLightningRayPPOTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_save_checkpoint")
    fake = ast.ClassDef(name="Trainer", bases=[], keywords=[], body=[method], decorator_list=[])
    fs = ModuleType("verl.utils.fs")
    fs.local_mkdir_safe = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
    roles = ModuleType("verl.trainer.ppo.utils")
    roles.Role = SimpleNamespace(Critic="critic")
    monkeypatch.setitem(sys.modules, "verl.utils.fs", fs)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.utils", roles)
    namespace = {"os": os, "shutil": shutil, "torch": SimpleNamespace(
        save=lambda state, path: Path(path).write_bytes(b"data"))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=helpers + [fake], type_ignores=[])), str(source), "exec"), namespace)
    old = tmp_path / "global_step_20"
    old.mkdir()
    (old / "sentinel").write_text("recoverable")
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("20")

    def save(path, *args, **kwargs):
        Path(path).mkdir()
        for kind in ("model", "optim", "extra_state"):
            for rank in (0, 1):
                if failure == "partial" and kind == "optim" and rank == 1:
                    continue
                (Path(path) / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"state")

    trainer = namespace["Trainer"]()
    trainer.config = SimpleNamespace(agentlightning={"critic_only": True}, trainer=SimpleNamespace(
        default_local_dir=str(tmp_path), default_hdfs_dir=None, n_gpus_per_node=2, nnodes=1))
    trainer.global_steps = 40
    trainer.use_critic = True
    trainer.critic_wg = SimpleNamespace(save_checkpoint=save)
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {})
    if failure == "tracker":
        namespace["_atomic_write_text"] = lambda *args: (_ for _ in ()).throw(OSError("injected tracker failure"))
    if failure:
        with pytest.raises((ValueError, OSError)):
            trainer._save_checkpoint()
        assert (old / "sentinel").read_text() == "recoverable"
        assert tracker.read_text() == "20"
    else:
        trainer._save_checkpoint()
        assert tracker.read_text() == "40"
        assert not old.exists()
        assert len(list((tmp_path / "global_step_40/critic").glob("*.pt"))) == 6
