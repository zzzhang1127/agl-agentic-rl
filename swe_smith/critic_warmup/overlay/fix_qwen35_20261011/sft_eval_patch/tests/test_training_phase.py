"""Critic-only interlock tests, independent of distributed training dependencies."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("training_phase_test", ROOT / "agentlightning/verl/training_phase.py")
phase = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase)


@pytest.mark.parametrize("step", [0, 1, 100, 1000000])
def test_critic_only_never_releases_actor(step):
    assert not phase.actor_update_allowed(critic_only=True, use_critic=True, global_steps=step, critic_warmup=0)


@pytest.mark.parametrize("step,expected", [(0, False), (9, False), (10, True), (11, True)])
def test_existing_step_warmup_unchanged(step, expected):
    assert phase.actor_update_allowed(critic_only=False, use_critic=True, global_steps=step, critic_warmup=10) is expected


def test_critic_only_without_critic_rejected():
    with pytest.raises(ValueError, match="enabled critic"):
        phase.actor_update_allowed(critic_only=True, use_critic=False, global_steps=0, critic_warmup=0)


@pytest.mark.parametrize("value", ["false", None, 0, 1])
def test_ambiguous_flag_rejected(value):
    with pytest.raises(ValueError, match="booleans"):
        phase.actor_update_allowed(critic_only=value, use_critic=True, global_steps=0, critic_warmup=0)


@pytest.mark.parametrize("name", ["global_steps", "critic_warmup"])
@pytest.mark.parametrize("value", [-1, True, False, 1.5, "10", None])
def test_invalid_step_configuration_rejected(name, value):
    arguments = dict(critic_only=True, use_critic=True, global_steps=0, critic_warmup=0)
    arguments[name] = value
    with pytest.raises(ValueError, match="nonnegative integer"):
        phase.actor_update_allowed(**arguments)


@pytest.mark.parametrize("value", ["true", None, 0, 1])
def test_ambiguous_use_critic_rejected(value):
    with pytest.raises(ValueError, match="booleans"):
        phase.actor_update_allowed(critic_only=False, use_critic=value, global_steps=0, critic_warmup=0)


@pytest.mark.parametrize("step,warmup", [(0, 100), (100, 100), (1000000, 100)])
def test_critic_only_overrides_step_warmup(step, warmup):
    assert not phase.actor_update_allowed(critic_only=True, use_critic=True, global_steps=step, critic_warmup=warmup)


def test_noncritic_algorithm_retains_existing_behavior():
    assert phase.actor_update_allowed(critic_only=False, use_critic=False, global_steps=0, critic_warmup=0)


def test_real_trainer_actor_call_is_guarded():
    # Execute the actual trainer interlock/branch with a sentinel actor method,
    # avoiding imports of Ray/verl or any distributed initialization.
    tree = ast.parse((ROOT / "agentlightning/verl/trainer.py").read_text(encoding="utf-8"))
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "update_actor" for t in n.targets)]
    branches = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name) and n.test.id == "update_actor"]
    assert len(assignments) == len(branches) == 1
    actor_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == "_update_actor"]
    assert len(actor_calls) == 1
    assert actor_calls[0] in list(ast.walk(branches[0]))
    code = compile(ast.Module(body=[assignments[0], branches[0]], type_ignores=[]), "trainer-interlock", "exec")
    def forbidden(*args, **kwargs):
        raise AssertionError("critic-only must never reach the actor branch")
    trainer = SimpleNamespace(
        config=SimpleNamespace(agentlightning={"critic_only": True}, trainer=SimpleNamespace(critic_warmup=0)),
        use_critic=True, global_steps=1000000, _update_actor=forbidden,
    )
    scope = {"self": trainer, "actor_update_allowed": phase.actor_update_allowed, "marked_timer": forbidden}
    exec(code, scope)
    assert scope["update_actor"] is False


def test_real_fit_rejects_invalid_phase_before_other_work():
    tree = ast.parse((ROOT / "agentlightning/verl/trainer.py").read_text(encoding="utf-8"))
    fits = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fit"]
    assert len(fits) == 1
    # The first executable statement after the docstring must be the preflight.
    preflight = fits[0].body[1]
    assert isinstance(preflight, ast.Expr) and isinstance(preflight.value, ast.Call)
    assert isinstance(preflight.value.func, ast.Name)
    assert preflight.value.func.id == "actor_update_allowed"
    code = compile(ast.Module(body=[preflight], type_ignores=[]), "trainer-preflight", "exec")
    trainer = SimpleNamespace(
        config=SimpleNamespace(agentlightning={"critic_only": True}, trainer=SimpleNamespace(critic_warmup=0)),
        use_critic=False,
    )
    with pytest.raises(ValueError, match="enabled critic"):
        exec(code, {"self": trainer, "actor_update_allowed": phase.actor_update_allowed})
