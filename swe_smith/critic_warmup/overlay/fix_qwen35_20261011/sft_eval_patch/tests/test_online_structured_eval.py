"""Real online bridge -> restored pytest -> evidence gate -> reward guard."""
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("outcome", ["pass", "fail", "unavailable"])
def test_online_evidence_reaches_guard(tmp_path, monkeypatch, outcome):
    bridge = load("bridge_test", ROOT / "examples/swe_smith/agents/structured_eval.py")
    guard = load("guard_test", ROOT / "agentlightning/verl/reward_guard.py")
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (tmp_path / "test_sample.py").write_text(
        "def test_pass(): pass\ndef test_fail(): assert False\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "tests")
    (tmp_path / "test_sample.py").write_text("# removed\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "Remove F2P Tests")
    monkeypatch.setenv("AGL_PYTHON", sys.executable)
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    # Preserve xdist autoload when installed, just as the online environment does.
    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
    smith = SimpleNamespace(TESTBED=str(tmp_path), _git_base=lambda: ["git"])
    node = "test_sample.py::test_" + outcome
    result = bridge.evaluate(smith, {"FAIL_TO_PASS": [node]}, 30, f2p_only=True)
    event = {"event_type": "reward", "data": {"value": result["reward"],
             "reward_eligible": result["reward_eligible"], "evaluation_state": result["evaluation_state"]}}
    rollout = SimpleNamespace(rollout_id="fixture", final_reward=result["reward"],
                              error_message=None, triplet_events=[event])
    if outcome == "unavailable":
        assert not result["reward_eligible"]
        with pytest.raises(ValueError):
            guard.require_explicit_training_rewards([rollout], require_evaluator_approval=True)
    else:
        assert result["reward_eligible"], result
        assert result["reward"] == (1.0 if outcome == "pass" else 0.0)
        guard.require_explicit_training_rewards([rollout], require_evaluator_approval=True)
