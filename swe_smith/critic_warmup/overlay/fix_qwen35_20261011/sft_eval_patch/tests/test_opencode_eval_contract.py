"""Exercise the OpenCode entrypoint without model, network, or testbed access."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

PATH = Path(__file__).resolve().parents[3] / "examples/swe_smith/agents/opencode_agent.py"


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("terminal_kind", ["task_stop", "compaction", "unknown"])
@pytest.mark.parametrize("runner_rc", [0, 1])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("binary,ratio", [(1.0, 1.0), (0.0, 0.75)])
def test_main_accepts_five_evaluation_values(tmp_path, monkeypatch, binary, ratio, structured, runner_rc, terminal, terminal_kind):
    events, evaluation_calls = [], []
    def evaluate(meta, timeout, *, f2p_only):
        evaluation_calls.append((meta, timeout, f2p_only))
        return binary, bool(binary), "fixture result", False, ratio
    smith = SimpleNamespace(
        checkout_bug_commit=lambda iid: None,
        relocate_git=lambda: None,
        capture_patch=lambda: "fixture patch",
        evaluate=evaluate,
        post_event=lambda kind, data, **kwargs: events.append((kind, data)),
    )
    def structured_evaluate(actual_smith, meta, timeout, *, f2p_only):
        assert actual_smith is smith
        evaluation_calls.append((meta, timeout, f2p_only))
        return {"reward": binary, "resolved": bool(binary), "reason": "fixture result", "timed_out": False,
                "reward_eligible": True, "evaluation_state": "complete", "grader_protocol": "pytest-structured-v1",
                "p2p_scope": "f2p_same_files", "n_f2p": 4, "n_f2p_pass": int(4 * ratio),
                "n_p2p": 1, "n_p2p_ok": 1}
    monkeypatch.setenv("SMITH_STRUCTURED_EVAL", "1" if structured else "0")
    monkeypatch.setenv("SMITH_REQUIRE_TERMINAL", "1")
    monkeypatch.setitem(sys.modules, "structured_eval", SimpleNamespace(evaluate=structured_evaluate))
    monkeypatch.setitem(sys.modules, "smith_agent", smith)
    monkeypatch.setitem(sys.modules, "opencode_terminal", SimpleNamespace(
        classify_terminal_event=lambda *args, **kwargs: terminal_kind))
    # The script adds its ConfigMap directory on import; restore the test process.
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("opencode_contract", PATH)
    agent = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, agent)
    spec.loader.exec_module(agent)
    monkeypatch.setenv("AGL_TASK_INPUT", "fixture task")
    monkeypatch.setenv("AGL_EVAL_META", json.dumps({"instance_id": "fixture", "repo": "fixture/repo"}))
    monkeypatch.setenv("AGL_OPENAI_BASE_URL", "http://fixture.invalid/v1")
    monkeypatch.setenv("SMITH_EVAL_TIMEOUT", "42")
    monkeypatch.setenv("SMITH_F2P_ONLY", "1")
    monkeypatch.setattr(agent, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(agent, "OPENCODE_BIN", sys.executable)
    def local_path(value):
        return tmp_path / "oc-home" if str(value) == "/tmp/oc-home" else Path(value)
    monkeypatch.setattr(agent, "Path", local_path)
    closed = []
    monkeypatch.setattr(agent, "start_gateway_proxy", lambda *args: (SimpleNamespace(shutdown=lambda: closed.append(True)), 12345))
    output = (json.dumps({"type": "step_finish", "sessionID": "fixture",
                          "part": {"reason": "stop", "sessionID": "fixture"}}) + "\n") if terminal else ""
    class Process:
        pid = 123456789  # No real process is created or signalled.
        returncode = runner_rc
        stdout = io.StringIO(output)
        def poll(self):
            return None if terminal and self.stdout.tell() == 0 else runner_rc
        def wait(self, timeout=None):
            return runner_rc
    monkeypatch.setattr(agent.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(agent.select, "select", lambda streams, *args: (streams, [], []))
    monkeypatch.setattr(agent.os, "killpg", lambda *args: None, raising=False)
    assert agent.main() == 0
    assert evaluation_calls == [({"instance_id": "fixture", "repo": "fixture/repo"}, 42, True)]
    summary = json.loads((tmp_path / "logs/summary.json").read_text())
    assert summary["reward"] == binary
    # A terminal stop is deliberately ended by the supervisor and normalized to
    # success, even if the process exits nonzero after receiving SIGTERM.
    eligible = terminal and terminal_kind == "task_stop"
    assert summary["f2p_ratio"] == (0.0 if structured and not eligible else ratio)
    reward_events = [data for kind, data in events if kind == "reward"]
    assert len(reward_events) == 1
    assert reward_events[0]["value"] == binary  # Do not silently adopt dense reward.
    if structured:
        assert reward_events[0]["reward_eligible"] is eligible
        assert reward_events[0]["trajectory_terminal"] is eligible
        assert reward_events[0]["evaluation_state"] == ("complete" if eligible else "unavailable")
        assert (tmp_path / "logs/evaluation.json").is_file()
    else:
        assert "reward_eligible" not in reward_events[0]
    assert closed == [True]
