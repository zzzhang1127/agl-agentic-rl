"""Reject incomplete agent trajectories rather than zero-bootstrapping them."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("termination", ROOT / "examples/swe_smith/agents/rollout_termination.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(kind="step_finish", reason="stop", session="session1"):
    return {"type": kind, "sessionID": session, "part": {"reason": reason, "sessionID": session}}


@pytest.mark.parametrize("events,rc,expected", [
    ([event()], 0, True),
    ([event(reason="tool-calls"), event()], 0, True),
    ([event()], 124, False),
    ([event()], 1, False),
    ([], 0, False),
    ([event(reason="length"), event()], 0, False),
    ([event(reason="tool-calls")], 0, False),
    ([event(), event(kind="step_start")], 0, False),
    ([event(), event(session="session2")], 0, False),
    ([event(kind="error"), event()], 0, False),
])
def test_terminal_contract(tmp_path, events, rc, expected):
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    assert module.validate_opencode_terminal(path, rc)["trajectory_terminal"] is expected


@pytest.mark.parametrize("suffix", ['{"type":', '\x1b[31mnot-json', '[]'])
def test_corrupt_log_rejected(tmp_path, suffix):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event()) + "\n" + suffix)
    assert not module.validate_opencode_terminal(path, 0)["trajectory_terminal"]
