"""Tool-output truncation must be a pure function of the message it clips.

The proxy re-renders the whole conversation on every turn.  If the text it
produces for an *unchanged* message differs between turn k and turn k+1, the
re-rendered prompt stops being a continuation of the server's own context and
verl cuts the trajectory into two training rows (§44.11).

The bug: the combined-budget clip was scoped to the tool messages after the
LAST assistant message, so a burst clipped while it was the tail came back in
full once another assistant turn landed behind it.  Run with the pre-patch
backup on PYTHONPATH to watch `test_a_clipped_burst_stays_clipped` fail.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BACKUP = HERE / "opencode_agent.py.bak.09161700"


def _load(path: Path, name: str):
    """Import a module by path, tolerating backups that lack a .py suffix."""
    tmp = None
    if path.suffix != ".py":
        tmp = Path(tempfile.mkdtemp()) / f"{name}.py"
        shutil.copy(path, tmp)
        path = tmp
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        if tmp is not None:
            shutil.rmtree(tmp.parent, ignore_errors=True)


@pytest.fixture(scope="module")
def agent():
    os.environ.setdefault("AGL_SKIP_BOOTSTRAP", "1")
    return _load(HERE / "opencode_agent.py", "opencode_agent_under_test")


def _render(mod, messages):
    body = json.dumps({"messages": messages}).encode("utf-8")
    return json.loads(mod.truncate_openai_body(body))["messages"]


def _tool(ch: str, n: int) -> dict:
    return {"role": "tool", "tool_call_id": f"call_{ch}", "content": ch * n}


def _burst(mod) -> list[dict]:
    """Three parallel tool results that together blow the combined budget."""
    n = mod.TOOL_CHAR_LIMIT
    assert 3 * n > mod.TOOL_COMBINED_CHAR_LIMIT, "fixture no longer exceeds the budget"
    return [_tool("A", n), _tool("B", n), _tool("C", n)]


def test_a_clipped_burst_stays_clipped(agent):
    """The same burst must render identically before and after a later turn."""
    burst = _burst(agent)
    head = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a1"}]
    turn_k = _render(agent, head + burst)
    turn_k1 = _render(agent, head + burst + [{"role": "assistant", "content": "a2"}, _tool("D", 10)])

    for i in range(len(head), len(head) + len(burst)):
        assert turn_k[i]["content"] == turn_k1[i]["content"], (
            f"message {i} re-rendered differently once it was no longer the tail: "
            f"{len(turn_k[i]['content'])} chars -> {len(turn_k1[i]['content'])}"
        )


def test_the_budget_is_actually_biting(agent):
    """Anti-vacuity: without a clip, stability would be trivially true."""
    head = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a1"}]
    rendered = _render(agent, head + _burst(agent))
    assert any("tool output limit" in m["content"] for m in rendered), "no marker -> test proves nothing"


def test_every_burst_is_budgeted_not_just_the_last(agent):
    """An early burst is clipped on its own contents, independently of what follows."""
    burst = _burst(agent)
    msgs = (
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a1"}]
        + burst
        + [{"role": "assistant", "content": "a2"}, _tool("D", 10)]
    )
    rendered = _render(agent, msgs)
    assert "tool output limit" in rendered[4]["content"]
    assert rendered[6]["content"] == "D" * 10, "a small trailing burst must be left alone"


def test_a_burst_within_budget_is_untouched(agent):
    """Regression: don't start clipping conversations that used to pass through."""
    small = [_tool("A", 10), _tool("B", 20)]
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a1"}] + small
    rendered = _render(agent, msgs)
    assert rendered[2]["content"] == "A" * 10
    assert rendered[3]["content"] == "B" * 20


def test_per_message_limit_still_applies(agent):
    """The per-message clip is position-independent already; keep it that way."""
    msgs = [{"role": "user", "content": "u"}, _tool("A", agent.TOOL_CHAR_LIMIT + 500)]
    rendered = _render(agent, msgs)
    assert rendered[1]["content"].startswith("A" * agent.TOOL_CHAR_LIMIT)
    assert "truncated 500 chars" in rendered[1]["content"]


@pytest.mark.skipif(not BACKUP.exists(), reason="pre-patch backup not kept")
def test_the_old_code_really_had_the_bug(agent):
    """Control: the same fixture must FAIL against the pre-patch implementation."""
    old = _load(BACKUP, "opencode_agent_prepatch")
    burst = _burst(old)
    head = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a1"}]
    turn_k = _render(old, head + burst)
    turn_k1 = _render(old, head + burst + [{"role": "assistant", "content": "a2"}, _tool("D", 10)])
    assert turn_k[4]["content"] != turn_k1[4]["content"], "old code was already stable?"
