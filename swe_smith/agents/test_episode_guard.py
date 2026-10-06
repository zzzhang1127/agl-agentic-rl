"""Episode brakes on the opencode proxy: turn cap, loop detector, token bookkeeping,
and the reward shaping that consumes them (floor 0, no timeout penalty)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import opencode_agent as oa  # noqa: E402
import smith_agent as smith  # noqa: E402


def _completion(tool_calls: list[tuple[str, str]] | None, prompt_tokens: int = 100, completion_tokens: int = 10) -> dict:
    message: dict = {"role": "assistant", "content": "" if tool_calls else "done"}
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(tool_calls)
        ]
    return {
        "id": "x",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def test_turn_cap_refuses_the_call_after_the_cap() -> None:
    guard = oa.EpisodeGuard(max_turns=3, loop_repeat=0)
    for i in range(3):
        assert guard.before_request() is None
        assert guard.after_response(_completion([("bash", json.dumps({"command": f"ls {i}"}))])) is None
    stop = guard.before_request()
    assert stop is not None and "turn cap 3" in stop
    assert guard.turn_cap_hit and not guard.loop_detected
    assert guard.n_turns == 3
    # once stopped, stays stopped
    assert guard.before_request() == stop


def test_loop_detector_fires_on_identical_consecutive_tool_calls() -> None:
    guard = oa.EpisodeGuard(max_turns=0, loop_repeat=3)
    same = [("bash", '{"command":"pytest -x"}')]
    assert guard.after_response(_completion(same)) is None
    assert guard.after_response(_completion(same)) is None
    stop = guard.after_response(_completion(same))
    assert stop is not None and "repeated 3x" in stop
    assert guard.loop_detected and not guard.turn_cap_hit
    assert guard.before_request() == stop


def test_loop_counter_resets_when_the_call_changes() -> None:
    guard = oa.EpisodeGuard(max_turns=0, loop_repeat=3)
    a = [("bash", '{"command":"pytest -x"}')]
    b = [("read", '{"filePath":"a.py"}')]
    assert guard.after_response(_completion(a)) is None
    assert guard.after_response(_completion(a)) is None
    assert guard.after_response(_completion(b)) is None
    assert guard.after_response(_completion(a)) is None
    assert guard.after_response(_completion(a)) is None
    assert not guard.loop_detected
    assert guard.repeat_count == 2


def test_plain_text_replies_do_not_count_as_a_loop() -> None:
    guard = oa.EpisodeGuard(max_turns=0, loop_repeat=2)
    for _ in range(5):
        assert guard.after_response(_completion(None)) is None
    assert not guard.loop_detected
    assert guard.n_turns == 5


def test_signature_covers_names_and_arguments_of_all_calls() -> None:
    both = _completion([("read", '{"filePath":"a"}'), ("read", '{"filePath":"b"}')])
    one = _completion([("read", '{"filePath":"a"}')])
    assert oa.EpisodeGuard.tool_signature(both) != oa.EpisodeGuard.tool_signature(one)
    assert oa.EpisodeGuard.tool_signature(_completion(None)) is None
    assert oa.EpisodeGuard.tool_signature({"choices": []}) is None


def test_token_bookkeeping_tracks_max_prompt_and_total_completion() -> None:
    guard = oa.EpisodeGuard(max_turns=0, loop_repeat=0)
    guard.after_response(_completion(None, prompt_tokens=1000, completion_tokens=5))
    guard.after_response(_completion(None, prompt_tokens=30000, completion_tokens=7))
    guard.after_response(_completion(None, prompt_tokens=2000, completion_tokens=1))
    assert guard.max_prompt_tokens == 30000
    assert guard.completion_tokens == 13


def test_synthetic_stop_has_no_tool_calls_and_streams() -> None:
    raw = oa.synthetic_stop_completion("[episode ended by trainer: turn cap 40 reached]")
    comp = json.loads(raw)
    assert comp["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in comp["choices"][0]["message"]
    sse = oa.completion_json_to_sse(raw)
    assert sse is not None and sse.endswith(b"data: [DONE]\n\n")
    assert b"tool_calls" not in sse


def test_reward_shaping_floor_zero_and_penalties() -> None:
    # a solved train rollout within T0 turns keeps 1.0
    assert smith.length_penalized_reward(1.0, 20, 40, t0=25, lam=0.2, is_train=True) == 1.0
    # at the cap it loses the full lambda
    assert smith.length_penalized_reward(1.0, 40, 40, t0=25, lam=0.2, is_train=True) == pytest.approx(0.8)
    # validation is never shaped
    assert smith.length_penalized_reward(1.0, 40, 40, t0=25, lam=0.2, is_train=False) == 1.0
    # unsolved rollouts are untouched (no negative rewards)
    assert smith.length_penalized_reward(0.0, 40, 40, t0=25, lam=0.2, is_train=True) == 0.0
    assert (
        smith.prompt_length_penalty(
            0.0, 60000, soft_start=30000, hard_cap=40960, max_pen=0.2, is_train=True, solved=False
        )
        == 0.0
    )
    shaped = smith.prompt_length_penalty(
        1.0, 40960, soft_start=30000, hard_cap=40960, max_pen=0.2, is_train=True, solved=True
    )
    assert shaped == pytest.approx(0.8)


def test_no_timeout_penalty_left_in_source() -> None:
    src = (HERE / "opencode_agent.py").read_text(encoding="utf-8")
    assert "reward -= 0.2" not in src
    assert "reward = max(0.0, reward)" in src
