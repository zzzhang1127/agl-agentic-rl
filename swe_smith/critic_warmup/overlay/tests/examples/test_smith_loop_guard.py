# Copyright (c) Microsoft. All rights reserved.

"""Coverage for the SWE-smith opt-in no-progress loop guard and termination status.

The loop guard must be conservative: a stuck cycle is only declared when the
*normalized action* + *equivalent observation* + *unchanged workspace
fingerprint* triple repeats periodically, never on command repetition alone.
These tests exercise the detector directly (pure) and the ``run_agent_loop``
integration (with ``_run`` and the fingerprint monkeypatched so no shell or
workspace is actually used).
"""

from __future__ import annotations

import shutil
import subprocess
import types
from typing import Any

import pytest

from examples.swe_smith.agents import loop_guard, smith_agent


def test_normalize_action_strips_only_outer_whitespace() -> None:
    assert loop_guard.normalize_action("ls -la\n") == "ls -la"
    assert loop_guard.normalize_action("  echo hi  ") == "echo hi"
    # Interior whitespace is preserved verbatim: collapsing it would change the
    # meaning of shell quoted strings, so distinct commands stay distinct.
    assert loop_guard.normalize_action('echo "a  b"') == 'echo "a  b"'
    assert loop_guard.normalize_action("echo 'a  b'") == "echo 'a  b'"
    assert loop_guard.normalize_action("ls  -la") == "ls  -la"  # not collapsed
    assert loop_guard.normalize_action("") == ""


# --- Pure LoopGuard cycle detection ----------------------------------------


def test_period1_exact_stuck_cycle_detected() -> None:
    guard = loop_guard.LoopGuard(min_repeats=3, max_period=2)
    fired = [guard.observe("pytest -q", "0\nall passed", "fp") for _ in range(3)]
    assert fired == [False, False, True]
    assert guard.detected and guard.period == 1


def test_period2_alternating_cycle_detected() -> None:
    guard = loop_guard.LoopGuard(min_repeats=3, max_period=2)
    seq = [("cmdA", "obsA"), ("cmdB", "obsB")] * 3  # AB AB AB
    fired = [guard.observe(a, o, "fp") for a, o in seq]
    assert fired[-1] is True and guard.period == 2


def test_two_repeats_are_not_enough() -> None:
    guard = loop_guard.LoopGuard(min_repeats=3, max_period=2)
    fired = [guard.observe("pytest -q", "same", "fp") for _ in range(2)]
    assert fired == [False, False] and not guard.detected


def test_changed_code_then_same_test_is_not_loop() -> None:
    # Same command + same observation, but the workspace fingerprint changes each
    # turn (the agent is editing), so this is progress, not a stuck loop.
    guard = loop_guard.LoopGuard(min_repeats=3, max_period=2)
    fired = [guard.observe("pytest -q", "1 passed", f"fp{i}") for i in range(3)]
    assert not any(fired) and not guard.detected


def test_different_paths_are_not_loop() -> None:
    # Same command on different files -> different observation.
    guard = loop_guard.LoopGuard(min_repeats=3, max_period=2)
    fired = [guard.observe("sed -n 1p x.py", f"line {i}", "fp") for i in range(3)]
    assert not any(fired)


def test_nonidentical_output_is_not_loop() -> None:
    # Output keeps ticking (a timer / counter), so it is polling-with-progress.
    guard = loop_guard.LoopGuard(min_repeats=3, max_period=2)
    fired = [guard.observe("pytest -q", f"3 passed in {i}s", "fp") for i in range(3)]
    assert not any(fired)


def test_missing_fingerprint_never_false_positives() -> None:
    guard = loop_guard.LoopGuard(min_repeats=3, max_period=2)
    fired = [guard.observe("pytest -q", "same", None) for _ in range(10)]
    assert not any(fired) and not guard.detected


def test_none_fingerprint_resets_history() -> None:
    guard = loop_guard.LoopGuard(min_repeats=3, max_period=2)
    assert guard.observe("pytest -q", "same", "fp") is False
    assert guard.observe("pytest -q", "same", "fp") is False
    assert guard.observe("pytest -q", "same", None) is False  # clears the buffer
    assert guard.observe("pytest -q", "same", "fp") is False  # back to 1
    assert guard.observe("pytest -q", "same", "fp") is False  # back to 2
    assert not guard.detected


def test_polling_with_progress_not_killed() -> None:
    guard = loop_guard.LoopGuard(min_repeats=3, max_period=2)
    fired = [guard.observe("sleep 1; cat out.log", f"line {i}", "fp") for i in range(20)]
    assert not any(fired) and not guard.detected


def test_reset_clears_detection() -> None:
    guard = loop_guard.LoopGuard(min_repeats=2, max_period=1)
    guard.observe("a", "o", "fp")
    guard.observe("a", "o", "fp")  # detected (min_repeats=2)
    assert guard.detected
    guard.reset()
    assert not guard.detected and guard.period == 0
    assert guard.observe("a", "o", "fp") is False  # fresh buffer, only 1


# --- Workspace fingerprint --------------------------------------------------


def test_workspace_fingerprint_changes_with_content(tmp_path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    fp1 = loop_guard.workspace_fingerprint(str(tmp_path))
    (tmp_path / "a.py").write_text("x = 2\n")
    fp2 = loop_guard.workspace_fingerprint(str(tmp_path))
    assert fp1 is not None and fp2 is not None and fp1 != fp2


def test_workspace_fingerprint_sees_untracked_add(tmp_path) -> None:
    fp1 = loop_guard.workspace_fingerprint(str(tmp_path))
    (tmp_path / "new.py").write_text("print(1)\n")
    fp2 = loop_guard.workspace_fingerprint(str(tmp_path))
    assert fp1 != fp2


def test_workspace_fingerprint_ignores_git_and_caches(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"\x00\x01")
    (tmp_path / "src.py").write_text("a\n")
    fp1 = loop_guard.workspace_fingerprint(str(tmp_path))
    (tmp_path / ".git" / "HEAD").write_text("other\n")
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"\x00\x02")
    fp2 = loop_guard.workspace_fingerprint(str(tmp_path))
    assert fp1 == fp2


def test_workspace_fingerprint_missing_root_returns_none(tmp_path) -> None:
    assert loop_guard.workspace_fingerprint(str(tmp_path / "nope")) is None


def test_workspace_fingerprint_changes_on_commit_cycle(tmp_path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@e.st"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "c1"], check=True)
    fp_clean = loop_guard.workspace_fingerprint(str(tmp_path), git_base=["git"])

    # Edit without committing: working tree dirty -> fingerprint changes.
    (tmp_path / "a.py").write_text("x = 2\n")
    fp_dirty = loop_guard.workspace_fingerprint(str(tmp_path), git_base=["git"])

    # Commit the edit: tree is clean again, but the HEAD tree identity changed,
    # so the fingerprint must still differ (without HEAD^{tree} it would equal
    # fp_clean and mask an edit+commit cycle).
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "c2"], check=True)
    fp_clean2 = loop_guard.workspace_fingerprint(str(tmp_path), git_base=["git"])

    assert fp_clean is not None and fp_dirty is not None and fp_clean2 is not None
    assert fp_clean != fp_dirty
    assert fp_clean != fp_clean2


def test_workspace_fingerprint_bounded_file_count(tmp_path) -> None:
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text(f"x={i}\n")
    assert loop_guard.workspace_fingerprint(str(tmp_path), max_files=3) is None


def test_workspace_fingerprint_bounded_bytes(tmp_path) -> None:
    (tmp_path / "a.py").write_text("x" * 100)
    assert loop_guard.workspace_fingerprint(str(tmp_path), max_bytes=10) is None


def test_workspace_fingerprint_hashes_binary_content(tmp_path) -> None:
    (tmp_path / "img.bin").write_bytes(b"\x00\x01\x02\xff")
    fp1 = loop_guard.workspace_fingerprint(str(tmp_path))
    (tmp_path / "img.bin").write_bytes(b"\x00\x01\x02\xfe")
    fp2 = loop_guard.workspace_fingerprint(str(tmp_path))
    assert fp1 is not None and fp2 is not None and fp1 != fp2


def test_workspace_fingerprint_unreadable_file_returns_none(tmp_path, monkeypatch) -> None:
    (tmp_path / "a.py").write_text("x\n")
    monkeypatch.setattr(loop_guard, "_hash_file_into", lambda *a, **k: False)
    assert loop_guard.workspace_fingerprint(str(tmp_path)) is None


def test_workspace_fingerprint_untracked_unicode_and_special_chars(tmp_path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@e.st"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "tracked.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)

    fp1 = loop_guard.workspace_fingerprint(str(tmp_path), git_base=["git"])

    names = ["中文.py", "file with spaces.py"]
    for n in names:
        (tmp_path / n).write_text("x\n")
    for raw in ('file"quotes".py', "file\ttabs.py"):
        try:
            (tmp_path / raw).write_text("x\n")
            names.append(raw)
        except OSError:
            pass

    fp2 = loop_guard.workspace_fingerprint(str(tmp_path), git_base=["git"])
    assert fp2 is not None and fp1 != fp2

    (tmp_path / names[0]).write_text("y\n")
    fp3 = loop_guard.workspace_fingerprint(str(tmp_path), git_base=["git"])
    assert fp3 is not None and fp2 != fp3


def test_workspace_fingerprint_rename_copy_skips_second_path(tmp_path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@e.st"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)

    fp1 = loop_guard.workspace_fingerprint(str(tmp_path), git_base=["git"])

    # Produce a porcelain rename record: R  new\0old\0
    subprocess.run(["git", "-C", str(tmp_path), "mv", "a.py", "b.py"], check=True)
    (tmp_path / "untracked.py").write_text("u\n")

    fp2 = loop_guard.workspace_fingerprint(str(tmp_path), git_base=["git"])
    assert fp2 is not None and fp1 != fp2


# --- Env config parsing -----------------------------------------------------


def test_loop_guard_config_default_disabled(monkeypatch) -> None:
    monkeypatch.delenv("SMITH_LOOP_GUARD", raising=False)
    assert smith_agent.loop_guard_config_from_env() is None


def test_loop_guard_config_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SMITH_LOOP_GUARD", "1")
    monkeypatch.setenv("SMITH_LOOP_GUARD_MIN_REPEATS", "4")
    monkeypatch.setenv("SMITH_LOOP_GUARD_MAX_PERIOD", "3")
    cfg = smith_agent.loop_guard_config_from_env()
    assert cfg == {"min_repeats": 4, "max_period": 3}


def test_loop_guard_disabled_without_module_returns_none(monkeypatch) -> None:
    monkeypatch.delenv("SMITH_LOOP_GUARD", raising=False)
    monkeypatch.setattr(smith_agent, "loop_guard", None)
    assert smith_agent.loop_guard_config_from_env() is None


def test_loop_guard_enabled_without_module_raises(monkeypatch) -> None:
    monkeypatch.setenv("SMITH_LOOP_GUARD", "1")
    monkeypatch.setattr(smith_agent, "loop_guard", None)
    with pytest.raises(SystemExit):
        smith_agent.loop_guard_config_from_env()


# --- run_agent_loop integration --------------------------------------------


class _CountingClient:
    """Client with a mutable ``calls`` counter the test can read."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

        outer = self

        class _Completions:
            def create(self, **_kwargs: Any) -> Any:
                outer.calls += 1
                choice = types.SimpleNamespace(
                    message=types.SimpleNamespace(content=outer._content),
                    finish_reason="stop",
                )
                usage = types.SimpleNamespace(prompt_tokens=5)
                return types.SimpleNamespace(choices=[choice], usage=usage)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_run_agent_loop_detects_stuck_loop_and_reports_loop_detected(monkeypatch) -> None:
    monkeypatch.setattr(smith_agent, "_run", lambda _action, _timeout: ("all passed\n", 0))
    monkeypatch.setattr(smith_agent.loop_guard, "workspace_fingerprint", lambda *a, **k: "fp")
    client = _CountingClient("```bash\npytest -q\n```")
    status: dict[str, Any] = {}

    submitted, turns_used, _mpt = smith_agent.run_agent_loop(
        client,
        "fix the bug",
        max_turns=100,
        cmd_timeout=1,
        obs_cap=100,
        max_tokens=64,
        loop_guard_cfg={"min_repeats": 3, "max_period": 2},
        status_out=status,
    )

    assert submitted is False
    assert status["termination"] == "loop_detected"
    assert client.calls == 3  # stops at the 3rd repeat, does not spin to max_turns
    assert turns_used == 3


def test_run_agent_loop_default_disabled_reaches_turn_limit(monkeypatch) -> None:
    monkeypatch.setattr(smith_agent, "_run", lambda _action, _timeout: ("out\n", 0))
    client = _CountingClient("```bash\nls\n```")
    status: dict[str, Any] = {}

    submitted, turns_used, _mpt = smith_agent.run_agent_loop(
        client,
        "fix the bug",
        max_turns=4,
        cmd_timeout=1,
        obs_cap=100,
        max_tokens=64,
        status_out=status,
    )

    assert submitted is False
    assert turns_used == 4
    assert status["termination"] == "turn_limit"


def test_run_agent_loop_submit_sets_submit_termination(monkeypatch) -> None:
    monkeypatch.setattr(
        smith_agent,
        "_run",
        lambda _action, _timeout: (smith_agent.SUBMIT_MARKER + "\n", 0),
    )
    client = _CountingClient("```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```")
    status: dict[str, Any] = {}

    submitted, turns_used, _mpt = smith_agent.run_agent_loop(
        client,
        "fix the bug",
        max_turns=4,
        cmd_timeout=1,
        obs_cap=100,
        max_tokens=64,
        status_out=status,
    )

    assert submitted is True
    assert turns_used == 1
    assert status["termination"] == "submit"


def test_run_agent_loop_wall_timeout_reports_wall_timeout(monkeypatch) -> None:
    monkeypatch.setattr(smith_agent, "_run", lambda _action, _timeout: ("out\n", 0))
    # First call sets the deadline (t=1000 + 30s); every later call is past it.
    clock = iter([1000.0, 2000.0])
    monkeypatch.setattr(smith_agent.time, "time", lambda: next(clock, 2000.0))
    client = _CountingClient("```bash\nls\n```")
    status: dict[str, Any] = {}

    submitted, turns_used, _mpt = smith_agent.run_agent_loop(
        client,
        "fix the bug",
        max_turns=100,
        cmd_timeout=1,
        obs_cap=100,
        max_tokens=64,
        wall_timeout_s=30.0,
        status_out=status,
    )

    assert submitted is False
    assert turns_used == 0
    assert client.calls == 0  # wall check fires before the first query
    assert status["termination"] == "wall_timeout"
