"""Regression cases from grading diagnostics, independent of model answers."""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / "examples/swe_smith/opencode_smoke/eval_inside.py"
spec = importlib.util.spec_from_file_location("grader_node_regression", PATH)
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


def test_color_and_parameter_spaces():
    text = "\x1b[32mPASSED\x1b[0m tests/test_config.py::test_x[not an int]\n"
    assert grader.parse_test_statuses(text) == {"tests/test_config.py::test_x[not an int]": "PASSED"}


def test_summary_failure_and_verbose_progress():
    text = "FAILED tests/a.py::test_x[a b] - AssertionError\ntests/a.py::test_y[c d] PASSED [ 50%]"
    assert grader.parse_test_statuses(text) == {
        "tests/a.py::test_x[a b]": "FAILED", "tests/a.py::test_y[c d]": "PASSED",
    }


def test_conflicting_reports_do_not_become_success():
    assert grader.parse_test_statuses("FAILED tests/a.py::test_x\nPASSED tests/a.py::test_x")["tests/a.py::test_x"] == "CONFLICT"


def test_truncated_node_uses_whitespace_not_prefix():
    truncated = "tests/a.py::test_x[quoted"
    full = "tests/a.py::test_x[quoted spaced-neil]"
    assert grader.resolve_test_nodes([truncated], [full, truncated + "-benjamin]"]) == {truncated: full}


@pytest.mark.parametrize("collected", [[], ["tests/a.py::test_x[a b]", "tests/a.py::test_x[a c]"]])
def test_missing_and_ambiguous_mapping_block(collected):
    with pytest.raises(ValueError):
        grader.resolve_test_nodes(["tests/a.py::test_x[a"], collected)


def test_exact_match_wins():
    node = "tests/a.py::test_x[a]"
    assert grader.resolve_test_nodes([node], [node]) == {node: node}


def test_alias_collision_blocks():
    full = "tests/a.py::test_x[a b]"
    with pytest.raises(ValueError):
        grader.resolve_test_nodes([full, "tests/a.py::test_x[a"], [full])


def test_go_subtests_and_package_failure_are_distinct():
    events = [
        {"Action": "start", "Package": "pkg"},
        {"Action": "run", "Package": "pkg", "Test": "TestRoot"},
        {"Action": "run", "Package": "pkg", "Test": "TestRoot/child"},
        {"Action": "fail", "Package": "pkg", "Test": "TestRoot/child"},
        {"Action": "fail", "Package": "pkg", "Test": "TestRoot"},
        {"Action": "fail", "Package": "pkg"},
    ]
    evidence = grader.structured_go_statuses(events)
    assert evidence["packages"] == {"pkg": "FAILED"}
    assert grader.resolve_go_test_statuses(["TestRoot", "TestRoot/child", "TestRoot/other"], evidence) == {
        "TestRoot": "FAILED", "TestRoot/child": "FAILED", "TestRoot/other": "MISSING",
    }


def test_go_build_failure_does_not_fabricate_test_failures():
    evidence = grader.structured_go_statuses([
        {"Action": "build-fail", "ImportPath": "pkg"},
        {"Action": "fail", "Package": "pkg", "FailedBuild": "pkg"},
    ])
    assert evidence["build_failures"] == ["pkg"]
    assert grader.resolve_go_test_statuses(["TestNotRun"], evidence) == {"TestNotRun": "MISSING"}


def test_go_same_name_in_two_packages_is_ambiguous():
    events = [{"Action": action, "Package": package, "Test": "TestSame"}
              for package in ("pkg1", "pkg2") for action in ("run", "pass")]
    with pytest.raises(ValueError, match="ambiguous"):
        grader.resolve_go_test_statuses(["TestSame"], grader.structured_go_statuses(events))


@pytest.mark.parametrize("actions,expected", [
    (["run", "pass"], "PASSED"), (["run", "skip"], "SKIPPED"),
    (["run"], "MISSING"), (["pass"], "CONFLICT"),
    (["run", "fail", "run", "pass"], "CONFLICT"),
])
def test_go_terminal_evidence(actions, expected):
    events = [{"Action": action, "Package": "pkg", "Test": "ExampleValue"} for action in actions]
    assert grader.structured_go_statuses(events)["tests"]["pkg"]["ExampleValue"] == expected


def phase(when, outcome="passed", *, xfail=False, node="tests/a.py::test_x[a - b]"):
    return {"kind": "test", "nodeid": node, "when": when, "outcome": outcome, "wasxfail": xfail}


@pytest.mark.parametrize("events,expected", [
    ([phase("setup"), phase("call"), phase("teardown")], "PASSED"),
    ([phase("setup"), phase("call")], "MISSING"),
    ([phase("setup"), phase("call", "failed")], "FAILED"),
    ([phase("setup"), phase("call"), phase("teardown", "failed")], "ERROR"),
    ([phase("setup", "failed")], "ERROR"),
    ([phase("setup"), phase("call", "skipped", xfail=True), phase("teardown")], "XFAIL"),
    ([phase("setup"), phase("call", xfail=True), phase("teardown")], "XPASS"),
    ([phase("setup", "skipped"), phase("teardown")], "SKIPPED"),
    ([phase("setup", "skipped", xfail=True), phase("teardown")], "XFAIL"),
    ([phase("call", "failed"), phase("call")], "CONFLICT"),
])
def test_phase_aggregation(events, expected):
    assert grader.structured_pytest_statuses(events) == {"tests/a.py::test_x[a - b]": expected}


def test_partial_json_report_retains_completed_events(tmp_path):
    path = tmp_path / "events.jsonl"
    event = phase("call", "failed")
    path.write_text(json.dumps(event) + '\n{"kind":', encoding="utf-8")
    events, intact = grader.read_pytest_events(str(path))
    assert events == [event]
    assert not intact
    assert grader.structured_pytest_statuses(events)[event["nodeid"]] == "FAILED"


@pytest.mark.parametrize("events", [
    [],
    [{"kind": "collection", "nodes": ["a"]}],
    [{"kind": "collection", "nodes": ["a", "a"]}, {"kind": "session_finish", "exitstatus": 0}],
    [{"kind": "collection", "nodes": ["a"]}, {"kind": "session_finish", "exitstatus": 2}],
    [{"kind": "collection", "nodes": ["a"]}, {"kind": "collection_error"},
     {"kind": "session_finish", "exitstatus": 0}],
])
def test_partial_or_failed_collection_rejected(events):
    with pytest.raises(ValueError):
        grader.collected_pytest_nodes(events)


def test_real_pytest_collection_and_reports(tmp_path):
    (tmp_path / "agl_pytest_report.py").write_text(grader.PYTEST_REPORT_PLUGIN, encoding="utf-8")
    (tmp_path / "test_sample.py").write_text(
        'import pytest\n'
        '@pytest.mark.parametrize("x", [1], ids=["not an int - value"])\n'
        'def test_pass(x): pass\n'
        '@pytest.mark.xfail(reason="known")\n'
        'def test_xfail(): assert False\n'
        'def test_fail(): assert False\n', encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    events_path = tmp_path / "collection.jsonl"
    env["AGL_PYTEST_REPORT"] = str(events_path)
    args = [sys.executable, "-m", "pytest", "--color=no", "-p", "no:cacheprovider", "-p", "agl_pytest_report"]
    collected = subprocess.run(args + ["--collect-only", "test_sample.py"], cwd=tmp_path,
                               env=env, capture_output=True, text=True, timeout=30)
    assert collected.returncode == 0, collected.stdout + collected.stderr
    events, intact = grader.read_pytest_events(str(events_path))
    assert intact
    nodes = grader.collected_pytest_nodes(events)
    assert "test_sample.py::test_pass[not an int - value]" in nodes
    env["AGL_PYTEST_REPORT"] = str(tmp_path / "reports.jsonl")
    result = subprocess.run(args + nodes, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    events, intact = grader.read_pytest_events(env["AGL_PYTEST_REPORT"])
    assert intact
    assert grader.structured_pytest_statuses(events) == {
        nodes[0]: "PASSED", nodes[1]: "XFAIL", nodes[2]: "FAILED",
    }


@pytest.mark.skipif(os.name != "posix", reason="grader process groups require Linux container")
def test_timeout_keeps_partial_output(tmp_path, monkeypatch):
    monkeypatch.setattr(grader, "TESTBED", str(tmp_path))
    output, rc = grader.run_with_deadline(
        [sys.executable, "-c", "import time; print('before-timeout', flush=True); time.sleep(20)"],
        time.monotonic() + 1,
    )
    assert rc == 124
    assert "before-timeout" in output


def test_expired_deadline_never_launches(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("expired budget must not launch a subprocess")
    monkeypatch.setattr(grader.subprocess, "Popen", forbidden)
    assert grader.run_with_deadline(["never"], time.monotonic() - 1)[1] == 124


def test_evaluate_real_restoration_mapping_and_reports(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Grader Test")
    test_file = tmp_path / "test_sample.py"
    test_file.write_text('import pytest\n@pytest.mark.parametrize("x", [1], ids=["not an int - value"])\n'
                         'def test_pass(x): pass\ndef test_fail(): assert False\n', encoding="utf-8")
    git("add", "test_sample.py")
    git("commit", "-m", "initial tests")
    test_file.write_text("# Removed tests\n", encoding="utf-8")
    git("add", "test_sample.py")
    git("commit", "-m", "Remove F2P Tests")
    monkeypatch.setattr(grader, "TESTBED", str(tmp_path))
    monkeypatch.setattr(grader, "HIDDEN_GIT", str(tmp_path / ".git"))
    monkeypatch.setattr(grader, "GIT", "git")
    monkeypatch.setattr(grader, "PYTHON", sys.executable)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    # No xdist autoload in the isolated fixture. Only the probe is stubbed.
    original_run = grader.run_with_deadline
    def run(args, deadline, **kwargs):
        if args == [sys.executable, "-c", "import xdist"]:
            return "", 1
        return original_run(args, deadline, **kwargs)
    monkeypatch.setattr(grader, "run_with_deadline", run)
    node = "test_sample.py::test_pass[not"
    result = grader.evaluate({"FAIL_TO_PASS": [node], "PASS_TO_PASS": ["test_sample.py_extra::test_other"]}, 30)
    assert result["resolved"], result
    assert result["n_p2p"] == 0
    assert result["node_mapping"][node] == "test_sample.py::test_pass[not an int - value]"
    assert result["reward_eligible"]
    gate_path = Path(__file__).resolve().parents[1] / "validate_results.py"
    gate_spec = importlib.util.spec_from_file_location("integration_gate", gate_path)
    gate = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(gate)
    gate.validate_eval({"eval": result, "n_f2p": 1, "smith_rc": 0, "timed_out": False})
    result = grader.evaluate({"FAIL_TO_PASS": [node], "PASS_TO_PASS": ["test_sample.py::test_fail"]}, 30)
    assert not result["resolved"]
    assert result["reward_eligible"]
    assert result["p2p_statuses"] == {"test_sample.py::test_fail": "FAILED"}


def test_restore_failure_blocks_execution(monkeypatch):
    calls = []
    def run(args, deadline, **kwargs):
        calls.append(args)
        return "bad git state", 128
    monkeypatch.setattr(grader, "run_with_deadline", run)
    result = grader.evaluate({"FAIL_TO_PASS": ["tests/a.py::test_x"]}, 30)
    assert len(calls) == 1
    assert not result["resolved"] and not result["reward_eligible"]
    assert result["f2p_statuses"] == {"tests/a.py::test_x": "MISSING"}
    assert result["grading_error"] == "test restoration failed"


def test_non_python_nodes_do_not_run_pytest(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("non-Python nodes must not use pytest")
    monkeypatch.setattr(grader, "run_with_deadline", forbidden)
    result = grader.evaluate({"FAIL_TO_PASS": ["TestGoFunction"]}, 30)
    assert result["evaluation_state"] == "unavailable"
    assert not result["reward_eligible"]
