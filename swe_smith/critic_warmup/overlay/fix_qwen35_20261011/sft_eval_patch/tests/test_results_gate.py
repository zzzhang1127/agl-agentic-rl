import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

TOOL = Path(__file__).resolve().parents[1] / "validate_results.py"
spec = importlib.util.spec_from_file_location("results_gate", TOOL)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def result(iid, passed=True, trajectory_timeout=False, eval_timeout=False):
    return {
        "instance_id": iid, "n_f2p": 1,
        "smith_rc": 124 if trajectory_timeout else 0,
        "timed_out": trajectory_timeout,
        "eval": {
            "resolved": passed and not eval_timeout,
            "timed_out": eval_timeout,
            "pytest_rc": 124 if eval_timeout else (0 if passed else 1),
            "n_f2p": 1, "n_f2p_pass": int(passed and not eval_timeout),
            "n_p2p": 0, "n_p2p_ok": 0,
            "f2p_statuses": {"test_a": "MISSING" if eval_timeout else ("PASSED" if passed else "FAILED")},
            "p2p_statuses": {},
        },
    }


@pytest.mark.parametrize("overrides", [
    {"evaluation_state": "unavailable"},
    {"evaluation_state": "timeout"},
    {"reward_eligible": False},
    {"report_intact": False},
    {"restore_rc": 128},
    {"collection_rc": 2},
    {"grader_protocol": "unknown"},
])
def test_structured_gate_rejects_unavailable_even_with_pass_counts(overrides):
    row = result("a")
    row["eval"].update({"grader_protocol": "pytest-structured-v1", "evaluation_state": "complete",
                        "reward_eligible": True, "report_intact": True, "restore_rc": 0, "collection_rc": 0})
    row["eval"].update(overrides)
    with pytest.raises(gate.InvalidResults):
        gate.validate_eval(row)


def structured_result():
    row = result("a")
    node = "tests/a.py::test_x[a - b]"
    ev = row["eval"]
    ev.update({"grader_protocol": "pytest-structured-v1", "evaluation_state": "complete",
               "reward_eligible": True, "report_intact": True, "restore_rc": 0, "collection_rc": 0,
               "p2p_scope": "f2p_same_files", "f2p_statuses": {node: "PASSED"},
               "collected_nodes": [node], "node_mapping": {node: node},
               "collection_events": [{"kind": "collection", "nodes": [node]},
                                     {"kind": "session_finish", "exitstatus": 0}],
               "test_events": [{"kind": "test", "nodeid": node, "when": when,
                                "outcome": "passed", "wasxfail": False}
                               for when in ("setup", "call", "teardown")]
                              + [{"kind": "session_finish", "exitstatus": 0}]})
    ev["requested_nodes_sha256"] = hashlib.sha256(
        json.dumps([node], ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    rehash_events(ev)
    return row


def rehash_events(ev):
    for key, digest in (("collection_events", "collection_sha256"), ("test_events", "report_sha256")):
        ev[digest] = hashlib.sha256("".join(json.dumps(e, ensure_ascii=True) + "\n" for e in ev[key]).encode()).hexdigest()


def test_structured_valid_evidence_accepted():
    row = structured_result()
    assert gate.validate_eval(row) is row["eval"]


@pytest.mark.parametrize("mode", ["missing_events", "bad_hash", "failed_call", "missing_teardown",
                                   "duplicate_call", "bad_mapping", "collection_failure", "bool_rc"])
def test_structured_gate_recomputes_evidence(mode):
    row = structured_result()
    ev = row["eval"]
    if mode == "missing_events":
        ev["test_events"] = []
    elif mode == "bad_hash":
        ev["report_sha256"] = "0" * 64
    elif mode == "failed_call":
        ev["test_events"][1]["outcome"] = "failed"
    elif mode == "missing_teardown":
        ev["test_events"].pop(2)
    elif mode == "duplicate_call":
        ev["test_events"].insert(2, dict(ev["test_events"][1]))
    elif mode == "bad_mapping":
        ev["node_mapping"][next(iter(ev["node_mapping"]))] = "not-collected"
    elif mode == "collection_failure":
        ev["collection_events"][-1]["exitstatus"] = 2
    elif mode == "bool_rc":
        ev["restore_rc"] = False
    if mode != "bad_hash":
        rehash_events(ev)
    with pytest.raises(gate.InvalidResults):
        gate.validate_eval(row)


def write_result(directory, row):
    p = directory / row["instance_id"] / "result.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(row), encoding="utf-8")
    return p


@pytest.fixture
def run(tmp_path):
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
    dirs = [tmp_path / "shard0", tmp_path / "shard1"]
    for d in dirs:
        d.mkdir()
    write_result(dirs[0], result("a", trajectory_timeout=True))
    write_result(dirs[1], result("b", passed=False))
    write_result(dirs[0], result("c", passed=False, eval_timeout=True))
    output = tmp_path / "summary.json"

    def invoke(step="10", directories=None):
        args = [sys.executable, str(TOOL), "--ids-file", str(ids), "--checkpoint-step", str(step), "--output", str(output)]
        for d in directories or dirs:
            args.extend(["--shard-dir", str(d)])
        return subprocess.run(args, capture_output=True, text=True, timeout=10)

    return ids, dirs, output, invoke


def test_cli_complete_counts_and_no_history_scan(run, tmp_path):
    ids, dirs, output, invoke = run
    write_result(tmp_path / "old_run", result("unrelated"))
    p = invoke()
    assert p.returncode == 0, p.stderr
    s = json.loads(output.read_text())
    assert s["complete"] is True
    assert (s["n_expected"], s["n_completed"], s["n_resolved"]) == (3, 3, 1)
    assert s["resolved_rate"] == 1 / 3
    assert s["checkpoint_step"] == 10
    assert (s["n_trajectory_timeout"], s["n_eval_timeout"], s["n_outer_timeout_rc124"]) == (1, 1, 1)
    assert s["ids_sha256"] == hashlib.sha256(ids.read_bytes()).hexdigest()


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate", "bad_json", "wrong_id", "nested", "duplicate_key"])
def test_set_and_layout_errors_fail_closed(run, mode):
    _, dirs, output, invoke = run
    p = dirs[0] / "a" / "result.json"
    if mode == "missing":
        p.unlink()
    elif mode == "extra":
        write_result(dirs[1], result("extra"))
    elif mode == "duplicate":
        write_result(dirs[1], result("a", trajectory_timeout=True))
    elif mode == "bad_json":
        p.write_text("{")
    elif mode == "wrong_id":
        p.write_text(json.dumps(result("other")))
    elif mode == "nested":
        write_result(dirs[0] / "old", result("old"))
    else:
        p.write_text('{"instance_id":"a", "instance_id":"a"}')
    proc = invoke()
    assert proc.returncode != 0
    assert not output.exists()
    assert "rejected" in proc.stderr


@pytest.mark.parametrize("ev", [None, {}, {"resolved": False},
    {"resolved": False, "reason": "eval parse failed"},
    {"resolved": False, "reason": "missing git dir"},
    {"resolved": False, "reason": "checkout failed", "n_f2p": 1, "n_f2p_pass": 0},
])
def test_invalid_or_infra_eval_not_scored_failure(run, ev):
    _, dirs, output, invoke = run
    r = result("a")
    r["eval"] = ev
    write_result(dirs[0], r)
    assert invoke().returncode != 0
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["runner", "resolved_type", "timeout_type", "count_type", "bad_pass_count", "missing_status", "collection_error", "missing_tests", "inconsistent_resolved", "timeout_success"])
def test_schema_and_unavailable_evaluation(run, mutation):
    _, dirs, output, invoke = run
    r = result("a")
    ev = r["eval"]
    if mutation == "runner": r["error"] = "docker failed"
    if mutation == "resolved_type": ev["resolved"] = "false"
    if mutation == "timeout_type": r["timed_out"] = 0
    if mutation == "count_type": ev["n_f2p"] = True
    if mutation == "bad_pass_count": ev["n_f2p_pass"] = 2
    if mutation == "missing_status": del ev["f2p_statuses"]
    if mutation == "collection_error": ev["pytest_rc"] = 2
    if mutation == "missing_tests": ev.update(resolved=False, n_f2p_pass=0, f2p_statuses={"test_a": "MISSING"}, pytest_rc=1)
    if mutation == "inconsistent_resolved": ev["resolved"] = False
    if mutation == "timeout_success": ev.update(timed_out=True, pytest_rc=124)
    write_result(dirs[0], r)
    assert invoke().returncode != 0
    assert not output.exists()


def test_marker_timed_out_with_non_124_rc_rejected(run):
    """eval_inside sets timed_out=True when the pytest *output* contains the
    literal "[timed out after" marker even if pytest exited with rc 0/1. That is
    a false positive, not a harness timeout. The gate must reject it fail-closed
    (rc must be 124) instead of scoring it as a completed+failed result."""
    _, dirs, output, invoke = run
    r = result("a", passed=False)
    r["eval"].update(timed_out=True, pytest_rc=1, resolved=False,
                     f2p_statuses={"test_a": "FAILED"})
    write_result(dirs[0], r)
    proc = invoke()
    assert proc.returncode != 0
    assert not output.exists()
    assert "inconsistent evaluation timeout" in proc.stderr


def test_harness_timeout_rc124_scored_completed(run):
    """The legitimate harness timeout (rc==124, timed_out=True, resolved=False)
    IS a scored completed+failed result (not unavailable)."""
    _, dirs, output, invoke = run
    r = result("a", passed=False, eval_timeout=True)
    write_result(dirs[0], r)
    proc = invoke()
    assert proc.returncode == 0, proc.stderr
    s = json.loads(output.read_text())
    assert s["n_completed"] == 3 and s["n_eval_timeout"] == 2


@pytest.mark.parametrize("step", ["0", "-1", "True", "1.5"])
def test_invalid_step_cli(run, step):
    _, _, output, invoke = run
    assert invoke(step).returncode != 0
    assert not output.exists()


@pytest.mark.parametrize("step", [True, False, 1.0, "1", 0, -1])
def test_step_api_strict_integer(run, step):
    ids, dirs, _, _ = run
    with pytest.raises(gate.InvalidResults):
        gate.collect(ids, dirs, step)


@pytest.mark.parametrize("ids_value", [[], ["a", "a"], [1], {}, [""]])
def test_bad_manifest(run, ids_value):
    ids, _, output, invoke = run
    ids.write_text(json.dumps(ids_value))
    assert invoke().returncode != 0
    assert not output.exists()


def test_duplicate_and_overlapping_directories(run):
    _, dirs, output, invoke = run
    assert invoke(directories=[dirs[0], dirs[0]]).returncode != 0
    assert invoke(directories=[dirs[0], dirs[0] / "a"]).returncode != 0
    assert not output.exists()


def test_repeated_success_and_failed_recheck_never_overwrite(run):
    _, dirs, output, invoke = run
    assert invoke().returncode == 0
    original = output.read_bytes()
    assert invoke().returncode != 0
    assert output.read_bytes() == original
    (dirs[0] / "a" / "result.json").write_text("{")
    assert invoke().returncode != 0
    assert output.read_bytes() == original


def test_atomic_publish_race_preserves_winner(tmp_path, monkeypatch):
    output = tmp_path / "summary.json"
    link = gate.os.link

    def raced_link(source, target):
        assert json.loads(Path(source).read_text())["complete"] is True
        Path(target).write_text("winner")
        link(source, target)

    monkeypatch.setattr(gate.os, "link", raced_link)
    with pytest.raises(FileExistsError):
        gate.publish({"complete": True}, output)
    assert output.read_text() == "winner"
    assert not list(tmp_path.glob(".summary-*.tmp"))


def test_atomic_publish_failure_cleans_temp(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("no hard-link support")
    monkeypatch.setattr(gate.os, "link", fail)
    output = tmp_path / "summary.json"
    with pytest.raises(OSError):
        gate.publish({"complete": True}, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".summary-*.tmp"))
