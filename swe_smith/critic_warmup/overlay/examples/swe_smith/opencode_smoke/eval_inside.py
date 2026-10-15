#!/usr/bin/env python3
"""Grade a SWE-smith worktree after OpenCode. Runs inside the container.

Uses /usr/bin/git.real + the hidden git dir that was copied back after the
agent finished. The agent never sees .git during the rollout.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

TESTBED = "/testbed"
HIDDEN_GIT = os.environ.get("SMITH_HIDDEN_GIT_DIR", "/opt/agl_tmp")
GIT = os.environ.get("AGL_GIT", "/usr/bin/git.real")
INSTANCE_JSON = os.environ.get("AGL_INSTANCE_JSON", "/opt/agl_eval/instance.json")
PYTHON = os.environ.get("AGL_PYTHON", "")
if not PYTHON:
    for cand in (
        "/opt/miniconda3/envs/testbed/bin/python",
        "/opt/conda/envs/testbed/bin/python",
        "python",
    ):
        if cand == "python" or os.path.isfile(cand):
            PYTHON = cand
            break

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_STATUS_RE = re.compile(
    r"^\s*(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(.+?)\s*$|"
    r"^\s*(.+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)"
    r"(?:\s+\[\s*\d+%\])?\s*$"
)


def git_base() -> list[str]:
    return [GIT, "--git-dir", HIDDEN_GIT, "--work-tree", TESTBED, "-c", "safe.directory=*"]


def git_run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*git_base(), *args],
        cwd=TESTBED,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def restore_f2p_tests(test_nodes: list[str]) -> None:
    paths = sorted({node.split("::", 1)[0] for node in test_nodes if node})
    if not paths:
        return
    proc = git_run(["checkout", "HEAD~1", "--", *paths], timeout=120)
    if proc.returncode != 0:
        sys.stderr.write(f"restore_f2p_tests failed: {proc.stderr}\n")


def capture_patch() -> str:
    proc = git_run(["-c", "core.fileMode=false", "diff", "HEAD"], timeout=60)
    return proc.stdout or ""


def parse_test_statuses(test_output: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in _ANSI_RE.sub("", test_output).splitlines():
        m = _STATUS_RE.search(line)
        if not m:
            continue
        if m.group(1):
            status, node = m.group(1), m.group(2)
        else:
            node, status = m.group(3), m.group(4)
        # Pytest short summaries append failure details after this separator.
        # Live grading will use structured reports; this is the legacy-log path.
        node = node.split(" - ", 1)[0]
        previous = statuses.get(node)
        statuses[node] = status if previous is None or previous == status else "CONFLICT"
    return statuses


# Embedded because the container harness mounts this evaluator as a single file.
# The controller writes reports under xdist; worker writes would duplicate them.
PYTEST_REPORT_PLUGIN = '''import json
import os


def _record(event):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    with open(os.environ["AGL_PYTEST_REPORT"], "a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(event, ensure_ascii=True) + chr(10))
        stream.flush()


def pytest_collection_finish(session):
    _record({"kind": "collection", "nodes": [item.nodeid for item in session.items]})


def pytest_collectreport(report):
    if report.failed:
        _record({"kind": "collection_error", "nodeid": report.nodeid})


def pytest_runtest_logreport(report):
    _record({"kind": "test", "nodeid": report.nodeid, "when": report.when,
             "outcome": report.outcome, "wasxfail": hasattr(report, "wasxfail")})


def pytest_sessionfinish(session, exitstatus):
    _record({"kind": "session_finish", "exitstatus": int(exitstatus)})
'''


def read_pytest_events(path: str) -> tuple[list[dict[str, Any]], bool]:
    """Keep complete events before cancellation; flag any damaged/missing stream."""
    events = []
    try:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    return events, False
                if not isinstance(event, dict):
                    return events, False
                events.append(event)
    except OSError:
        return events, False
    return events, True


def collected_pytest_nodes(events: list[dict[str, Any]]) -> list[str]:
    """Accept one successful, nonempty collection, never partial collection."""
    collections = [e.get("nodes") for e in events if e.get("kind") == "collection"]
    finished = [e.get("exitstatus") for e in events if e.get("kind") == "session_finish"]
    if (len(collections) != 1 or finished != [0]
            or any(e.get("kind") == "collection_error" for e in events)):
        raise ValueError("pytest collection incomplete or failed")
    nodes = collections[0]
    if (not isinstance(nodes, list) or not nodes
            or any(not isinstance(n, str) or not n for n in nodes)
            or len(nodes) != len(set(nodes))):
        raise ValueError("invalid or duplicate collected node IDs")
    return nodes


def structured_pytest_statuses(events: list[dict[str, Any]]) -> dict[str, str]:
    """Require complete phases for success; retain real failures before a crash."""
    phases: dict[str, dict[str, dict[str, Any]]] = {}
    conflicts: set[str] = set()
    for event in events:
        if event.get("kind") != "test":
            continue
        node = event.get("nodeid")
        if not isinstance(node, str) or not node:
            raise ValueError("invalid report node ID")
        when, outcome = event.get("when"), event.get("outcome")
        if (when not in ("setup", "call", "teardown")
                or outcome not in ("passed", "failed", "skipped")
                or type(event.get("wasxfail")) is not bool):
            raise ValueError("invalid pytest phase report")
        reports = phases.setdefault(node, {})
        if when in reports:
            # Reruns/repeated selections cannot erase an earlier failure.
            conflicts.add(node)
        reports[when] = event
    statuses = {}
    for node, reports in phases.items():
        setup, call, teardown = (reports.get(p) for p in ("setup", "call", "teardown"))
        if node in conflicts:
            status = "CONFLICT"
        elif any(r and r["outcome"] == "failed" for r in (setup, teardown)):
            status = "ERROR"
        elif call and call["outcome"] == "failed":
            status = "FAILED"
        elif not setup or not teardown or teardown["outcome"] != "passed":
            status = "MISSING"
        elif setup["outcome"] == "skipped":
            status = "XFAIL" if setup["wasxfail"] else "SKIPPED"
        elif not call:
            status = "MISSING"
        elif call["outcome"] == "skipped":
            status = "XFAIL" if call["wasxfail"] else "SKIPPED"
        else:
            status = "XPASS" if call["wasxfail"] else "PASSED"
        statuses[node] = status
    return statuses


def structured_go_statuses(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate go test -json without confusing package and subtest outcomes.

    Package-less requested names must subsequently resolve to exactly one
    package. Build failures have no fabricated per-test FAILED statuses.
    """
    packages: dict[str, str] = {}
    tests: dict[str, dict[str, str]] = {}
    running: set[tuple[str, str]] = set()
    build_failures: set[str] = set()
    outcomes = {"pass": "PASSED", "fail": "FAILED", "skip": "SKIPPED"}
    for event in events:
        action, package = event.get("Action"), event.get("Package")
        # Go 1.24 may also emit BuildEvent records, which are not test reports.
        if action in ("build-output", "build-fail"):
            if action == "build-fail":
                path = event.get("ImportPath")
                if not isinstance(path, str) or not path:
                    raise ValueError("build failure missing package identity")
                build_failures.add(path)
            continue
        if action not in ("start", "run", "pause", "cont", "pass", "bench", "fail", "output", "skip"):
            raise ValueError("unknown Go test event action")
        if not isinstance(package, str) or not package:
            raise ValueError("Go test event missing package identity")
        name = event.get("Test", "")
        if not isinstance(name, str):
            raise ValueError("invalid Go test name")
        failed_build = event.get("FailedBuild")
        if failed_build:
            if action != "fail" or name or not isinstance(failed_build, str):
                raise ValueError("invalid Go build failure event")
            build_failures.add(failed_build)
        if name:
            statuses = tests.setdefault(package, {})
            identity = (package, name)
            if action == "run":
                if identity in running or name in statuses:
                    statuses[name] = "CONFLICT"
                else:
                    statuses[name] = "MISSING"
                running.add(identity)
            elif action in outcomes:
                previous = statuses.get(name)
                # Without an observed run, a terminal record alone is not
                # sufficient success evidence (e.g. a damaged/cached stream).
                status = outcomes[action]
                statuses[name] = status if identity in running and previous == "MISSING" else "CONFLICT"
                running.discard(identity)
        elif action in outcomes:
            packages[package] = outcomes[action] if package not in packages else "CONFLICT"
    return {"packages": packages, "tests": tests, "build_failures": sorted(build_failures)}


def resolve_go_test_statuses(requested: list[str], evidence: dict[str, Any]) -> dict[str, str]:
    """Do not infer a child's outcome from its parent's, or merge packages."""
    statuses = {}
    for name in requested:
        matches = [group[name] for group in evidence["tests"].values() if name in group]
        if len(matches) > 1:
            raise ValueError(f"ambiguous Go test name across packages: {name!r}")
        statuses[name] = matches[0] if matches else "MISSING"
    return statuses


def resolve_test_nodes(requested: list[str], collected: list[str]) -> dict[str, str]:
    """Resolve only exact IDs or an unambiguous observed whitespace truncation."""
    available = set(collected)
    mapping: dict[str, str] = {}
    for node in requested:
        if node in available:
            mapping[node] = node
            continue
        matches = [full for full in available if full.split() and full.split()[0] == node]
        if len(matches) != 1:
            raise ValueError(f"test node unavailable or ambiguous: {node!r}: {sorted(matches)!r}")
        mapping[node] = matches[0]
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("multiple requested test nodes resolve to the same collected node")
    return mapping


def shq(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def run_with_deadline(args: list[str], deadline: float, *, env: dict[str, str] | None = None) -> tuple[str, int]:
    """Run inside the grader's total budget, retaining output on cancellation."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return "[evaluation deadline exhausted]", 124
    # This evaluator runs in Linux containers. A separate session makes only
    # this invocation's process group eligible for timeout cancellation.
    proc = subprocess.Popen(
        args, cwd=TESTBED, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=remaining)
        return output, proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The group may have exited between the deadline and signal.
        output, _ = proc.communicate()
        return output + "\n[evaluation deadline exhausted]", 124


def collect_python_nodes(files: list[str], deadline: float, directory: str) -> dict[str, Any]:
    """Collect with repository plugins intact, without parsing terminal text."""
    plugin = Path(directory) / "agl_pytest_report.py"
    plugin.write_text(PYTEST_REPORT_PLUGIN, encoding="utf-8")
    report = Path(directory) / "collection.jsonl"
    # An evidence file must belong to exactly one invocation.
    with report.open("x", encoding="utf-8"):
        pass
    env = dict(os.environ)
    env["AGL_PYTEST_REPORT"] = str(report)
    env["PYTHONPATH"] = directory + os.pathsep + env.get("PYTHONPATH", "")
    # Keep xdist's option parser for repositories with --dist in addopts,
    # while collecting in the controller only (no distributed workers).
    _, xdist_rc = run_with_deadline([PYTHON, "-c", "import xdist"], deadline)
    serial_args = ["-p", "xdist", "-n0"] if xdist_rc == 0 else ["-p", "no:xdist"]
    output, rc = run_with_deadline(
        [PYTHON, "-m", "pytest", "--collect-only", "--color=no", "-p", "no:cacheprovider",
         *serial_args, "-p", "agl_pytest_report", "--", *files],
        deadline, env=env,
    )
    events, intact = read_pytest_events(str(report))
    evidence = {
        "collection_rc": rc, "collection_log": output, "collection_events": events,
        "collection_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "collected_nodes": [],
    }
    if rc == 0 and intact:
        evidence["collected_nodes"] = collected_pytest_nodes(events)
    return evidence


def run_bash(command: str, timeout: int) -> tuple[str, int]:
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=TESTBED,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]", 124


def evaluate(eval_meta: dict[str, Any], timeout: int) -> dict[str, Any]:
    fail_to_pass = list(eval_meta.get("FAIL_TO_PASS") or [])
    pass_to_pass = list(eval_meta.get("PASS_TO_PASS") or [])
    if not fail_to_pass:
        return {"resolved": False, "reason": "no FAIL_TO_PASS", "reward": 0.0}

    deadline = time.monotonic() + timeout
    f2p_files = sorted({t.split("::", 1)[0] for t in fail_to_pass})
    # Preserve the historical same-file scope, not a misleading path prefix.
    pass_to_pass = [t for t in pass_to_pass if t.split("::", 1)[0] in f2p_files]
    nodes = fail_to_pass + pass_to_pass
    statuses: dict[str, str] = {}
    evidence: dict[str, Any] = {"grader_protocol": "pytest-structured-v1", "p2p_scope": "f2p_same_files"}
    output, rc = "", 125
    failure = ""
    complete = False
    try:
        if len(set(fail_to_pass)) != len(fail_to_pass) or len(set(pass_to_pass)) != len(pass_to_pass):
            raise ValueError("duplicate requested test IDs")
        if any("::" not in node for node in nodes):
            raise ValueError("unsupported test IDs: non-pytest backend requires protocol verification")
        if any(Path(f).is_absolute() or ".." in Path(f).parts or f.startswith("-") for f in f2p_files):
            raise ValueError("invalid test file path")
        restore_output, rc = run_with_deadline(
            [*git_base(), "checkout", "HEAD~1", "--", *f2p_files], deadline,
        )
        evidence["restore_rc"] = rc
        evidence["restore_log"] = restore_output
        output += restore_output
        if rc != 0:
            raise ValueError("test restoration failed")
        with tempfile.TemporaryDirectory(prefix="agl-pytest-") as directory:
            collection = collect_python_nodes(f2p_files, deadline, directory)
            evidence.update(collection)
            output += collection["collection_log"]
            rc = collection["collection_rc"]
            if not collection["collected_nodes"]:
                raise ValueError("test collection unavailable")
            mapping = resolve_test_nodes(list(dict.fromkeys(nodes)), collection["collected_nodes"])
            evidence["node_mapping"] = mapping
            evidence["requested_nodes_sha256"] = hashlib.sha256(
                json.dumps(nodes, ensure_ascii=True, separators=(",", ":")).encode()
            ).hexdigest()
            report = Path(directory) / "execution.jsonl"
            with report.open("x", encoding="utf-8"):
                pass
            env = dict(os.environ)
            env["AGL_PYTEST_REPORT"] = str(report)
            env["PYTHONPATH"] = directory + os.pathsep + env.get("PYTHONPATH", "")
            probe_output, probe_rc = run_with_deadline([PYTHON, "-c", "import xdist"], deadline)
            if probe_rc == 124:
                rc = 124
                output += probe_output
                raise ValueError("evaluation deadline exhausted during xdist probe")
            xdist_args = ["-n4"] if probe_rc == 0 else ["-p", "no:xdist"]
            test_output, rc = run_with_deadline(
                [PYTHON, "-m", "pytest", "-rA", "--color=no", "-p", "no:cacheprovider",
                 "-p", "agl_pytest_report", *xdist_args, "--", *dict.fromkeys(mapping.values())],
                deadline, env=env,
            )
            output += test_output
            events, intact = read_pytest_events(str(report))
            evidence["test_events"] = events
            evidence["report_intact"] = intact
            evidence["report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
            full_statuses = structured_pytest_statuses(events)
            statuses = {node: full_statuses.get(full, "MISSING") for node, full in mapping.items()}
            finishes = [e.get("exitstatus") for e in events if e.get("kind") == "session_finish"]
            complete = (intact and finishes == [rc] and rc in (0, 1)
                        and not any(e.get("kind") == "collection_error" for e in events)
                        and set(full_statuses) == set(mapping.values())
                        and all(s not in ("MISSING", "CONFLICT") for s in statuses.values()))
            if not complete:
                failure = "execution incomplete or inconsistent"
    except (ValueError, OSError) as exc:
        failure = str(exc)
    timed_out = rc == 124
    # Unknown/infra errors remain unusable as rewards even if a caller sees 0.0.
    evidence["reward_eligible"] = complete and not timed_out
    evidence["evaluation_state"] = "complete" if complete else ("timeout" if timed_out else "unavailable")
    if failure:
        evidence["grading_error"] = failure
    f2p_pass = [t for t in fail_to_pass if statuses.get(t) in ("PASSED", "XFAIL")]
    p2p_ok = [t for t in pass_to_pass if statuses.get(t) in ("PASSED", "XFAIL")]
    resolved = complete and rc == 0 and len(f2p_pass) == len(fail_to_pass) and len(p2p_ok) == len(pass_to_pass)
    reason = (
        f"{'EVAL TIMEOUT — ' if timed_out else ''}"
        f"FAIL_TO_PASS {len(f2p_pass)}/{len(fail_to_pass)} passed, "
        f"PASS_TO_PASS {len(p2p_ok)}/{len(pass_to_pass)} ok (f2p_only)"
    )
    return {
        **evidence,
        "resolved": resolved,
        "reward": 1.0 if resolved else 0.0,
        "reason": reason,
        "timed_out": timed_out,
        "pytest_rc": rc,
        "n_f2p": len(fail_to_pass),
        "n_f2p_pass": len(f2p_pass),
        "n_p2p": len(pass_to_pass),
        "n_p2p_ok": len(p2p_ok),
        "f2p_pass_ids": f2p_pass,
        "f2p_fail_ids": [t for t in fail_to_pass if t not in f2p_pass],
        "p2p_ok_ids": p2p_ok,
        "p2p_fail_ids": [t for t in pass_to_pass if t not in p2p_ok],
        "f2p_statuses": {t: statuses.get(t, "MISSING") for t in fail_to_pass},
        "p2p_statuses": {t: statuses.get(t, "MISSING") for t in pass_to_pass},
        "pytest_tail": "\n".join(output.splitlines()[-80:]),
        "pytest_log": output[-120000:],
    }


def main() -> int:
    meta = json.loads(open(INSTANCE_JSON, encoding="utf-8").read())
    if not os.path.isdir(HIDDEN_GIT):
        print(json.dumps({"resolved": False, "reason": f"missing git dir {HIDDEN_GIT}", "reward": 0.0}))
        return 2
    patch = capture_patch()
    result = evaluate(meta, int(os.environ.get("SMITH_EVAL_TIMEOUT", "600")))
    result["patch_chars"] = len(patch)
    result["patch_head"] = patch[:4000]
    result["patch"] = patch
    out = os.environ.get("AGL_EVAL_OUT", "")
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if out:
        open(out, "w", encoding="utf-8").write(text)
    print(text)
    return 0 if result.get("resolved") else 1


if __name__ == "__main__":
    raise SystemExit(main())
