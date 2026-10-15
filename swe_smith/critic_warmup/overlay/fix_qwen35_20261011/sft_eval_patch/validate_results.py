#!/usr/bin/env python3
"""One-shot fail-closed gate for explicit SWE-smith shard directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


class InvalidResults(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidResults(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise InvalidResults(f"{name} must be an integer >= {minimum}")
    return value


def boolean(value, name):
    if type(value) is not bool:
        raise InvalidResults(f"{name} must be a boolean")
    return value


def validate_structured_evidence(ev):
    """Recompute reported outcomes rather than trusting completion flags."""
    for key in ("restore_rc", "collection_rc"):
        if integer(ev.get(key), key) != 0:
            raise InvalidResults(f"unsuccessful {key}")
    if ev.get("p2p_scope") != "f2p_same_files":
        raise InvalidResults("unknown P2P scope")
    streams = {}
    for key, digest_key in (("collection_events", "collection_sha256"), ("test_events", "report_sha256")):
        events = ev.get(key)
        if not isinstance(events, list) or not events or any(not isinstance(e, dict) for e in events):
            raise InvalidResults(f"missing/invalid {key}")
        encoded = "".join(json.dumps(e, ensure_ascii=True) + "\n" for e in events).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != ev.get(digest_key):
            raise InvalidResults(f"{key} hash mismatch")
        streams[key] = events
    collection = streams["collection_events"]
    if any(e.get("kind") not in ("collection", "session_finish") for e in collection):
        raise InvalidResults("collection contains errors/unknown events")
    collected = [e.get("nodes") for e in collection if e.get("kind") == "collection"]
    finishes = [e.get("exitstatus") for e in collection if e.get("kind") == "session_finish"]
    if len(collected) != 1 or finishes != [0] or type(finishes[0]) is not int:
        raise InvalidResults("collection incomplete")
    available = collected[0]
    if (not isinstance(available, list) or not available
            or any(not isinstance(n, str) or not n for n in available)
            or len(set(available)) != len(available) or ev.get("collected_nodes") != available):
        raise InvalidResults("collected nodes inconsistent")
    f2p, p2p = ev.get("f2p_statuses"), ev.get("p2p_statuses")
    if not isinstance(f2p, dict) or not isinstance(p2p, dict):
        raise InvalidResults("missing status dictionaries")
    requested = list(f2p) + list(p2p)
    digest = hashlib.sha256(json.dumps(requested, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    if digest != ev.get("requested_nodes_sha256"):
        raise InvalidResults("requested nodes hash mismatch")
    mapping = ev.get("node_mapping")
    if not isinstance(mapping, dict) or set(mapping) != set(requested):
        raise InvalidResults("mapping does not cover requested nodes")
    used = set()
    for node, full in mapping.items():
        matches = [node] if node in available else [n for n in available if n.split() and n.split()[0] == node]
        if matches != [full] or full in used:
            raise InvalidResults("mapping not exact/unique or contains aliases")
        used.add(full)
    files = {n.split("::", 1)[0] for n in f2p}
    if any(n.split("::", 1)[0] not in files for n in p2p):
        raise InvalidResults("P2P outside declared scope")
    phases = {}
    events = streams["test_events"]
    finishes = [e.get("exitstatus") for e in events if e.get("kind") == "session_finish"]
    if finishes != [ev.get("pytest_rc")] or type(finishes[0]) is not int:
        raise InvalidResults("execution finish mismatch")
    for event in events:
        kind = event.get("kind")
        if kind in ("collection", "session_finish"):
            continue
        if kind != "test":
            raise InvalidResults("execution contains errors/unknown events")
        node, when = event.get("nodeid"), event.get("when")
        if (not isinstance(node, str) or node not in used or when not in ("setup", "call", "teardown")
                or event.get("outcome") not in ("passed", "failed", "skipped")
                or type(event.get("wasxfail")) is not bool):
            raise InvalidResults("invalid test phase")
        reports = phases.setdefault(node, {})
        if when in reports:
            raise InvalidResults("duplicate test phase")
        reports[when] = event
    if set(phases) != used:
        raise InvalidResults("missing test reports")
    for requested_node, full in mapping.items():
        reports = phases[full]
        setup, call, teardown = (reports.get(k) for k in ("setup", "call", "teardown"))
        if any(r and r["outcome"] == "failed" for r in (setup, teardown)):
            status = "ERROR"
        elif call and call["outcome"] == "failed":
            status = "FAILED"
        elif not setup or not teardown or teardown["outcome"] != "passed":
            raise InvalidResults("incomplete successful test phases")
        elif setup["outcome"] == "skipped":
            status = "XFAIL" if setup["wasxfail"] else "SKIPPED"
        elif not call:
            raise InvalidResults("missing call phase")
        elif call["outcome"] == "skipped":
            status = "XFAIL" if call["wasxfail"] else "SKIPPED"
        else:
            status = "XPASS" if call["wasxfail"] else "PASSED"
        if any(requested_node in group and group[requested_node] != status for group in (f2p, p2p)):
            raise InvalidResults("status disagrees with phase evidence")
    if ev.get("resolved") and ev.get("pytest_rc") != 0:
        raise InvalidResults("resolved with nonzero exit status")


def validate_eval(row):
    if row.get("error"):
        raise InvalidResults("runner infrastructure error")
    ev = row.get("eval")
    if not isinstance(ev, dict):
        raise InvalidResults("missing eval object")
    if "grader_protocol" in ev:
        if ev["grader_protocol"] != "pytest-structured-v1":
            raise InvalidResults("unknown grader protocol")
        if ev.get("evaluation_state") != "complete" or ev.get("reward_eligible") is not True:
            raise InvalidResults("structured evaluation incomplete or reward unavailable")
        if ev.get("report_intact") is not True or ev.get("restore_rc") != 0 or ev.get("collection_rc") != 0:
            raise InvalidResults("structured evaluation evidence incomplete")
        validate_structured_evidence(ev)
    resolved = boolean(ev.get("resolved"), "eval.resolved")
    timed_out = boolean(ev.get("timed_out"), "eval.timed_out")
    boolean(row.get("timed_out"), "timed_out")
    integer(row.get("smith_rc"), "smith_rc")
    rc = integer(ev.get("pytest_rc"), "eval.pytest_rc")
    counts = {key: integer(ev.get(key), f"eval.{key}") for key in (
        "n_f2p", "n_f2p_pass", "n_p2p", "n_p2p_ok"
    )}
    if not counts["n_f2p"] or counts["n_f2p_pass"] > counts["n_f2p"] or counts["n_p2p_ok"] > counts["n_p2p"]:
        raise InvalidResults("invalid evaluation test counts")
    if integer(row.get("n_f2p"), "n_f2p", 1) != counts["n_f2p"]:
        raise InvalidResults("top-level/eval n_f2p mismatch")
    # MISSING outside a timeout cannot distinguish collection/runner faults
    # from a model failure, so it must not silently become a scored failure.
    for key, total, passed in (("f2p_statuses", counts["n_f2p"], counts["n_f2p_pass"]),
                               ("p2p_statuses", counts["n_p2p"], counts["n_p2p_ok"])):
        statuses = ev.get(key)
        if not isinstance(statuses, dict) or len(statuses) != total:
            raise InvalidResults(f"missing/inconsistent {key}")
        allowed = {"PASSED", "XFAIL", "FAILED", "ERROR", "SKIPPED", "XPASS"}
        if timed_out:
            allowed.add("MISSING")
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or v not in allowed for k, v in statuses.items()):
            raise InvalidResults(f"unavailable/unknown test statuses in {key}")
        if sum(v in ("PASSED", "XFAIL") for v in statuses.values()) != passed:
            raise InvalidResults(f"pass count disagrees with {key}")
    # eval_inside.py derives timed_out as `rc == 124 or "[timed out after" in
    # output` (see examples/swe_smith/opencode_smoke/eval_inside.py), so a test
    # that merely prints that literal marker (or any non-harness path) can set
    # timed_out=True with a non-124 pytest_rc. That is a false positive, not a
    # harness timeout, and must NOT be scored as a completed+failed result.
    # Requiring rc == 124 here keeps those rows fail-closed as "inconsistent
    # evaluation timeout" (unavailable) instead of letting an infra/marker
    # artifact through as a scored failure. Do not relax this to timed_out alone.
    if timed_out:
        if rc != 124 or resolved:
            raise InvalidResults("inconsistent evaluation timeout")
    elif rc not in (0, 1):
        raise InvalidResults(f"unavailable evaluation: pytest_rc={rc}")
    expected = not timed_out and counts["n_f2p_pass"] == counts["n_f2p"] and counts["n_p2p_ok"] == counts["n_p2p"]
    if resolved != expected:
        raise InvalidResults("resolved disagrees with evaluator test-count rule")
    return ev


def collect(ids_file, shard_dirs, checkpoint_step):
    integer(checkpoint_step, "checkpoint_step", 1)
    manifest_bytes = Path(ids_file).read_bytes()
    ids = json.loads(manifest_bytes, object_pairs_hook=unique_object)
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise InvalidResults("manifest must be a non-empty list of non-empty string IDs")
    if len(ids) != len(set(ids)):
        raise InvalidResults("duplicate manifest IDs")
    if not shard_dirs:
        raise InvalidResults("at least one explicit shard directory required")
    dirs = [Path(d).resolve(strict=True) for d in shard_dirs]
    if len(dirs) != len(set(dirs)) or any(a in b.parents for a in dirs for b in dirs if a != b):
        raise InvalidResults("duplicate or overlapping shard directories")
    rows = {}
    problems = []
    for directory in dirs:
        if not directory.is_dir():
            raise InvalidResults(f"not a shard directory: {directory}")
        for path in sorted(directory.rglob("result.json")):
            try:
                if path.parent.parent != directory or path.is_symlink() or path.parent.is_symlink():
                    raise InvalidResults("unexpected/symlink result layout; expected shard/instance/result.json")
                row = json.loads(path.read_bytes(), object_pairs_hook=unique_object)
                if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str) or not row["instance_id"]:
                    raise InvalidResults("missing valid instance_id")
                iid = row["instance_id"]
                if iid != path.parent.name:
                    raise InvalidResults("instance_id disagrees with directory name")
                if iid in rows:
                    raise InvalidResults(f"duplicate result ID {iid}")
                rows[iid] = row
                validate_eval(row)
            except (ValueError, OSError) as exc:
                problems.append(f"{path}: {exc}")
    missing = sorted(set(ids) - rows.keys())
    extra = sorted(rows.keys() - set(ids))
    if missing:
        problems.append(f"missing IDs: {missing}")
    if extra:
        problems.append(f"extra IDs: {extra}")
    if problems:
        raise InvalidResults(json.dumps({"complete": False, "n_expected": len(ids), "n_observed": len(rows), "unavailable_or_invalid": problems}, ensure_ascii=False))
    results = [rows[i] for i in ids]
    resolved = sum(r["eval"]["resolved"] for r in results)
    return {
        "complete": True,
        "checkpoint_step": checkpoint_step,
        "n_expected": len(ids),
        "n_completed": len(results),
        "n_resolved": resolved,
        "resolved_rate": resolved / len(ids),
        "n_trajectory_timeout": sum(r["timed_out"] for r in results),
        "n_eval_timeout": sum(r["eval"]["timed_out"] for r in results),
        "n_outer_timeout_rc124": sum(r["smith_rc"] == 124 for r in results),
        "ids_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "shard_dirs": [str(d) for d in dirs],
    }


def publish(summary, output):
    """Publish atomically; never replace an existing or concurrently written file."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise InvalidResults(f"refusing to overwrite summary: {output}")
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=".summary-", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # link() atomically publishes a complete file without replacing a target.
        # Unsupported filesystems fail closed; no overwrite fallback.
        os.link(temp, output)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, action="append", type=Path)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = collect(args.ids_file, args.shard_dir, args.checkpoint_step)
        publish(summary, args.output)
    except (ValueError, OSError) as exc:
        print(f"result gate rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
