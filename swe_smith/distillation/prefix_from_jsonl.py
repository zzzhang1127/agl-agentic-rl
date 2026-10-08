#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Shortest-prefix docker scan over an assembled teacher jsonl (train split).

Flash / Qwen teacher rows already have smith-protocol ``messages``. status.json
in the sweep dirs often does not. This driver looks up ``instance.json`` for
FAIL_TO_PASS, replays prefixes, and writes a resume-safe jsonl.

Long traces (n_turns > --min-turns, default 15) are scanned first: those are
the ones a global turn cap threw away. Shorter resolved traces are kept as-is
without docker (already short; still get a terminal submit).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from examples.swe_smith.shortest_prefix import (
        assistant_actions,
        ensure_terminal_submit,
        filter_reason,
        row_from_messages,
    )
    from examples.swe_smith.shortest_prefix_replay import replay_eval_prefix
except ImportError:
    from shortest_prefix import (  # type: ignore
        assistant_actions,
        ensure_terminal_submit,
        filter_reason,
        row_from_messages,
    )
    from shortest_prefix_replay import replay_eval_prefix  # type: ignore


def _load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        iid = str(row.get("instance_id") or "")
        if iid:
            done.add(iid)
    return done


def _instance_roots(raw: str) -> list[Path]:
    return [Path(part) for part in raw.split(":") if part.strip()]


def load_instance(iid: str, roots: list[Path], row: dict[str, Any]) -> dict[str, Any] | None:
    for root in roots:
        path = root / iid / "instance.json"
        if path.is_file():
            inst = json.loads(path.read_text(encoding="utf-8"))
            if row.get("image_name") and not inst.get("image_name"):
                inst["image_name"] = row["image_name"]
            return inst
    if row.get("FAIL_TO_PASS") or row.get("fail_to_pass"):
        return {
            "instance_id": iid,
            "image_name": row.get("image_name"),
            "FAIL_TO_PASS": row.get("FAIL_TO_PASS") or row.get("fail_to_pass"),
            "PASS_TO_PASS": row.get("PASS_TO_PASS") or row.get("pass_to_pass") or [],
            "problem_statement": row.get("problem_statement") or "",
            "repo": row.get("repo"),
        }
    return None


def keep_short_as_is(row: dict[str, Any]) -> dict[str, Any]:
    messages = ensure_terminal_submit(row.get("messages") or [])
    return {
        "instance_id": row.get("instance_id"),
        "n_turns_original": row.get("n_turns"),
        "n_turns": len(assistant_actions(messages)),
        "submitted": True,
        "resolved": True,
        "messages": messages,
        "k_prefix": row.get("n_turns"),
        "short_kept": True,
    }


def process_one(row: dict[str, Any], inst: dict[str, Any]) -> dict[str, Any]:
    iid = str(row.get("instance_id") or inst.get("instance_id") or "")
    messages = row.get("messages") or []
    actions = assistant_actions(messages)
    submitted = bool(row.get("submitted", True))
    resolved = bool(row.get("resolved", True))
    reason = filter_reason(actions, submitted=submitted, resolved=resolved)
    if reason:
        return {"instance_id": iid, "drop": reason}

    cache: dict[int, bool] = {}

    def eval_prefix(k: int) -> bool:
        if k in cache:
            return cache[k]
        ok = replay_eval_prefix(inst, actions, k)
        cache[k] = bool(ok)
        return cache[k]

    return row_from_messages(iid, messages, eval_prefix=eval_prefix, submitted=submitted, resolved=resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--instance-roots", default=os.environ.get("AGL_INSTANCE_ROOTS", ""))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("AGL_PREFIX_WORKERS", "2")))
    parser.add_argument("--min-turns", type=int, default=int(os.environ.get("AGL_PREFIX_MIN_TURNS", "15")))
    parser.add_argument("--max-traces", type=int, default=int(os.environ.get("AGL_PREFIX_MAX", "0")))
    parser.add_argument("--scan-short", action="store_true", help="also docker-scan traces with n_turns <= min-turns")
    args = parser.parse_args()

    src = Path(args.jsonl)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    roots = _instance_roots(args.instance_roots)
    done = _load_done(out)
    rows: list[dict[str, Any]] = []
    n_bad = 0
    buf = ""
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            buf += line
            try:
                row = json.loads(buf)
            except json.JSONDecodeError:
                if len(buf) > 8_000_000:
                    n_bad += 1
                    buf = ""
                continue
            buf = ""
            if not isinstance(row, dict):
                n_bad += 1
                continue
            iid = str(row.get("instance_id") or "")
            if not iid or iid in done:
                continue
            if not row.get("resolved"):
                continue
            rows.append(row)
    if buf.strip():
        n_bad += 1
    print(json.dumps({"loaded": len(rows), "bad_json_rows": n_bad, "already_done": len(done)}), flush=True)
    rows.sort(key=lambda r: int(r.get("n_turns") or 0), reverse=True)

    lock = threading.Lock()
    n_keep = n_drop = n_short = 0
    t0 = time.time()

    def write_row(payload: dict[str, Any]) -> None:
        nonlocal n_keep, n_drop, n_short
        with lock:
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if payload.get("drop"):
                n_drop += 1
            else:
                n_keep += 1
                if payload.get("short_kept"):
                    n_short += 1
            if (n_keep + n_drop) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "progress": n_keep + n_drop,
                            "keep": n_keep,
                            "drop": n_drop,
                            "short_kept": n_short,
                            "sec": int(time.time() - t0),
                        }
                    ),
                    flush=True,
                )

    todo_scan: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        iid = str(row["instance_id"])
        n_turns = int(row.get("n_turns") or 0)
        messages = row.get("messages") or []
        reason = filter_reason(
            assistant_actions(messages),
            submitted=bool(row.get("submitted", True)),
            resolved=bool(row.get("resolved", True)),
        )
        if reason:
            write_row({"instance_id": iid, "drop": reason})
            continue
        if n_turns <= args.min_turns and not args.scan_short:
            write_row(keep_short_as_is(row))
            continue
        if args.max_traces > 0 and len(todo_scan) >= args.max_traces:
            continue
        inst = load_instance(iid, roots, row)
        if not inst:
            write_row({"instance_id": iid, "drop": "no_instance"})
            continue
        todo_scan.append((row, inst))

    print(json.dumps({"queued_scan": len(todo_scan), "already_written_short": n_keep, "workers": args.workers}), flush=True)
    if todo_scan:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = [pool.submit(process_one, row, inst) for row, inst in todo_scan]
            for fut in as_completed(futs):
                try:
                    write_row(fut.result())
                except Exception as exc:
                    write_row({"instance_id": "unknown", "drop": f"worker_exc:{type(exc).__name__}"})

    print(json.dumps({"keep": n_keep, "drop": n_drop, "short_kept": n_short, "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
