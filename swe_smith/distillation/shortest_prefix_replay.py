#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Docker replay driver for shortest successful prefixes.

Reads a smith sweep directory (``status.json`` + ``result.json`` per instance),
replays candidate prefixes inside a fresh SWE image, and writes a jsonl of
kept (or dropped) rows. Requires the host docker + images already used by
``run_smith_sweep.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from shortest_prefix import (  # noqa: E402
    assistant_actions,
    filter_reason,
    row_from_status,
)


def ensure_run_image(name: str) -> str:
    """Images are pre-pulled on the eval host; return the name unchanged."""
    return name

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def _replay_eval(instance: dict[str, Any], actions: list[str], k: int) -> bool:
    prefix = list(actions[:k])
    if not any("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in item for item in prefix):
        prefix.append(SUBMIT)
    image = ensure_run_image(str(instance.get("image_name") or ""))
    job_dir = Path(tempfile.mkdtemp(prefix="prefix-"))
    cid = ""
    try:
        proc = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--network",
                "none",
                "-w",
                "/testbed",
                image,
                "sleep",
                "infinity",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return False
        cid = proc.stdout.strip()
        script = "set -e\ncd /testbed\n"
        for action in prefix:
            script += action + "\n"
        subprocess.run(["docker", "exec", "-w", "/testbed", cid, "bash", "-lc", script], check=False, timeout=300)
        eval_proc = subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                "SMITH_HIDDEN_GIT_DIR=/opt/agl_tmp",
                "-e",
                "AGL_GIT=/usr/bin/git",
                "-e",
                f"SMITH_EVAL_TIMEOUT={os.environ.get('SMITH_EVAL_TIMEOUT', '600')}",
                "-e",
                "AGL_INSTANCE_JSON=/opt/agl_eval/instance.json",
                cid,
                "python",
                "/opt/agl/eval_inside.py",
            ],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("SMITH_EVAL_TIMEOUT", "600")) + 30,
        )
        try:
            payload = json.loads(eval_proc.stdout.strip().splitlines()[-1])
        except Exception:
            return False
        return bool(payload.get("resolved"))
    except Exception:
        return False
    finally:
        if cid:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        shutil.rmtree(job_dir, ignore_errors=True)


def main() -> int:
    root = Path(os.environ.get("AGL_SWEEP_ROOT", "."))
    out = Path(os.environ.get("AGL_PREFIX_JSONL", str(root / "shortest_prefix.jsonl")))
    n_keep = n_drop = 0
    with out.open("w", encoding="utf-8") as handle:
        for result_path in sorted(root.glob("*/result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            status_path = result_path.parent / "status.json"
            if not status_path.is_file():
                n_drop += 1
                handle.write(json.dumps({"instance_id": result_path.parent.name, "drop": "no_status"}) + "\n")
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            actions = assistant_actions(status.get("messages") or [])
            submitted = bool(status.get("submitted") or result.get("submitted"))
            resolved = bool(result.get("resolved") or (result.get("eval") or {}).get("resolved"))
            reason = filter_reason(actions, submitted=submitted, resolved=resolved)
            if reason:
                n_drop += 1
                handle.write(json.dumps({"instance_id": result.get("instance_id"), "drop": reason}) + "\n")
                continue

            instance = {
                "image_name": result.get("image_name") or status.get("image_name"),
                "instance_id": result.get("instance_id"),
            }

            def eval_prefix(k: int, inst: dict[str, Any] = instance, acts: list[str] = actions) -> bool:
                return _replay_eval(inst, acts, k)

            row = row_from_status(status_path, result, eval_prefix=eval_prefix)
            if not row or row.get("drop"):
                n_drop += 1
                handle.write(json.dumps(row or {"instance_id": result.get("instance_id"), "drop": "replay_drift"}) + "\n")
                continue
            n_keep += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"keep": n_keep, "drop": n_drop, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
