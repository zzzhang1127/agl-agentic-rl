#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Docker replay driver for shortest successful prefixes.

Reads a smith sweep directory (``status.json`` + ``result.json`` per instance)
or is imported by ``prefix_from_jsonl.py``. Each candidate prefix is replayed
inside a fresh SWE image, then graded with ``eval_inside.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    from examples.swe_smith.shortest_prefix import (
        assistant_actions,
        filter_reason,
        row_from_status,
    )
except ImportError:
    from shortest_prefix import (  # type: ignore
        assistant_actions,
        filter_reason,
        row_from_status,
    )

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
EVAL_TIMEOUT = int(os.environ.get("SMITH_EVAL_TIMEOUT", "600"))
CMD_TIMEOUT = int(os.environ.get("SMITH_CMD_TIMEOUT", "120"))
EVAL_PY = Path(
    os.environ.get(
        "AGL_EVAL_INSIDE",
        "/workspace/agl-checkpoints/swe_smith_smoke/eval_inside.py",
    )
)


def resolve_image(name: str) -> str | None:
    """Return a local image id; never pull (eval host already has the SWE images)."""
    if not name:
        return None
    for ref in (name, f"dockerproxy.net/{name}", f"docker.1ms.run/{name}"):
        proc = subprocess.run(
            ["docker", "image", "inspect", "-f", "{{.Id}}", ref],
            capture_output=True,
            text=True,
        )
        img_id = (proc.stdout or "").strip()
        if proc.returncode == 0 and img_id:
            return img_id
    return None


def _parse_eval_stdout(text: str) -> dict[str, Any] | None:
    blob = (text or "").strip()
    if not blob:
        return None
    try:
        payload = json.loads(blob)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = blob.find("{")
        end = blob.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(blob[start : end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None


def replay_eval_prefix(instance: dict[str, Any], actions: list[str], k: int) -> bool:
    """Replay actions[:k] (append submit if missing) and return eval.resolved."""
    prefix = list(actions[:k])
    if not any("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in item for item in prefix):
        prefix.append(SUBMIT)
    image = resolve_image(str(instance.get("image_name") or ""))
    if not image or not EVAL_PY.is_file():
        return False
    if not (instance.get("FAIL_TO_PASS") or instance.get("fail_to_pass")):
        return False

    job_dir = Path(tempfile.mkdtemp(prefix="prefix-"))
    cid = ""
    try:
        inst_path = job_dir / "instance.json"
        inst_path.write_text(json.dumps(instance, ensure_ascii=False), encoding="utf-8")
        actions_path = job_dir / "actions.json"
        actions_path.write_text(json.dumps(prefix, ensure_ascii=False), encoding="utf-8")
        runner = job_dir / "replay.py"
        runner.write_text(
            "import json, subprocess, sys\n"
            "actions = json.load(open('/tmp/prefix_actions.json', encoding='utf-8'))\n"
            f"timeout = {CMD_TIMEOUT}\n"
            "for action in actions:\n"
            "    try:\n"
            "        subprocess.run(['bash', '-lc', action], cwd='/testbed', timeout=timeout)\n"
            "    except subprocess.TimeoutExpired:\n"
            "        pass\n"
            "    except Exception:\n"
            "        pass\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--network",
                "none",
                "-e",
                "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "-v",
                f"{inst_path}:/opt/agl_eval/instance.json:ro",
                "-v",
                f"{EVAL_PY}:/opt/agl_eval/eval_inside.py:ro",
                "-w",
                "/testbed",
                image,
                "sleep",
                "infinity",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return False
        cid = proc.stdout.strip()
        subprocess.run(["docker", "cp", str(runner), f"{cid}:/tmp/prefix_replay.py"], check=False)
        subprocess.run(["docker", "cp", str(actions_path), f"{cid}:/tmp/prefix_actions.json"], check=False)
        replay_timeout = min(3600, CMD_TIMEOUT * max(len(prefix), 1) + 60)
        subprocess.run(
            ["docker", "exec", "-w", "/testbed", cid, "python", "/tmp/prefix_replay.py"],
            capture_output=True,
            timeout=replay_timeout,
        )
        subprocess.run(
            [
                "docker",
                "exec",
                cid,
                "bash",
                "-lc",
                "if [ ! -d /opt/agl_tmp/HEAD ] && [ -d /testbed/.git ]; then "
                "rm -rf /opt/agl_tmp; mv /testbed/.git /opt/agl_tmp; fi",
            ],
            capture_output=True,
            timeout=60,
        )
        eval_proc = subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                "SMITH_HIDDEN_GIT_DIR=/opt/agl_tmp",
                "-e",
                "AGL_GIT=/usr/bin/git",
                "-e",
                f"SMITH_EVAL_TIMEOUT={EVAL_TIMEOUT}",
                "-e",
                "AGL_INSTANCE_JSON=/opt/agl_eval/instance.json",
                cid,
                "python",
                "/opt/agl_eval/eval_inside.py",
            ],
            capture_output=True,
            text=True,
            timeout=EVAL_TIMEOUT + 60,
        )
        payload = _parse_eval_stdout(eval_proc.stdout or "")
        if payload is None:
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
            inst_path = result_path.parent / "instance.json"
            if not status_path.is_file():
                n_drop += 1
                handle.write(json.dumps({"instance_id": result_path.parent.name, "drop": "no_status"}) + "\n")
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            instance = {}
            if inst_path.is_file():
                instance = json.loads(inst_path.read_text(encoding="utf-8"))
            instance.setdefault("image_name", result.get("image_name") or status.get("image_name"))
            instance.setdefault("instance_id", result.get("instance_id"))
            actions = assistant_actions(status.get("messages") or [])
            submitted = bool(status.get("submitted") or result.get("submitted"))
            resolved = bool(result.get("resolved") or (result.get("eval") or {}).get("resolved"))
            reason = filter_reason(actions, submitted=submitted, resolved=resolved)
            if reason:
                n_drop += 1
                handle.write(json.dumps({"instance_id": result.get("instance_id"), "drop": reason}) + "\n")
                continue

            def eval_prefix(k: int, inst: dict[str, Any] = instance, acts: list[str] = actions) -> bool:
                return replay_eval_prefix(inst, acts, k)

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
