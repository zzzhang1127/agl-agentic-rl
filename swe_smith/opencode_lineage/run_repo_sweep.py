#!/usr/bin/env python3
"""Run every SWE-smith instance from one repo through OpenCode, 2-wide.

A worker starts the next problem as soon as its current OpenCode process exits.
vLLM stays on one GPU; concurrency is capped so KV cache does not OOM.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import run_opencode_smoke as smoke

EXAMPLE = Path("/workspace/projects/agent-lightning/examples/swe_smith")
SPLIT = os.environ.get("AGL_SWEEP_SPLIT", "val").strip().lower()
_DATASETS = {
    "val": [EXAMPLE / "val_dataset_filtered.jsonl"],
    "train": [EXAMPLE / "train_dataset_mixed.jsonl"],
    "all": [
        EXAMPLE / "val_dataset_filtered.jsonl",
        EXAMPLE / "train_dataset_mixed.jsonl",
    ],
}
DATASETS = _DATASETS.get(SPLIT, _DATASETS["val"])
SWEEP_ROOT = Path(os.environ.get("AGL_SWEEP_ROOT", str(smoke.ROOT / "val_baseline")))
SHARD_COUNT = max(1, int(os.environ.get("AGL_SWEEP_SHARD_COUNT", "1")))
SHARD_INDEX = int(os.environ.get("AGL_SWEEP_SHARD_INDEX", "0"))
PROGRESS = SWEEP_ROOT / (
    f"progress.shard{SHARD_INDEX}.jsonl" if SHARD_COUNT > 1 else "progress.jsonl"
)
SUMMARY = SWEEP_ROOT / "summary.json"
BASELINE = SWEEP_ROOT / "baseline.json"
FOCUS_REPO = os.environ.get("AGL_SWEEP_REPO", "ALL").strip()
WORKERS = max(1, int(os.environ.get("AGL_SWEEP_WORKERS", "3")))
LIMIT = os.environ.get("AGL_SWEEP_LIMIT", "").strip()
IDS_FILE = os.environ.get("AGL_SWEEP_IDS_FILE", "").strip()
RESUME = os.environ.get("AGL_SWEEP_RESUME", "1") != "0"
SAMPLE_TIMEOUT = int(os.environ.get("AGL_SAMPLE_TIMEOUT", "600"))
EVAL_TIMEOUT = int(os.environ.get("SMITH_EVAL_TIMEOUT", "600"))
OPENCODE_CFG = Path(os.environ.get("AGL_OPENCODE_JSON", str(smoke.OPENCODE_JSON)))
SWEEP_MODEL = os.environ.get("AGL_SWEEP_MODEL", "agl/Qwen3-8B")

_print_lock = threading.Lock()
_progress_lock = threading.Lock()
_vllm_lock = threading.Lock()
_image_lock = threading.Lock()
_run_image_cache: dict[str, str] = {}


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def f2p_list(row: dict[str, Any]) -> list[str]:
    v = row.get("FAIL_TO_PASS") or []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            v = [v]
    return list(v) if isinstance(v, list) else []


def load_repo_instances() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in DATASETS:
        if not path.exists():
            raise SystemExit(f"missing dataset {path}")
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            repo = str(row.get("repo") or "")
            iid = str(row.get("instance_id") or "")
            if not iid or iid in seen:
                continue
            if FOCUS_REPO not in ("", "ALL", "*") and repo != FOCUS_REPO:
                continue
            seen.add(iid)
            row["_split"] = path.name
            row["_n_f2p"] = len(f2p_list(row))
            found.append(row)
    found.sort(key=lambda r: (str(r.get("image_name") or ""), r["_n_f2p"], r["instance_id"]))
    if IDS_FILE:
        # Restrict to an explicit instance_id list (JSON array), e.g. the first 100 val
        # problems for the every-20-steps collapse probe (2026-09-17).
        wanted = set(json.load(open(IDS_FILE)))
        found = [r for r in found if r["instance_id"] in wanted]
        missing = wanted - {r["instance_id"] for r in found}
        if missing:
            raise SystemExit(f"{len(missing)} ids from {IDS_FILE} not in datasets, e.g. {sorted(missing)[:3]}")
    if LIMIT:
        found = found[: int(LIMIT)]
    if SHARD_COUNT > 1:
        found = [row for i, row in enumerate(found) if i % SHARD_COUNT == SHARD_INDEX]
    return found


STEP_BUDGET = int(os.environ.get("AGL_STEP_BUDGET", "0") or 0)   # 2026-09-21: 0 = no budget text (all runs before)


def prompt_text(instance: dict[str, Any]) -> str:
    problem = instance.get("problem_statement") or ""
    budget = ""
    if STEP_BUDGET > 0:
        budget = (f"You have a budget of at most {STEP_BUDGET} steps (one step = one reply of yours, with or without tool calls). "
                  "Work efficiently. You will be told when the budget is running low; when that happens, finish your edit and "
                  "stop with a short summary instead of exploring further.\n\n")
    return f"""You are a software engineer. Fix the bug described below in the repository at /testbed.

Work only on source files under /testbed. Do not modify tests, conftest.py, pytest.ini, tox.ini, setup.cfg, or pyproject.toml.

git is disabled and the .git directory is not present. Do not try to recover history, check out other commits, or read git metadata.

The network is disabled except the local model API. Do not use curl, wget, pip install, or fetch code from the internet. Everything you need is already in /testbed.

{budget}Make a general fix consistent with the codebase. When you believe the bug is fixed, stop and summarize the source files you changed.

<pr_description>
{problem}
</pr_description>
"""


def ensure_run_image(image: str) -> str:
    with _image_lock:
        cached = _run_image_cache.get(image)
        if cached:
            return cached
        smoke.pull_image(image)
        rid = smoke.resolve_run_image(image)
        _run_image_cache[image] = rid
        return rid


def job_dir_for(instance_id: str) -> Path:
    return SWEEP_ROOT / instance_id


def result_path(instance_id: str) -> Path:
    return job_dir_for(instance_id) / "result.json"


def append_progress(row: dict[str, Any]) -> None:
    with _progress_lock:
        with PROGRESS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_opencode_trace(jsonl_path: Path) -> dict[str, Any]:
    """Extract tools, token usage, compaction, and overflow from OpenCode jsonl."""
    tools: dict[str, int] = {}
    event_types: dict[str, int] = {}
    prompt_tokens = 0
    completion_tokens = 0
    compact_events = 0
    overflow_events = 0
    overflow_messages: list[str] = []
    errors: list[str] = []
    if not jsonl_path.exists():
        return {
            "tools": tools,
            "event_types": event_types,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "compacted": False,
            "compact_events": 0,
            "overflowed": False,
            "overflow_events": 0,
            "overflow_messages": [],
            "errors": [],
        }
    for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        typ = str(obj.get("type") or "")
        event_types[typ] = event_types.get(typ, 0) + 1
        blob = json.dumps(obj, ensure_ascii=False)
        low = blob.lower()
        # 2026-09-20: only count real compaction events (by event type). The old substring test on the
        # whole event blob counted repo content such as funcy's `compact()` / pyasn1 `supportCompactZero`
        # as compactions (38 false positives in batch1's "235 compacted"). NOTE: opencode's `run --format
        # json` stream does not emit a compaction event at all; the proxy's <task>.compactK.json files
        # are the ground truth for compaction (see teacher_proxy.py / assemble_teacher_traj.py).
        if "compact" in typ.lower():
            compact_events += 1
        if "contextoverflow" in low or "maximum context length" in low or "context overflow" in low:
            overflow_events += 1
            overflow_messages.append(blob[:400])
        if typ == "error":
            errors.append(blob[:400])
        if typ == "tool_use":
            part = obj.get("part") or {}
            st = part.get("state") or {}
            key = f"{part.get('tool')}:{st.get('status')}"
            tools[key] = tools.get(key, 0) + 1
        for key, bucket in (
            ("prompt_tokens", "prompt"),
            ("input_tokens", "prompt"),
            ("completion_tokens", "completion"),
            ("output_tokens", "completion"),
        ):
            for m in re.finditer(rf'"{key}"\s*:\s*(\d+)', blob):
                val = int(m.group(1))
                if bucket == "prompt":
                    prompt_tokens += val
                else:
                    completion_tokens += val
    # 2026-09-21 fallback when the proxy log is unavailable: a compaction shows up in the stream as a
    # step whose input context collapses (>20k tokens, then <50% of it on the next step).
    ctx_drops = 0
    prev_in = 0
    for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"step_finish"' not in line:
            continue
        try:
            tok = ((json.loads(line).get("part") or {}).get("tokens") or {})
        except json.JSONDecodeError:
            continue
        cur = int(tok.get("input") or 0) + int((tok.get("cache") or {}).get("read") or 0)
        if prev_in > 20000 and 0 < cur < 0.5 * prev_in:
            ctx_drops += 1
        if cur:
            prev_in = cur
    return {
        "tools": tools,
        "event_types": event_types,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "compacted": compact_events > 0,
        "compact_events": compact_events,
        "ctx_drops": ctx_drops,
        "overflowed": overflow_events > 0,
        "overflow_events": overflow_events,
        "overflow_messages": overflow_messages[:5],
        "errors": errors[:5],
    }


COMPACT_DIR = Path(os.environ.get("TRUNC_COMPACT_DIR", str(Path(__file__).resolve().parent / "compact_events")))


def container_ip(cid: str) -> str:
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}", cid],
            text=True, capture_output=True, timeout=30,
        ).stdout.split()
        return out[0] if out else ""
    except Exception:
        return ""


def proxy_compactions(ip: str, t0: float, t1: float) -> int | None:
    """Compaction requests the trunc proxy logged for this container (ground truth); None = no log."""
    # only proxies that record compactions (tool_trunc_proxy.py) create <port>.jsonl; teacher sweeps go
    # through teacher_proxy.py (own compactK.json files) and fall back to the ctx-drop count here.
    port = os.environ.get("AGL_VLLM_URL", "").rstrip("/").rsplit(":", 1)[-1]
    f = COMPACT_DIR / f"{port}.jsonl"
    if not ip or not f.is_file():
        return None
    n = 0
    if True:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ip") == ip and t0 - 1 <= float(row.get("ts", 0)) <= t1 + 1:
                n += 1
    return n


def run_one_instance(instance: dict[str, Any], run_image: str | None = None) -> dict[str, Any]:
    iid = str(instance["instance_id"])
    run_image = run_image or ensure_run_image(str(instance.get("image_name")))
    short = iid.replace(".", "-")[-48:]
    name = f"agl-sw-{os.environ.get('AGL_SWEEP_TAG', '')}{SHARD_INDEX}-{short}"  # AGL_SWEEP_TAG: avoid name clashes between concurrent sweeps
    job_dir = job_dir_for(iid)
    if job_dir.exists() and not RESUME:
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    inst_path = job_dir / "instance.json"
    prompt_path = job_dir / "prompt.txt"
    inst_path.write_text(json.dumps(instance, ensure_ascii=False), encoding="utf-8")
    prompt_path.write_text(prompt_text(instance), encoding="utf-8")
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    log(f"start {iid} f2p={instance['_n_f2p']} split={instance['_split']}")
    hidden_host = smoke.ROOT / f"hidden.git.{name}"
    try:
        run_args = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--hostname",
            name,
            "--dns",
            "127.0.0.1",
            "--add-host",
            "host.docker.internal:host-gateway",
            "--add-host",
            "github.com:127.0.0.1",
            "--add-host",
            "pypi.org:127.0.0.1",
            "--add-host",
            "files.pythonhosted.org:127.0.0.1",
            "--add-host",
            "huggingface.co:127.0.0.1",
            "--add-host",
            "pypi.python.org:127.0.0.1",
            "--cap-add",
            "NET_ADMIN",
            "-e",
            "PATH=/opt/agl_deny/bin:/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "-e",
            "HOME=/tmp/oc-home",
            "-e",
            "XDG_CONFIG_HOME=/tmp/oc-home/.config",
            "-e",
            "XDG_DATA_HOME=/tmp/oc-home/.local/share",
            "-e",
            "OPENCODE_CONFIG=/tmp/oc-home/.config/opencode/opencode.json",
            "-e",
            "OPENAI_API_KEY=dummy",
            "-v",
            f"{smoke.OPENCODE_BIN}:/usr/local/bin/opencode:ro",
            "-v",
            f"{smoke.RG_BIN}:/usr/local/bin/rg:ro",
            "-v",
            f"{OPENCODE_CFG}:/tmp/oc-home/.config/opencode/opencode.json:ro",
            "-v",
            f"{prompt_path}:/opt/agl_eval/prompt.txt:ro",
            "-v",
            f"{inst_path}:/opt/agl_eval/instance.json:ro",
            "-v",
            f"{smoke.EVAL_PY}:/opt/agl_eval/eval_inside.py:ro",
            "-w",
            "/testbed",
            run_image,
            "sleep",
            "infinity",
        ]
        proc = subprocess.run(run_args, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {proc.stderr}")
        cid = proc.stdout.strip()
        c_ip = container_ip(cid)
        c_t0 = time.time()
        hidden_host = smoke.ROOT / f"hidden.git.{cid[:12]}"
        smoke.docker_exec(
            cid,
            ["bash", "-lc", "mkdir -p /tmp/oc-home/.config/opencode /tmp/oc-home/.local/share"],
            timeout=30,
        )
        smoke.prepare_container(cid, iid)
        oc_log = job_dir / "opencode.jsonl"
        oc_err = job_dir / "opencode.stderr"
        runner = job_dir / "inside.sh"
        runner.write_text(
            "#!/bin/bash\n"
            "set -u\n"
            "export PATH=/opt/agl_deny/bin:/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
            "export HOME=/tmp/oc-home\n"
            "export XDG_CONFIG_HOME=/tmp/oc-home/.config\n"
            "export XDG_DATA_HOME=/tmp/oc-home/.local/share\n"
            "export OPENCODE_CONFIG=/tmp/oc-home/.config/opencode/opencode.json\n"
            f"timeout {SAMPLE_TIMEOUT} /usr/local/bin/opencode run --pure --auto --format json "
            f"--dir /testbed -m {SWEEP_MODEL} --title sweep "
            '"$(cat /opt/agl_eval/prompt.txt)"\n',
            encoding="utf-8",
        )
        subprocess.check_call(["docker", "cp", str(runner), f"{cid}:/tmp/inside.sh"])
        smoke.docker_exec(cid, ["bash", "-lc", "chmod +x /tmp/inside.sh"], timeout=10)
        t0 = time.time()
        metrics_before = smoke.vllm_metrics()
        gpu_before = smoke.nvidia_sample()
        with oc_log.open("w", encoding="utf-8") as out, oc_err.open("w", encoding="utf-8") as err:
            oc_proc = subprocess.run(
                ["docker", "exec", "-w", "/testbed", cid, "bash", "/tmp/inside.sh"],
                stdout=out,
                stderr=err,
                text=True,
            )
        metrics_after = smoke.vllm_metrics()
        gpu_after = smoke.nvidia_sample()
        wall = time.time() - t0
        smoke.docker_exec(cid, ["bash", "-lc", "rm -rf /opt/agl_tmp && mkdir -p /opt/agl_tmp"], timeout=30)
        subprocess.check_call(["docker", "cp", f"{hidden_host}/.", f"{cid}:/opt/agl_tmp/"])
        eval_proc = subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                "SMITH_HIDDEN_GIT_DIR=/opt/agl_tmp",
                "-e",
                "AGL_GIT=/usr/bin/git.real",
                "-e",
                f"SMITH_EVAL_TIMEOUT={EVAL_TIMEOUT}",
                "-e",
                "AGL_INSTANCE_JSON=/opt/agl_eval/instance.json",
                cid,
                "python",
                "/opt/agl_eval/eval_inside.py",
            ],
            text=True,
            capture_output=True,
            timeout=EVAL_TIMEOUT + 60,
        )
        (job_dir / "eval.json").write_text(eval_proc.stdout or eval_proc.stderr, encoding="utf-8")
        try:
            eval_json = json.loads(eval_proc.stdout)
        except Exception:
            eval_json = {
                "resolved": False,
                "reason": "eval parse failed",
                "stdout": (eval_proc.stdout or "")[-2000:],
                "stderr": (eval_proc.stderr or "")[-2000:],
            }
        pytest_log = eval_json.get("pytest_log") or eval_json.get("pytest_tail") or ""
        if pytest_log:
            (job_dir / "pytest.log").write_text(str(pytest_log), encoding="utf-8")
        patch = eval_json.get("patch") or eval_json.get("patch_head") or ""
        if patch:
            (job_dir / "patch.diff").write_text(str(patch), encoding="utf-8")
        trace = parse_opencode_trace(oc_log)
        n_proxy = proxy_compactions(c_ip, c_t0, time.time())
        trace["compact_source"] = "proxy" if n_proxy is not None else "ctx_drop"
        trace["compact_events"] = n_proxy if n_proxy is not None else trace["ctx_drops"]
        trace["compacted"] = trace["compact_events"] > 0
        (job_dir / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        prompt_delta = int(metrics_after.get("prompt_tokens_total", 0) - metrics_before.get("prompt_tokens_total", 0))
        gen_delta = int(
            metrics_after.get("generation_tokens_total", 0) - metrics_before.get("generation_tokens_total", 0)
        )
        result = {
            "instance_id": iid,
            "split": instance["_split"],
            "repo": instance.get("repo"),
            "n_f2p": instance["_n_f2p"],
            "image_name": instance.get("image_name"),
            "container": name,
            "opencode_rc": oc_proc.returncode,
            "wall_seconds": round(wall, 1),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "vllm_prompt_tokens": prompt_delta,
            "vllm_completion_tokens": gen_delta,
            "opencode_prompt_tokens": trace["prompt_tokens"],
            "opencode_completion_tokens": trace["completion_tokens"],
            "tools": trace["tools"],
            "event_types": trace["event_types"],
            "compacted": trace["compacted"],
            "compact_events": trace["compact_events"],
            "compact_source": trace["compact_source"],
            "ctx_drops": trace["ctx_drops"],
            "container_ip": c_ip,
            "overflowed": trace["overflowed"],
            "overflow_events": trace["overflow_events"],
            "overflow_messages": trace["overflow_messages"],
            "opencode_errors": trace["errors"],
            "eval": eval_json,
        }
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        result = {
            "instance_id": iid,
            "split": instance["_split"],
            "n_f2p": instance["_n_f2p"],
            "error": str(exc)[:2000],
            "eval": {"resolved": False, "reason": f"runner error: {exc}", "n_f2p_pass": 0},
        }
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        if hidden_host.exists():
            shutil.rmtree(hidden_host, ignore_errors=True)
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    ev = result.get("eval") or {}
    log(
        f"done {iid} wall={result.get('wall_seconds')} resolved={ev.get('resolved')} "
        f"f2p={ev.get('n_f2p_pass')}/{ev.get('n_f2p')} gen={result.get('vllm_completion_tokens')} "
        f"compacted={result.get('compacted')} overflowed={result.get('overflowed')} {ev.get('reason')}"
    )
    append_progress(
        {
            "instance_id": iid,
            "resolved": ev.get("resolved"),
            "reward": ev.get("reward"),
            "n_f2p_pass": ev.get("n_f2p_pass"),
            "n_f2p": ev.get("n_f2p"),
            "n_p2p_ok": ev.get("n_p2p_ok"),
            "n_p2p": ev.get("n_p2p"),
            "f2p_fail_ids": ev.get("f2p_fail_ids"),
            "p2p_fail_ids": ev.get("p2p_fail_ids"),
            "patch_chars": ev.get("patch_chars"),
            "wall_seconds": result.get("wall_seconds"),
            "opencode_rc": result.get("opencode_rc"),
            "vllm_prompt_tokens": result.get("vllm_prompt_tokens"),
            "vllm_completion_tokens": result.get("vllm_completion_tokens"),
            "opencode_completion_tokens": result.get("opencode_completion_tokens"),
            "tools": result.get("tools"),
            "compacted": result.get("compacted"),
            "compact_events": result.get("compact_events"),
            "overflowed": result.get("overflowed"),
            "overflow_events": result.get("overflow_events"),
            "reason": ev.get("reason"),
        }
    )
    return result


def summarize(results: list[dict[str, Any]], t0: float, gpu_idle: dict[str, Any]) -> dict[str, Any]:
    resolved = [r for r in results if (r.get("eval") or {}).get("resolved")]
    any_f2p = [r for r in results if int((r.get("eval") or {}).get("n_f2p_pass") or 0) > 0]
    f2p_pass = sum(int((r.get("eval") or {}).get("n_f2p_pass") or 0) for r in results)
    f2p_total = sum(int((r.get("eval") or {}).get("n_f2p") or r.get("n_f2p") or 0) for r in results)
    p2p_ok = sum(int((r.get("eval") or {}).get("n_p2p_ok") or 0) for r in results)
    p2p_total = sum(int((r.get("eval") or {}).get("n_p2p") or 0) for r in results)
    rows = []
    for r in results:
        ev = r.get("eval") or {}
        rows.append(
            {
                "instance_id": r.get("instance_id"),
                "split": r.get("split"),
                "resolved": bool(ev.get("resolved")),
                "reward": ev.get("reward"),
                "n_f2p": ev.get("n_f2p", r.get("n_f2p")),
                "n_f2p_pass": ev.get("n_f2p_pass"),
                "n_p2p": ev.get("n_p2p"),
                "n_p2p_ok": ev.get("n_p2p_ok"),
                "f2p_fail_ids": ev.get("f2p_fail_ids"),
                "p2p_fail_ids": ev.get("p2p_fail_ids"),
                "patch_chars": ev.get("patch_chars"),
                "wall_seconds": r.get("wall_seconds"),
                "opencode_rc": r.get("opencode_rc"),
                "vllm_prompt_tokens": r.get("vllm_prompt_tokens"),
                "vllm_completion_tokens": r.get("vllm_completion_tokens"),
                "opencode_prompt_tokens": r.get("opencode_prompt_tokens"),
                "opencode_completion_tokens": r.get("opencode_completion_tokens"),
                "tools": r.get("tools"),
                "event_types": r.get("event_types"),
                "compacted": r.get("compacted"),
                "compact_events": r.get("compact_events"),
                "overflowed": r.get("overflowed"),
                "overflow_events": r.get("overflow_events"),
                "reason": ev.get("reason"),
                "error": r.get("error"),
            }
        )
    n = max(len(results), 1)
    n_compacted = sum(1 for r in results if r.get("compacted"))
    n_overflowed = sum(1 for r in results if r.get("overflowed"))
    gen_tokens = sum(int(r.get("vllm_completion_tokens") or 0) for r in results)
    prompt_tokens = sum(int(r.get("vllm_prompt_tokens") or 0) for r in results)
    summary = {
        "repo": FOCUS_REPO,
        "split": SPLIT,
        "harness": "opencode",
        "model": "Qwen3-8B",
        "max_model_len": 40960,
        "workers": WORKERS,
        "n_instances": len(results),
        "n_resolved": len(resolved),
        "resolved_rate": round(len(resolved) / n, 4),
        "n_any_f2p_pass": len(any_f2p),
        "f2p_pass": f2p_pass,
        "f2p_total": f2p_total,
        "f2p_pass_rate": round(f2p_pass / f2p_total, 4) if f2p_total else 0.0,
        "p2p_ok": p2p_ok,
        "p2p_total": p2p_total,
        "p2p_ok_rate": round(p2p_ok / p2p_total, 4) if p2p_total else 0.0,
        "n_compacted": n_compacted,
        "n_overflowed": n_overflowed,
        "vllm_prompt_tokens_sum": prompt_tokens,
        "vllm_completion_tokens_sum": gen_tokens,
        "n_repos": len({r.get("repo") for r in results}),
        "total_wall_seconds": round(time.time() - t0, 1),
        "gpu_with_vllm": gpu_idle,
        "resolved_ids": [r["instance_id"] for r in resolved],
        "any_f2p_ids": [r["instance_id"] for r in any_f2p],
        "results": rows,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    BASELINE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = SWEEP_ROOT / "baseline.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "instance_id,resolved,reward,f2p_pass,f2p_total,p2p_ok,p2p_total,patch_chars,"
            "wall_seconds,gen_tokens,prompt_tokens,compacted,overflowed,opencode_rc,reason\n"
        )
        for row in rows:
            reason = str(row.get("reason") or "").replace(",", ";").replace("\n", " ")
            fh.write(
                f"{row.get('instance_id')},{int(bool(row.get('resolved')))},{row.get('reward')},"
                f"{row.get('n_f2p_pass')},{row.get('n_f2p')},{row.get('n_p2p_ok')},{row.get('n_p2p')},"
                f"{row.get('patch_chars')},{row.get('wall_seconds')},{row.get('vllm_completion_tokens')},"
                f"{row.get('vllm_prompt_tokens')},{int(bool(row.get('compacted')))},{int(bool(row.get('overflowed')))},"
                f"{row.get('opencode_rc')},{reason}\n"
            )
    return summary


def main() -> int:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    instances = load_repo_instances()
    if not instances:
        raise SystemExit(f"no instances for repo {FOCUS_REPO} split {SPLIT}")
    images = sorted({str(r.get("image_name")) for r in instances})
    log(f"repo={FOCUS_REPO} split={SPLIT} shard={SHARD_INDEX}/{SHARD_COUNT} n={len(instances)} images={len(images)} workers={WORKERS} opencode={OPENCODE_CFG} vllm={smoke.VLLM_URL}")
    index_path = SWEEP_ROOT / (f"index.shard{SHARD_INDEX}.json" if SHARD_COUNT > 1 else "index.json")
    index_path.write_text(
        json.dumps(
            [
                {
                    "instance_id": r["instance_id"],
                    "split": r["_split"],
                    "repo": r.get("repo"),
                    "image_name": r.get("image_name"),
                    "n_f2p": r["_n_f2p"],
                }
                for r in instances
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    if not smoke.RG_BIN.is_file():
        raise SystemExit(f"ripgrep missing: {smoke.RG_BIN}")
    smoke.wait_vllm(timeout=900)
    gpu_idle = smoke.nvidia_sample()
    log(f"gpu {gpu_idle}")
    pending: list[dict[str, Any]] = []
    done: list[dict[str, Any]] = []
    for row in instances:
        iid = str(row["instance_id"])
        prev = result_path(iid)
        if RESUME and prev.exists():
            done.append(json.loads(prev.read_text(encoding="utf-8")))
            log(f"resume skip {iid}")
        else:
            pending.append(row)
    t0 = time.time()
    if pending:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = [pool.submit(run_one_instance, row) for row in pending]
            for fut in as_completed(futs):
                done.append(fut.result())
    # Keep dataset order in the summary.
    by_id = {r.get("instance_id"): r for r in done}
    ordered = [by_id[r["instance_id"]] for r in instances if r["instance_id"] in by_id]
    all_done: list[dict[str, Any]] = []
    for path in SWEEP_ROOT.glob("*/result.json"):
        try:
            all_done.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    summary = summarize(all_done or ordered, t0, gpu_idle)
    log(f"wrote {SUMMARY}")
    print(
        json.dumps(
            {
                "repo": FOCUS_REPO,
                "split": SPLIT,
                "n": summary["n_instances"],
                "n_resolved": summary["n_resolved"],
                "resolved_rate": summary["resolved_rate"],
                "n_any_f2p_pass": summary["n_any_f2p_pass"],
                "f2p_pass_rate": summary["f2p_pass_rate"],
                "total_wall_seconds": summary["total_wall_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
