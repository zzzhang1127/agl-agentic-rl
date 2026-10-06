#!/usr/bin/env python3
"""Val sweep with official smith_agent, knobs matched to the OpenCode baseline.

OpenCode val used: 600s agent wall, context 40960, output 8192, no turn cap.
Pytest grading still runs after the agent, same as OpenCode.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import run_opencode_smoke as smoke

EXAMPLE = Path("/workspace/projects/agent-lightning/examples/swe_smith")
SPLIT = os.environ.get("AGL_SWEEP_SPLIT", "val").strip().lower()
_DATASETS = {
    "val": [EXAMPLE / "val_dataset_filtered.jsonl"],
    "train": [EXAMPLE / "train_dataset_mixed.jsonl"],
}
DATASETS = _DATASETS.get(SPLIT, _DATASETS["val"])
SWEEP_ROOT = Path(os.environ.get("AGL_SWEEP_ROOT", str(smoke.ROOT / "val_smith")))
SHARD_COUNT = max(1, int(os.environ.get("AGL_SWEEP_SHARD_COUNT", "1")))
SHARD_INDEX = int(os.environ.get("AGL_SWEEP_SHARD_INDEX", "0"))
PROGRESS = SWEEP_ROOT / (
    f"progress.shard{SHARD_INDEX}.jsonl" if SHARD_COUNT > 1 else "progress.jsonl"
)
SUMMARY = SWEEP_ROOT / "summary.json"
FOCUS_REPO = os.environ.get("AGL_SWEEP_REPO", "ALL").strip()
WORKERS = max(1, int(os.environ.get("AGL_SWEEP_WORKERS", "3")))
LIMIT = os.environ.get("AGL_SWEEP_LIMIT", "").strip()
# 只跑指定子集:一个每行一个 instance_id 的文本文件。用于"把撞轮次上限的题
# 单独拎出来、换更高的 SMITH_MAX_TURNS 重跑"这类边际收益测量。
ONLY_FILE = os.environ.get("AGL_SWEEP_ONLY", "").strip()
RESUME = os.environ.get("AGL_SWEEP_RESUME", "1") != "0"
SAMPLE_TIMEOUT = int(os.environ.get("AGL_SAMPLE_TIMEOUT", "600"))
EVAL_TIMEOUT = int(os.environ.get("SMITH_EVAL_TIMEOUT", "600"))
MAX_TURNS = int(os.environ.get("SMITH_MAX_TURNS", "10000"))
MAX_TOKENS = int(os.environ.get("AGL_MAX_TOKENS", "8192"))
MODEL = os.environ.get("AGL_MODEL", "Qwen3-8B")
MAX_MODEL_LEN = int(os.environ.get("AGL_MAX_MODEL_LEN", "40960"))
SMITH_PY = Path(os.environ.get("AGL_SMITH_AGENT", str(smoke.ROOT / "smith_agent.py")))
ROLLOUT_PY = Path(os.environ.get("AGL_SMITH_ROLLOUT", str(smoke.ROOT / "smith_rollout.py")))
# 2026-10-06: 容器名前缀可配。smith_fleet.sh(探针)起跑时 `docker rm -f` 掉所有 ^agl-sm- 容器,
# 教师采集(teacher_smith_b5.sh)与探针并行跑,必须用别的前缀(agl-st)才不会每次探针都被清掉。
CONTAINER_PREFIX = os.environ.get("AGL_SWEEP_CONTAINER_PREFIX", "agl-sm").strip() or "agl-sm"

_print_lock = threading.Lock()
_progress_lock = threading.Lock()
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
        for line in path.open(encoding="utf-8"):
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
    if ONLY_FILE:
        keep = {
            ln.strip()
            for ln in Path(ONLY_FILE).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
        found = [row for row in found if row["instance_id"] in keep]
        if not found:
            raise SystemExit(f"AGL_SWEEP_ONLY={ONLY_FILE} 没匹配到任何 instance_id")
    if LIMIT:
        found = found[: int(LIMIT)]
    if SHARD_COUNT > 1:
        found = [row for i, row in enumerate(found) if i % SHARD_COUNT == SHARD_INDEX]
    return found


def openai_base_url() -> str:
    raw = smoke.VLLM_URL.rstrip("/")
    parsed = urllib.parse.urlparse(raw if "://" in raw else f"http://{raw}")
    port = parsed.port or 80
    return f"http://host.docker.internal:{port}/v1"


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


def prepare_smith_net(cid: str) -> None:
    deny = r"""
set -e
if command -v iptables >/dev/null 2>&1; then
  iptables -F OUTPUT || true
  iptables -P OUTPUT DROP || true
  iptables -A OUTPUT -o lo -j ACCEPT || true
  iptables -A OUTPUT -p tcp -d 172.17.0.1 -m multiport --dports 18000:18009 -j ACCEPT || true
  iptables -A OUTPUT -p tcp -d host.docker.internal -m multiport --dports 18000:18009 -j ACCEPT || true
  iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT || true
fi
mkdir -p /opt/agl_eval
"""
    proc = smoke.docker_exec(cid, ["bash", "-lc", deny], timeout=30)
    if proc.returncode != 0:
        log(f"warn: net lock: {proc.stderr}")


def maybe_hide_git(cid: str) -> None:
    """If timeout killed the agent before relocate_git, move .git for eval_inside."""
    smoke.docker_exec(
        cid,
        [
            "bash",
            "-lc",
            "if [ ! -d /opt/agl_tmp/HEAD ] && [ -d /testbed/.git ]; then "
            "rm -rf /opt/agl_tmp; mv /testbed/.git /opt/agl_tmp; fi",
        ],
        timeout=60,
    )


def run_one_instance(instance: dict[str, Any], run_image: str | None = None) -> dict[str, Any]:
    iid = str(instance["instance_id"])
    run_image = run_image or ensure_run_image(str(instance.get("image_name")))
    short = iid.replace(".", "-")[-48:]
    name = f"{CONTAINER_PREFIX}-{SHARD_INDEX}-{short}"
    job_dir = job_dir_for(iid)
    if job_dir.exists() and not RESUME:
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    inst_path = job_dir / "instance.json"
    inst_path.write_text(json.dumps(instance, ensure_ascii=False), encoding="utf-8")
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    log(f"start {iid} f2p={instance['_n_f2p']} split={instance['_split']}")
    result: dict[str, Any] = {"instance_id": iid, "split": instance["_split"], "n_f2p": instance["_n_f2p"]}
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
            "--cap-add",
            "NET_ADMIN",
            "-e",
            "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "-e",
            "HOME=/tmp/smith-home",
            "-e",
            "OPENAI_API_KEY=dummy",
            "-v",
            f"{SMITH_PY}:/opt/agl/smith_agent.py:ro",
            "-v",
            f"{ROLLOUT_PY}:/opt/agl/smith_rollout.py:ro",
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
        prepare_smith_net(cid)
        runner = job_dir / "inside.sh"
        runner.write_text(
            "#!/bin/bash\n"
            "set -u\n"
            "export PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
            "export HOME=/tmp/smith-home\n"
            f"timeout {SAMPLE_TIMEOUT} python /opt/agl/smith_rollout.py\n",
            encoding="utf-8",
        )
        subprocess.check_call(["docker", "cp", str(runner), f"{cid}:/tmp/inside.sh"])
        smoke.docker_exec(cid, ["bash", "-lc", "chmod +x /tmp/inside.sh"], timeout=10)
        t0 = time.time()
        metrics_before = smoke.vllm_metrics()
        gpu_before = smoke.nvidia_sample()
        smith_log = job_dir / "smith.log"
        smith_err = job_dir / "smith.stderr"
        env_pairs = [
            ("AGL_OPENAI_BASE_URL", openai_base_url()),
            ("AGL_MODEL", MODEL),
            ("AGL_MAX_TOKENS", str(MAX_TOKENS)),
            ("SMITH_MAX_TURNS", str(MAX_TURNS)),
            ("SMITH_OBS_CHAR_CAP", os.environ.get("SMITH_OBS_CHAR_CAP", "6000")),
            ("SMITH_CMD_TIMEOUT", os.environ.get("SMITH_CMD_TIMEOUT", "120")),
            ("SMITH_TEMPERATURE", os.environ.get("SMITH_TEMPERATURE", "1.0")),
            ("SMITH_MAX_FORMAT_ERRORS", os.environ.get("SMITH_MAX_FORMAT_ERRORS", "3")),
            ("AGL_INSTANCE_JSON", "/opt/agl_eval/instance.json"),
            ("AGL_STATUS_OUT", "/opt/agl_eval/status.json"),
            ("OPENAI_API_KEY", "dummy"),
        ]
        docker_env: list[str] = []
        for key, val in env_pairs:
            docker_env.extend(["-e", f"{key}={val}"])
        with smith_log.open("w", encoding="utf-8") as out, smith_err.open("w", encoding="utf-8") as err:
            oc_proc = subprocess.run(
                ["docker", "exec", "-w", "/testbed", *docker_env, cid, "bash", "/tmp/inside.sh"],
                stdout=out,
                stderr=err,
                text=True,
            )
        metrics_after = smoke.vllm_metrics()
        gpu_after = smoke.nvidia_sample()
        wall = time.time() - t0
        maybe_hide_git(cid)
        status: dict[str, Any] = {}
        # 2026-10-02: 这两处读取显式用 -w /,不用镜像默认的 /testbed。agent 偶尔会把自己的
        # /testbed 删掉(smoke 里 gctpviq9:它写的测试脚本清理时删到了 os.path.dirname(__file__)),
        # 之后 docker exec 连 chdir 都做不到,status.json 明明写出来了也读不到,看上去像
        # harness 解析失败。status.json 是绝对路径,不需要 cwd。
        status_blob = subprocess.run(
            ["docker", "exec", "-w", "/", cid, "bash", "-lc", "cat /opt/agl_eval/status.json 2>/dev/null || true"],
            text=True,
            capture_output=True,
            timeout=15,
        )
        try:
            if status_blob.stdout.strip():
                status = json.loads(status_blob.stdout)
        except json.JSONDecodeError:
            status = {"error": "status parse failed"}
        (job_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        err_s = str(status.get("error") or "")
        testbed_gone = (
            subprocess.run(
                ["docker", "exec", "-w", "/", cid, "bash", "-lc", "test -d /testbed"],
                capture_output=True,
                timeout=15,
            ).returncode
            != 0
        )
        checkout_failed = oc_proc.returncode not in (0, 124) and (
            "checkout" in err_s.lower() or "not a git repository" in err_s.lower()
        )
        if checkout_failed:
            eval_json = {
                "resolved": False,
                "reason": f"checkout failed: {err_s}"[:300],
                "n_f2p": instance["_n_f2p"],
                "n_f2p_pass": 0,
                "n_p2p": 0,
                "n_p2p_ok": 0,
            }
        elif testbed_gone:
            # agent 把自己的工作目录删了,仓库和改动一起没了,评测无从谈起。
            # 这是模型行为不是 harness 故障,算 resolved=False,但要能和解析失败区分开。
            eval_json = {
                "resolved": False,
                "reason": "sandbox destroyed: /testbed no longer exists after rollout",
                "n_f2p": instance["_n_f2p"],
                "n_f2p_pass": 0,
                "n_p2p": 0,
                "n_p2p_ok": 0,
            }
        else:
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
        log_text = smith_log.read_text(encoding="utf-8", errors="replace") + smith_err.read_text(
            encoding="utf-8", errors="replace"
        )
        overflowed = bool(status.get("overflowed")) or (
            "maximum context length" in log_text.lower() or "contextoverflow" in log_text.lower()
        )
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
            "smith_rc": oc_proc.returncode,
            "timed_out": oc_proc.returncode == 124 or wall >= SAMPLE_TIMEOUT - 2,
            "wall_seconds": round(wall, 1),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "vllm_prompt_tokens": prompt_delta,
            "vllm_completion_tokens": gen_delta,
            "submitted": status.get("submitted"),
            "n_turns": status.get("n_turns"),
            "max_prompt_tokens": status.get("max_prompt_tokens"),
            "overflowed": overflowed,
            "status_error": status.get("error"),
            "eval": eval_json,
            "max_model_len": MAX_MODEL_LEN,
            "max_tokens": MAX_TOKENS,
            "max_turns": MAX_TURNS,
            "sample_timeout": SAMPLE_TIMEOUT,
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
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    ev = result.get("eval") or {}
    log(
        f"done {iid} wall={result.get('wall_seconds')} resolved={ev.get('resolved')} "
        f"f2p={ev.get('n_f2p_pass')}/{ev.get('n_f2p')} turns={result.get('n_turns')} "
        f"timeout={result.get('timed_out')} overflowed={result.get('overflowed')} {ev.get('reason')}"
    )
    append_progress(
        {
            "instance_id": iid,
            "resolved": ev.get("resolved"),
            "n_f2p_pass": ev.get("n_f2p_pass"),
            "n_f2p": ev.get("n_f2p"),
            "n_p2p_ok": ev.get("n_p2p_ok"),
            "n_p2p": ev.get("n_p2p"),
            "wall_seconds": result.get("wall_seconds"),
            "smith_rc": result.get("smith_rc"),
            "timed_out": result.get("timed_out"),
            "n_turns": result.get("n_turns"),
            "overflowed": result.get("overflowed"),
            "reason": ev.get("reason"),
        }
    )
    return result


def summarize(results: list[dict[str, Any]], t0: float, gpu_idle: dict[str, Any]) -> dict[str, Any]:
    resolved = [r for r in results if (r.get("eval") or {}).get("resolved")]
    any_f2p = [r for r in results if int((r.get("eval") or {}).get("n_f2p_pass") or 0) > 0]
    f2p_pass = sum(int((r.get("eval") or {}).get("n_f2p_pass") or 0) for r in results)
    f2p_total = sum(int((r.get("eval") or {}).get("n_f2p") or r.get("n_f2p") or 0) for r in results)
    n_timeout = sum(1 for r in results if r.get("timed_out"))
    n_overflow = sum(1 for r in results if r.get("overflowed"))
    summary = {
        "n_instances": len(results),
        "n_resolved": len(resolved),
        "resolved_rate": (len(resolved) / len(results)) if results else 0.0,
        "n_any_f2p_pass": len(any_f2p),
        "f2p_pass_rate": (f2p_pass / f2p_total) if f2p_total else 0.0,
        "n_timeout": n_timeout,
        "n_overflowed": n_overflow,
        "max_model_len": MAX_MODEL_LEN,
        "max_tokens": MAX_TOKENS,
        "max_turns": MAX_TURNS,
        "sample_timeout": SAMPLE_TIMEOUT,
        "total_wall_seconds": round(time.time() - t0, 1),
        "gpu_idle": gpu_idle,
        "shard": SHARD_INDEX,
        "shard_count": SHARD_COUNT,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    if not SMITH_PY.is_file():
        raise SystemExit(f"missing smith_agent.py: {SMITH_PY}")
    if not ROLLOUT_PY.is_file():
        raise SystemExit(f"missing smith_rollout.py: {ROLLOUT_PY}")
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    instances = load_repo_instances()
    if not instances:
        raise SystemExit(f"no instances for repo {FOCUS_REPO} split {SPLIT}")
    images = sorted({str(r.get("image_name")) for r in instances})
    log(
        f"smith split={SPLIT} shard={SHARD_INDEX}/{SHARD_COUNT} n={len(instances)} "
        f"images={len(images)} workers={WORKERS} timeout={SAMPLE_TIMEOUT}s "
        f"max_turns={MAX_TURNS} max_tokens={MAX_TOKENS} ctx={MAX_MODEL_LEN} vllm={smoke.VLLM_URL}"
    )
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
    all_done: list[dict[str, Any]] = []
    for path in SWEEP_ROOT.glob("*/result.json"):
        try:
            all_done.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    summary = summarize(all_done or done, t0, gpu_idle)
    log(f"wrote {SUMMARY}")
    print(
        json.dumps(
            {
                "n": summary["n_instances"],
                "n_resolved": summary["n_resolved"],
                "resolved_rate": summary["resolved_rate"],
                "n_any_f2p_pass": summary["n_any_f2p_pass"],
                "n_timeout": summary["n_timeout"],
                "n_overflowed": summary["n_overflowed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
