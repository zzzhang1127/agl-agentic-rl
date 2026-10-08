#!/usr/bin/env python3
"""Start smith GRPO+HMPO from prefix (or ep3) after val-474 is n.s.

Reuses AGL server :18082 if healthy. Waits for GPUs 1,2,3,5 >=48G free.
Does not kill other users. Abort: touch ABORT_OVERNIGHT.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

E = Path("/data/zhizhong/agl-checkpoints/swe_smith_smoke")
P = E / "prefix_sft"
REPO = Path("/data/zhizhong/projects/agent-lightning")
CKPT = Path("/data/zhizhong/agl-checkpoints/smith_rl_hmpo")
LOG = P / "launch_hmpo.log"
ABORT = E / "ABORT_OVERNIGHT"
READY = P / "HMPO_READY.json"
GPUS = (1, 2, 3, 5)
SERVER = "http://127.0.0.1:18082"


def say(*a) -> None:
    line = time.strftime("[%m-%d %H:%M:%S] ") + " ".join(map(str, a))
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def gpu_free(idx: int) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={idx}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().split()[0])


def wait_gpus(need: int = 48000, timeout_s: int = 6 * 3600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if ABORT.is_file():
            raise SystemExit("ABORT_OVERNIGHT")
        frees = {g: gpu_free(g) for g in GPUS}
        say("gpu free", frees)
        if all(v >= need for v in frees.values()):
            return
        time.sleep(60)
    raise SystemExit("timeout waiting GPUs for HMPO")


def healthz() -> bool:
    r = sh(["curl", "-sf", "--max-time", "3", f"{SERVER}/healthz"])
    return r.returncode == 0


def init_model() -> str:
    prefix = Path("/data/zhizhong/models/MiniCPM5-2B-sft-prefix")
    ep3 = Path("/data/zhizhong/models/MiniCPM5-2B-sft-v3-ep3")
    parsed = {}
    if (P / "RESULT_PARSED.json").is_file():
        parsed = json.loads((P / "RESULT_PARSED.json").read_text())
    resolved = parsed.get("resolved")
    # If prefix scored worse than 134, still start from prefix (it is the latest SFT)
    # unless weights missing.
    if prefix.is_dir() and (prefix / "model.safetensors.index.json").is_file():
        say("HMPO init prefix resolved=", resolved)
        return str(prefix)
    say("HMPO init fallback ep3")
    return str(ep3)


def main() -> int:
    P.mkdir(parents=True, exist_ok=True)
    if READY.is_file():
        info = json.loads(READY.read_text())
        if info.get("skip"):
            say("HMPO skipped", info.get("reason"))
            return 0
    wait_gpus()
    if not healthz():
        say("AGL server 18082 not healthy; not starting a second stack")
        (P / "HMPO_BLOCKED.txt").write_text("agl server 18082 down\n", encoding="utf-8")
        return 3
    model = init_model()
    CKPT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "1,2,3,5",
            "CUDA_MPS_PIPE_DIRECTORY": "/data/zhizhong/mps_bypass_empty",
            "SMITH_HMPO": "1",
            "AGL_MODEL_NAME": model,
            "AGL_CKPT_DIR": str(CKPT),
            "AGL_SERVER_PORT": "18082",
            "AGL_KEY": env.get("AGL_KEY") or "dummy",
            "no_proxy": "127.0.0.1,localhost",
            "NO_PROXY": "127.0.0.1,localhost",
        }
    )
    # Refresh ConfigMap so pods get the restored smith_agent.py
    cm = sh(
        [
            "kubectl",
            "-n",
            "default",
            "create",
            "configmap",
            "swe-smith-agent-scripts",
            f"--from-file=smith_agent.py={REPO}/examples/swe_smith/agents/smith_agent.py",
            "--dry-run=client",
            "-o",
            "yaml",
        ]
    )
    if cm.returncode == 0:
        apply = subprocess.run(["kubectl", "-n", "default", "apply", "-f", "-"], input=cm.stdout, text=True)
        say("configmap apply rc", apply.returncode)
    else:
        say("configmap dry-run failed", cm.stderr[-400:] if cm.stderr else "")
    logf = P / "hmpo_trainer.log"
    cmd = [
        "bash",
        str(REPO / "examples/swe_smith/run.sh"),
        "trainer",
        "trainer.total_epochs=2",
        "trainer.save_freq=5",
        "trainer.test_freq=5",
        "data.train_batch_size=4",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.actor.optim.lr=2e-6",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "trainer.max_actor_ckpt_to_keep=2",
    ]
    say("start trainer", cmd)
    with logf.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT)
    (P / "hmpo_trainer.pid").write_text(str(proc.pid), encoding="utf-8")
    say("hmpo trainer pid", proc.pid, "log", logf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
