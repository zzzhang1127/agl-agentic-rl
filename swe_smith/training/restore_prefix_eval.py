#!/usr/bin/env python3
"""Restore smith harness to the val-474 protocol, then eval prefix SFT.

Drive script died because smoke smith_rollout.py lacked skip_special_tokens.
The 134 baseline used the public harness (native <function> parse + skip=False).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

E = Path("/data/zhizhong/agl-checkpoints/swe_smith_smoke")
P = E / "prefix_sft"
PUB_A = Path("/data/zhizhong/projects/agl-agentic-rl/swe_smith/agents/smith_agent.py")
PUB_R = Path("/data/zhizhong/projects/agl-agentic-rl/swe_smith/harness/smith_rollout.py")
CANON = Path("/data/zhizhong/projects/agent-lightning/examples/swe_smith/agents/smith_agent.py")
CANON_R = Path("/data/zhizhong/projects/agent-lightning/examples/swe_smith/opencode_smoke/smith_rollout.py")
HF = Path("/data/zhizhong/models/MiniCPM5-2B-sft-prefix")
LOG = P / "drive_prefix_eval.log"
TAG = "pfxeval"
OUT = "val_sft_prefix"


def say(*a) -> None:
    line = time.strftime("[%m-%d %H:%M:%S] ") + " ".join(map(str, a))
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def backup(src: Path, stamp: str) -> None:
    if not src.exists():
        return
    bak = src.with_name(src.name + f".bak_{stamp}")
    shutil.copy2(src, bak)
    say("backup", src, "->", bak)


def patch_hmpo_skip_prompt(path: Path) -> None:
    """When SMITH_HMPO=1, skip BOTH length shapers so the trainer sees raw 0/1."""
    text = path.read_text(encoding="utf-8")
    old = (
        "    if not use_hmpo:\n"
        "        reward = length_penalized_reward(reward, n_turns, max_turns, t0=len_pen_t0, lam=len_pen_lambda, is_train=is_train)\n"
        "\n"
        "    # Prompt-length penalty (plan B): stack a context-bloat penalty on the same\n"
        "    # SOLVED-train gating, keyed on the rollout's largest prompt_tokens.\n"
        "    prompt_pen_soft = int(os.environ.get(\"SMITH_PROMPT_PEN_SOFT_START\", \"50000\"))\n"
        "    prompt_pen_hard = int(os.environ.get(\"SMITH_PROMPT_PEN_HARD_CAP\", \"64000\"))\n"
        "    prompt_pen_max = float(os.environ.get(\"SMITH_PROMPT_PEN_MAX\", \"0.1\"))\n"
        "    if is_train and max_prompt_tokens >= prompt_pen_hard:\n"
        "        log.warning(\"max_prompt_tokens=%d >= hard_cap=%d (context near budget)\", max_prompt_tokens, prompt_pen_hard)\n"
        "    reward = prompt_length_penalty(\n"
        "        reward,\n"
        "        max_prompt_tokens,\n"
        "        soft_start=prompt_pen_soft,\n"
        "        hard_cap=prompt_pen_hard,\n"
        "        max_pen=prompt_pen_max,\n"
        "        is_train=is_train,\n"
        "        solved=resolved,\n"
        "    )\n"
    )
    new = (
        "    prompt_pen_soft = int(os.environ.get(\"SMITH_PROMPT_PEN_SOFT_START\", \"50000\"))\n"
        "    prompt_pen_hard = int(os.environ.get(\"SMITH_PROMPT_PEN_HARD_CAP\", \"64000\"))\n"
        "    prompt_pen_max = float(os.environ.get(\"SMITH_PROMPT_PEN_MAX\", \"0.1\"))\n"
        "    if is_train and max_prompt_tokens >= prompt_pen_hard:\n"
        "        log.warning(\"max_prompt_tokens=%d >= hard_cap=%d (context near budget)\", max_prompt_tokens, prompt_pen_hard)\n"
        "    if not use_hmpo:\n"
        "        reward = length_penalized_reward(reward, n_turns, max_turns, t0=len_pen_t0, lam=len_pen_lambda, is_train=is_train)\n"
        "        reward = prompt_length_penalty(\n"
        "            reward,\n"
        "            max_prompt_tokens,\n"
        "            soft_start=prompt_pen_soft,\n"
        "            hard_cap=prompt_pen_hard,\n"
        "            max_pen=prompt_pen_max,\n"
        "            is_train=is_train,\n"
        "            solved=resolved,\n"
        "        )\n"
    )
    if old not in text:
        if "if not use_hmpo:" in text and "prompt_length_penalty" in text:
            say("WARN hmpo prompt-skip pattern drifted; leaving", path)
        return
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    say("patched HMPO skip prompt penalty", path)


def main() -> int:
    stamp = time.strftime("%H%M%S")
    if not HF.is_dir() or not (HF / "model.safetensors.index.json").is_file():
        say("FATAL missing prefix weights", HF)
        return 6
    for src in (E / "smith_agent.py", E / "smith_rollout.py", CANON, CANON_R):
        backup(src, stamp)
    shutil.copy2(PUB_A, E / "smith_agent.py")
    shutil.copy2(PUB_A, CANON)
    shutil.copy2(PUB_R, E / "smith_rollout.py")
    shutil.copy2(PUB_R, CANON_R)
    patch_hmpo_skip_prompt(E / "smith_agent.py")
    shutil.copy2(E / "smith_agent.py", CANON)
    # keep public notebook copy in sync for the next git push
    shutil.copy2(E / "smith_agent.py", PUB_A)
    ra = (E / "smith_rollout.py").read_text(encoding="utf-8")
    aa = (E / "smith_agent.py").read_text(encoding="utf-8")
    for key in ("skip_special_tokens", "strip_turn_delims", "_STOP_STRINGS"):
        if key not in ra or key not in aa:
            say("FATAL missing", key)
            return 3
    if (E / "smith_agent.py").read_bytes() != CANON.read_bytes():
        say("FATAL smoke agent != CANON")
        return 3
    say("harness restored; native parse + skip_special_tokens=False")

    os.chdir(E)
    env = os.environ.copy()
    env["FLEET_CARDS"] = "5 3 1 2"
    env["FLEET_BUDGET"] = "30000"
    start = subprocess.run(
        ["./smith_fleet.sh", "start", str(HF), OUT, TAG],
        env=env,
        capture_output=True,
        text=True,
    )
    blob = (start.stdout or "") + (start.stderr or "")
    say("fleet start rc", start.returncode)
    for line in blob.splitlines()[-20:]:
        say("fleet:", line)
    if start.returncode != 0 or f"FLEET_UP {TAG}" not in blob:
        say("PREFIX_EVAL_FAIL FLEET")
        return 7
    wait = subprocess.run(["./smith_fleet.sh", "wait", TAG, OUT], env=env)
    say("wait rc", wait.returncode)
    subprocess.run(["./smith_fleet.sh", "stop", TAG], env=env)
    cmp = subprocess.run(
        ["python3", str(E / "rft1/compare_val.py"), OUT, "val_sft3_tmpl2"],
        capture_output=True,
        text=True,
    )
    (P / "RESULT.txt").write_text(cmp.stdout or "", encoding="utf-8")
    say((cmp.stdout or "").strip() or "empty compare")
    say("PREFIX_EVAL_FINISHED")
    return 0 if wait.returncode == 0 else wait.returncode


if __name__ == "__main__":
    raise SystemExit(main())
