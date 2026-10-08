#!/usr/bin/env python3
"""After prefix val-474: parse-smoke, then HMPO if McNemar is not a gain.

Does not start a second eval. Does not kill neighbor GPUs. Distill is staged
only after HMPO is either skipped (significant SFT) or launched.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

E = Path("/data/zhizhong/agl-checkpoints/swe_smith_smoke")
P = E / "prefix_sft"
RESULT = P / "RESULT.txt"
LOG = P / "overnight_chain.log"
ABORT = E / "ABORT_OVERNIGHT"


def say(*a) -> None:
    line = time.strftime("[%m-%d %H:%M:%S] ") + " ".join(map(str, a))
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_result(text: str) -> dict:
    """Loose parse of compare_val.py stdout. Fail closed: missing p → n.s."""
    resolved = None
    p_value = None
    sig = False
    m = re.search(r"(\d+)\s*/\s*474", text)
    if m:
        resolved = int(m.group(1))
    m = re.search(r"p\s*=\s*([0-9.]+)", text, re.I)
    if m:
        p_value = float(m.group(1))
    if re.search(r"\bSIG\b|significant", text, re.I) and p_value is not None and p_value < 0.05:
        sig = True
    better = bool(re.search(r"better|gain|improved", text, re.I))
    worse = bool(re.search(r"worse|drop|跌", text, re.I))
    return {
        "resolved": resolved,
        "p": p_value,
        "sig": sig,
        "better": better,
        "worse": worse,
        "raw": text[-2000:],
    }


def wait_result(timeout_s: int = 8 * 3600) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if ABORT.is_file():
            raise SystemExit("ABORT_OVERNIGHT")
        if RESULT.is_file() and RESULT.stat().st_size > 20:
            text = RESULT.read_text(encoding="utf-8", errors="replace")
            if "PREFIX_EVAL_FAIL" in text:
                say("RESULT has FAIL marker")
            if re.search(r"\d+\s*/\s*474", text) or "McNemar" in text or "p=" in text:
                return text
        # also accept drive log finished
        eval_log = P / "drive_prefix_eval.log"
        if eval_log.is_file() and "PREFIX_EVAL_FINISHED" in eval_log.read_text(errors="replace"):
            if RESULT.is_file():
                return RESULT.read_text(encoding="utf-8", errors="replace")
        time.sleep(60)
        if int(time.time() - t0) % 600 < 70:
            say("waiting RESULT", RESULT, "age", round(time.time() - t0))
    raise SystemExit("timeout waiting prefix RESULT")


def format_smoke_ok() -> bool:
    """Cheap format check: already-finished val_sft_prefix status files."""
    root = E / "val_sft_prefix"
    if not root.is_dir():
        say("no val_sft_prefix yet")
        return False
    n = 0
    fmt = 0
    submitted = 0
    resolved = 0
    for st in root.glob("*/status.json"):
        try:
            obj = json.loads(st.read_text(encoding="utf-8"))
        except Exception:
            continue
        n += 1
        reason = str(obj.get("reason") or obj.get("error") or "")
        if "format" in reason.lower():
            fmt += 1
        if obj.get("submitted"):
            submitted += 1
        if obj.get("resolved") or (obj.get("reward") or 0) > 0:
            resolved += 1
    say(f"format_smoke n={n} formatish={fmt} submitted={submitted} resolved={resolved}")
    if n < 50:
        return False
    # 134 baseline submitted a lot; if format errors dominate, do not train.
    return fmt / n < 0.45


def gpu_free(idx: int) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={idx}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().split()[0])


def write_hmpo_ready(info: dict) -> None:
    flag = P / "HMPO_READY.json"
    flag.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    say("wrote", flag)


def main() -> int:
    say("overnight_chain start")
    text = wait_result()
    info = parse_result(text)
    say("prefix result", {k: info[k] for k in ("resolved", "p", "sig", "better", "worse")})
    (P / "RESULT_PARSED.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    parse_ok = format_smoke_ok()
    gain = bool(info["sig"] and info["better"] and not info["worse"] and (info["resolved"] or 0) > 134)
    if gain:
        say("prefix SFT McNemar gain vs 134; skip HMPO unless asked")
        write_hmpo_ready({"skip": True, "reason": "sft_gain", **{k: info[k] for k in ("resolved", "p")}})
        return 0
    if not parse_ok:
        say("format smoke not ok; will not launch HMPO until parse/execute is healthy")
        write_hmpo_ready({"skip": True, "reason": "format_smoke_fail", **info})
        return 2
    # Launch is a separate script so a crash here does not double-start.
    write_hmpo_ready({"skip": False, "reason": "sft_n.s.", "parse_ok": True, **{k: info[k] for k in ("resolved", "p")}})
    launch = Path("/tmp/zz/launch_hmpo.py")
    if launch.is_file():
        say("exec launch_hmpo.py")
        return subprocess.call(["python3", "-u", str(launch)])
    say("HMPO_READY but launch_hmpo.py missing; distill scripts should still be written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
