#!/usr/bin/env python3
"""OPSD go/no-go 诊断的汇总:四种 hint 条件在同一 100 题子集上的 resolved/交卷/轮数,
配对 McNemar(exact)对 none 条件与对 ep3 全量跑(val_sft3_tmpl2)的同子集读数。"""
import json
import math
import statistics
from pathlib import Path

E = Path("/workspace/agl-checkpoints/swe_smith_smoke")
IDS = [ln.strip() for ln in (E / "hint_diag/val100_ids.txt").read_text().splitlines() if ln.strip()]
MODES = ["none", "tests", "files", "patch"]


def load(d: Path) -> dict:
    out = {}
    for iid in IDS:
        p = d / iid / "result.json"
        if p.is_file():
            try:
                out[iid] = json.loads(p.read_text())
            except Exception:
                pass
    return out


def resolved(r: dict) -> bool:
    return bool((r.get("eval") or {}).get("resolved"))


def mcnemar(a: dict, b: dict):
    common = [i for i in IDS if i in a and i in b]
    b_only = sum(1 for i in common if resolved(b[i]) and not resolved(a[i]))
    a_only = sum(1 for i in common if resolved(a[i]) and not resolved(b[i]))
    n = a_only + b_only
    if n == 0:
        return len(common), a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, j) for j in range(0, k + 1)) / 2**n * 2
    return len(common), a_only, b_only, min(1.0, p)


def row(name: str, rs: dict) -> str:
    n = len(rs)
    if n == 0:
        return f"{name:<14} n=0"
    res = sum(resolved(r) for r in rs.values())
    sub = sum(1 for r in rs.values() if r.get("submitted"))
    res_sub = sum(1 for r in rs.values() if r.get("submitted") and resolved(r))
    turns = [r.get("n_turns") for r in rs.values() if isinstance(r.get("n_turns"), int)]
    cap = sum(1 for t in turns if t >= 40)
    to = sum(1 for r in rs.values() if r.get("timed_out"))
    hint0 = sum(1 for r in rs.values() if r.get("hint_mode", "none") != "none" and not r.get("hint_chars"))
    hc = [r.get("hint_chars", 0) for r in rs.values() if r.get("hint_chars")]
    p50 = statistics.median(turns) if turns else float("nan")
    return (
        f"{name:<14} n={n:<3} resolved={res:<3} submitted={sub:<3} res/sub={res_sub}/{sub} "
        f"cap40={cap:<3} turns_p50={p50:<4} timeout={to} hint_empty={hint0} hint_chars_p50={statistics.median(hc) if hc else 0}"
    )


def main():
    ref = load(E / "val_sft3_tmpl2")
    print(row("ep3_full_ref", ref))
    data = {m: load(E / f"val_hint_{m}") for m in MODES}
    for m in MODES:
        print(row(m, data[m]))
    print("--- paired McNemar (exact), rows = condition vs none / vs ep3 full-run subset")
    for m in MODES:
        if not data[m]:
            continue
        if m != "none" and data["none"]:
            n, a_only, b_only, p = mcnemar(data["none"], data[m])
            print(f"{m:<8} vs none: n={n} none_only={a_only} {m}_only={b_only} p={p:.4f}")
        n, a_only, b_only, p = mcnemar(ref, data[m])
        print(f"{m:<8} vs ep3_ref: n={n} ref_only={a_only} {m}_only={b_only} p={p:.4f}")


if __name__ == "__main__":
    main()
