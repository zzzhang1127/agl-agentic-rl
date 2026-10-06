#!/usr/bin/env python3
import collections
import json
from pathlib import Path

root = Path("/workspace/agl-checkpoints/swe_smith_smoke/val_smith")
n_all = 474
rows = []
for path in root.glob("*/result.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    ev = data.get("eval") or {}
    rows.append((data, ev))

n = len(rows)
ok = sum(1 for _, ev in rows if ev.get("resolved"))
any_f2p = sum(1 for _, ev in rows if int(ev.get("n_f2p_pass") or 0) > 0)
f2p_pass = sum(int(ev.get("n_f2p_pass") or 0) for _, ev in rows)
f2p_total = sum(int(ev.get("n_f2p") or 0) for _, ev in rows)
p2p_ok = sum(int(ev.get("n_p2p_ok") or 0) for _, ev in rows)
p2p_total = sum(int(ev.get("n_p2p") or 0) for _, ev in rows)
timeout = sum(1 for data, _ in rows if data.get("timed_out") or data.get("smith_rc") == 124)
overflow = sum(1 for data, _ in rows if data.get("overflowed"))
print(f"done {n}/{n_all} ({n / n_all:.1%})")
if n:
    print(f"resolved {ok}/{n} = {ok / n:.1%}  of_all={ok / n_all:.1%}")
    print(f"any_f2p {any_f2p}/{n} = {any_f2p / n:.1%}")
print(f"f2p_tests {f2p_pass}/{f2p_total} = {f2p_pass / f2p_total:.1%}" if f2p_total else "f2p_tests 0")
print(f"p2p_tests {p2p_ok}/{p2p_total} = {p2p_ok / p2p_total:.1%}" if p2p_total else "p2p_tests 0")
print(f"timeout {timeout} overflowed {overflow}")
reasons = collections.Counter(str((ev.get("reason") or "ok"))[:60] for _, ev in rows)
print("reasons", dict(reasons.most_common(8)))
