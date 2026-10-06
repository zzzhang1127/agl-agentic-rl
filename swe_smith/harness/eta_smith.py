#!/usr/bin/env python3
import json
import time
from pathlib import Path

root = Path("/workspace/agl-checkpoints/swe_smith_smoke/val_smith")
n_all = 474
now = time.time()
mtimes = []
walls = []
for p in root.glob("*/result.json"):
    mtimes.append(p.stat().st_mtime)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        walls.append(float(d.get("wall_seconds") or 0))
    except Exception:
        pass
mtimes.sort()
n = len(mtimes)
print("n", n, "remaining", n_all - n)
if n >= 2:
    span = mtimes[-1] - mtimes[0]
    print("first_to_last_hours", round(span / 3600, 2))
    print("overall_per_hour", round(n / (span / 3600), 1) if span > 0 else None)
for minutes in (10, 20, 30):
    recent = [t for t in mtimes if t >= now - minutes * 60]
    print(f"last_{minutes}m", len(recent), "per_hour", round(len(recent) * 60 / minutes, 1))
if walls:
    walls.sort()
    print(
        "wall_p50",
        round(walls[len(walls) // 2], 1),
        "wall_p90",
        round(walls[int(len(walls) * 0.9)], 1),
        "wall_mean",
        round(sum(walls) / len(walls), 1),
    )
    print("near_timeout", sum(1 for w in walls if w >= 590))
print("newest_age_s", round(now - mtimes[-1], 1) if mtimes else None)
