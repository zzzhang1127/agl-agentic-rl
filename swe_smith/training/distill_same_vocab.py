#!/usr/bin/env python3
"""Same-vocab distill protocol (not RL). Teacher = original MiniCPM5-2B.

Do NOT start until HMPO is done or skipped. Previous GKD1/PG-OPD1 used a
gold-hint ep3 teacher on the student's tokens — that is not this recipe.

Two losses, both require teacher AND student rollouts:

1. token soft labels: KL(p_T(.|x) || p_S(.|x)) on assistant tokens of the
   student's on-policy trace (and a symmetric term on the teacher's trace).
2. GKD / PG-OPD: same KL family, but both models sample; PG-OPD uses
   A_t = log p_T(y_t) - log p_S(y_t) with PPO-clip.

Student init: MiniCPM SFT (prefix if it did not collapse vs 134, else ep3).
Teacher weights: /data/zhizhong/models/MiniCPM5-2B  (never ep3 / prefix / rft).
"""
from __future__ import annotations

import json
from pathlib import Path

NOTE = Path("/data/zhizhong/agl-checkpoints/swe_smith_smoke/prefix_sft/DISTILL_PROTOCOL.json")

PROTOCOL = {
    "kind": "self-distillation",
    "not": "RL",
    "teacher": "/data/zhizhong/models/MiniCPM5-2B",
    "teacher_val474": 123,
    "student_candidates": {
        "prefix": "/data/zhizhong/models/MiniCPM5-2B-sft-prefix",
        "ep3": "/data/zhizhong/models/MiniCPM5-2B-sft-v3-ep3",
        "ep3_val474": 134,
    },
    "risk": "teacher 123 < student 134; token KL can pull the student backward",
    "prior_failure": {
        "gkd1": 108,
        "pgopd1": 119,
        "bug": "teacher was ep3+gold-hint on student tokens, not dual rollout, not base MiniCPM5-2B",
    },
    "must": [
        "skip_special_tokens=False in smith_rollout._query",
        "parse_action accepts MiniCPM native bash function tags",
        "both teacher and student actually sample (status.json messages)",
        "loss only on assistant tokens",
        "eval val-474 vs val_sft3_tmpl2 134 McNemar",
    ],
}


def main() -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", NOTE)


if __name__ == "__main__":
    main()
