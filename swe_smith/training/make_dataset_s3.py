#!/usr/bin/env python3
"""从 train_dataset_screened.jsonl(741)生成 train_dataset_s3.jsonl(700)。

去掉的 41 道是「F2P_ONLY 过滤之后 P2P 列表为空」的题 —— 它们**没有回归护栏**。

为什么这 41 道必须去掉:`SMITH_F2P_ONLY=1` 把 P2P 收窄到只保留「含 F2P 测试的
那些文件」里的用例(实测只留 6.4%,中位 269 → 32)。这是为了不撞 600s 评测超时
(人类 09-21 明令不准把超时提到 1800s),代价是一部分题过滤后一条 P2P 都不剩。
对这些题,「破坏既有功能」在奖励里**完全看不见**:模型可以把文件删空、只让 F2P
那几条通过,照样拿满分。s3 改成 F2P 比例稠密奖励之后这个洞更值钱了 ——
部分分意味着「删一半」也能拿分,所以必须先把没护栏的题请出训练集。

数据集本身被 .gitignore(`examples/**/*dataset*.jsonl`,45MB),所以这个脚本
就是它唯一的定义。前一级是 screen_tasks.py(commit d43f5fd)。

    python3 make_dataset_s3.py            # 写 train_dataset_s3.jsonl
    python3 make_dataset_s3.py --dry-run  # 只统计
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "train_dataset_screened.jsonl"
DST = HERE / "train_dataset_s3.jsonl"


def kept_p2p(eval_meta: dict) -> list[str] | None:
    """复现 agents/smith_agent.py 在 SMITH_F2P_ONLY=1 下的 P2P 过滤。
    返回 None 表示这条本身没有 F2P(screen 之后不该出现)。"""
    f2p = eval_meta.get("FAIL_TO_PASS") or []
    p2p = eval_meta.get("PASS_TO_PASS") or []
    if not f2p:
        return None
    files = sorted({t.split("::", 1)[0] for t in f2p})
    return [t for t in p2p if any(t.startswith(f) for f in files)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    keep, drop, nof2p = [], [], 0
    for d in rows:
        em = d.get("eval_meta") or d
        k = kept_p2p(em)
        if k is None:
            nof2p += 1
            continue
        (drop if len(k) == 0 else keep).append(d)

    print(f"源 {SRC.name}: {len(rows)} 道")
    print(f"  无 F2P(异常,应为 0): {nof2p}")
    print(f"  过滤后 P2P 为空 → 丢弃: {len(drop)}")
    print(f"  保留: {len(keep)}")
    if args.dry_run:
        return 0
    with DST.open("w") as fh:
        for d in keep:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"已写 {DST} ({len(keep)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
