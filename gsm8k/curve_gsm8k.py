#!/usr/bin/env python3
"""把 trainer_gsm8k.log 抽成学习曲线,并给出"涨了没有"的判据。

为什么要单独写:日志里一步一行、1868 步 + 186 次验证,肉眼看不出趋势,
而且 reward 本身有采样噪声,不给噪声底线的"涨了"是没有意义的。

口径(重要,汇报时别说错):
  * gsm8k 的 reward 是**二值精确匹配**(gsm8k_agent.py:63-74),所以
    training/reward 和 val/reward 直接就是解题率,不需要换算。
  * val 固定 200 题(--val-size 200 --seed 42),但采样温度是官方的 1.0,
    所以同一批题重复验证也有噪声 —— 噪声底线按二项分布给:
    SE = sqrt(p(1-p)/200),基线 p=0.66 → 1σ ≈ 3.4pp。
    **要宣称涨了,至少得超过 2σ ≈ 6.7pp**,否则只能说"未被推翻"。
  * training/reward 是每步 32 条 rollout 的均值,SE ≈ 8.7pp,单步跳动不算信号,
    所以下面对训练曲线做滑动平均再看趋势。

用法: python curve_gsm8k.py [日志路径] [--csv 输出.csv]
"""
from __future__ import annotations

import argparse
import math
import re
import sys

# 日志里 ray 的 actor 前缀带 ANSI 颜色码,先剥掉再匹配,否则行首对不上
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"\bstep:(\d+)\b")


def _grab(line: str, key: str) -> float | None:
    m = re.search(rf"\b{re.escape(key)}:(-?[0-9.]+)\b", line)
    return float(m.group(1)) if m else None


def parse(path: str):
    train: dict[int, dict] = {}
    val: dict[int, dict] = {}
    with open(path, errors="replace") as f:
        for raw in f:
            line = ANSI.sub("", raw)
            m = STEP.search(line)
            if not m:
                continue
            step = int(m.group(1))
            vr = _grab(line, "val/reward")
            if vr is not None:
                val[step] = {
                    "reward": vr,
                    "n": _grab(line, "val/n_rollouts"),
                    "trace": _grab(line, "val/n_rollouts_w_trace"),
                    "rew_n": _grab(line, "val/n_rollouts_w_reward"),
                    "len": _grab(line, "val/mean_response_length_per_turn"),
                }
            tr = _grab(line, "training/reward")
            if tr is not None:
                train[step] = {
                    "reward": tr,
                    "n": _grab(line, "training/n_rollouts"),
                    "trace": _grab(line, "training/n_rollouts_w_trace"),
                    "rew_n": _grab(line, "training/n_rollouts_w_reward"),
                    "len": _grab(line, "response_length/training/avg_by_turn"),
                    "wall": _grab(line, "timing/step_start_wall"),
                    "secs": _grab(line, "timing_s/step"),
                }
    return train, val


def se(p: float, n: float) -> float:
    return math.sqrt(max(p * (1.0 - p), 0.0) / n) if n else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?",
                    default="/workspace/agl-checkpoints/gsm8k_r1/trainer_gsm8k.log")
    ap.add_argument("--csv")
    ap.add_argument("--window", type=int, default=20, help="训练曲线滑窗步数")
    a = ap.parse_args()

    train, val = parse(a.log)
    if not train and not val:
        print(f"没从 {a.log} 解出任何 step 行", file=sys.stderr)
        return 1

    base = val.get(0)
    print(f"日志: {a.log}")
    if base:
        s = se(base["reward"], base["n"])
        print(f"训前基线 (step 0): val/reward={base['reward']:.4f} "
              f"= {round(base['reward'] * base['n'])}/{int(base['n'])}  "
              f"1σ={s*100:.1f}pp  长度={base['len']:.1f} tok")
        # 采集完整性:整组丢 rollout 时 capture_rate 是瞎的,只有这两个比值能看出来
        if base["trace"] != base["n"] or base["rew_n"] != base["n"]:
            print(f"  !! 基线采集不完整 trace={base['trace']} reward={base['rew_n']} / {base['n']}")

    # --- 验证曲线 ---
    if len(val) > 1:
        print(f"\n{'step':>6} {'val/reward':>11} {'解出':>9} {'Δ vs 基线':>11} {'判据':>22} {'长度':>8} {'采集':>9}")
        for step in sorted(val):
            v = val[step]
            n = v["n"] or 1
            hit = round(v["reward"] * n)
            cell = f"{hit}/{int(n)}"
            if base and step:
                d = (v["reward"] - base["reward"]) * 100
                s2 = math.hypot(se(v["reward"], n), se(base["reward"], base["n"])) * 100
                z = d / s2 if s2 else 0.0
                # 2σ 以下一律只说"未被推翻",不说涨了
                verdict = ("涨了 (>2σ)" if z >= 2 else
                           "跌了 (<-2σ)" if z <= -2 else "噪声内,未被推翻")
                dcell, vcell = f"{d:+.2f}pp", f"{verdict} z={z:+.2f}"
            else:
                dcell, vcell = "—", "基线"
            cap = "满" if v["trace"] == n and v["rew_n"] == n else f"{v['trace']:.0f}/{v['rew_n']:.0f}"
            print(f"{step:>6} {v['reward']:>11.4f} {cell:>9} {dcell:>11} {vcell:>22} "
                  f"{(v['len'] or 0):>8.1f} {cap:>9}")

    # --- 训练曲线(滑窗) ---
    if train:
        steps = sorted(train)
        w = a.window
        print(f"\n训练曲线(每 {w} 步滑窗均值,单步 SE≈{se(0.6, 32)*100:.1f}pp 所以不看单步):")
        print(f"{'区间':>14} {'reward':>8} {'长度':>8} {'步耗时':>8} {'采集':>8}")
        for i in range(0, len(steps), w):
            chunk = [train[s] for s in steps[i:i + w]]
            rs = [c["reward"] for c in chunk if c["reward"] is not None]
            ls = [c["len"] for c in chunk if c["len"] is not None]
            ts = [c["secs"] for c in chunk if c["secs"] is not None]
            ok = all(c["trace"] == c["n"] and c["rew_n"] == c["n"]
                     for c in chunk if c["n"])
            print(f"{f'{steps[i]}-{steps[min(i+w-1, len(steps)-1)]}':>14} "
                  f"{sum(rs)/len(rs):>8.4f} {(sum(ls)/len(ls) if ls else 0):>8.1f} "
                  f"{(sum(ts)/len(ts) if ts else 0):>7.1f}s {'满' if ok else '有丢':>8}")

        secs = [train[s]["secs"] for s in steps if train[s]["secs"]]
        if secs:
            avg = sum(secs) / len(secs)
            print(f"\n已跑 {len(steps)} 步,平均 {avg:.1f}s/步。"
                  f"官方 2 epoch = 1868 步 → 纯训练约 {1868*avg/3600:.1f}h(验证另算)")

    if a.csv:
        with open(a.csv, "w") as f:
            f.write("step,split,reward,n,n_w_trace,n_w_reward,resp_len,step_secs\n")
            for step in sorted(set(train) | set(val)):
                for split, d in (("val", val.get(step)), ("train", train.get(step))):
                    if not d:
                        continue
                    f.write(f"{step},{split},{d['reward']},{d['n']},{d['trace']},"
                            f"{d['rew_n']},{d['len']},{d.get('secs') or ''}\n")
        print(f"\nCSV 已写 {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
