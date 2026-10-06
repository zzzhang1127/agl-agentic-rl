#!/usr/bin/env python3
"""s2 全量探针判决:把一次 474 评测和基线做配对 McNemar,给出 CONTINUE / STOP。

为什么必须配对检验而不是看绝对数 —— 10-04 实测:把 168 道在 40 轮失败的题原样重跑
(同模型、同配置、只是重新采样 temp=0.6),有 13 道翻成通过。单次 474 评测的 run-to-run
噪声量级就在十几道,所以"比基线少 10 道"本身不说明任何事。

用法: python3 s2_verdict.py <probe_dir> [baseline_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from math import comb

# 基线换成 10-05 23:16 跑完的 `val_sft3_fixed`(SFT ep3 = **127**/474),不是旧的
# `val_smith_mc_sft3`(136/474)。两次 harness 不是同一个:后者是 949 行的分叉拷贝,
# 前者是 canonical 1220 行(§55 教训 5 换过)。127 vs 136 差 9 道,在 p≈0.28 处
# 1σ≈9.9 —— 统计上不显著,但**口径不同就不能混用**:配对 McNemar 的 b/c 是逐题比,
# 换 harness 会在单题层面造出成片的伪翻转。s4 的所有探针都对 127 这一份。
BASELINE = "/workspace/agl-checkpoints/swe_smith_smoke/val_sft3_fixed"


def load(d: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for f in glob.glob(os.path.join(d, "*", "result.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        out[os.path.basename(os.path.dirname(f))] = bool((r.get("eval") or {}).get("resolved"))
    return out


def mcnemar(b: int, c: int) -> float:
    """双尾精确检验。b = 只有基线对,c = 只有探针对。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2**n)


def main() -> int:
    probe_dir = sys.argv[1]
    base_dir = sys.argv[2] if len(sys.argv) > 2 else BASELINE
    probe, base = load(probe_dir), load(base_dir)
    common = sorted(set(probe) & set(base))
    if len(common) < 400:
        print(f"VERDICT=INCOMPLETE 配对任务只有 {len(common)} 道,评测没跑完")
        return 2

    np_, nb = sum(probe[t] for t in common), sum(base[t] for t in common)
    b = sum(1 for t in common if base[t] and not probe[t])   # 基线对、探针错
    c = sum(1 for t in common if probe[t] and not base[t])   # 探针对、基线错
    p = mcnemar(b, c)

    name = os.path.basename(probe_dir.rstrip("/"))
    print(f"{name:28s} {np_}/{len(common)} = {np_/len(common):.2%}   "
          f"基线 {nb}/{len(common)} = {nb/len(common):.2%}   Δ={np_-nb:+d}")
    print(f"  配对: 基线独对={b}  探针独对={c}  McNemar p={p:.4f}")

    # 停止规则(人类 10-04「明显下降就不要训了」):
    #   显著下降 = 方向向下 且 p < 0.05 —— 这是"明显"的操作化定义。
    #   方向向下但不显著 → 继续,但标 WARN,连续两次 WARN 且逐次更低才停。
    # 可检出下限(MDE):同一个 McNemar 在当前不一致对数 n=b+c 下,最小的显著净差。
    # 不是事后找借口,是为了把"没测出来"和"没有效果"分开 —— 探针涨了 5 道而 MDE 是 12,
    # 那就是**没有判断力**,不能当增益报,也不能当失败报。
    n_disc = b + c
    mde = None
    for net in range(1, n_disc + 1):
        hi, lo = (n_disc + net) // 2, (n_disc - net) // 2
        if hi + lo == n_disc and mcnemar(lo, hi) < 0.05:
            mde = net
            break
    mde_s = f"{mde:+d} 道" if mde else "不可达(不一致对太少)"
    print(f"  不一致对 n={n_disc},本次可检出下限 MDE={mde_s}")

    if np_ < nb and p < 0.05:
        print("VERDICT=STOP 显著低于基线(配对 p<0.05),按指令停训")
        return 1
    if np_ < nb:
        print(f"VERDICT=CONTINUE_WARN 低于基线 {nb-np_} 道但不显著(p={p:.3f}),在噪声量级内")
        return 0
    # 向上方向要和向下用同一把尺,否则就是"跌了算噪声、涨了算成绩"的双标。
    if np_ > nb and p < 0.05:
        print(f"VERDICT=GAIN 显著高于基线 +{np_-nb} 道 "
              f"(+{(np_-nb)/len(common):.2%},配对 p={p:.4f})")
        return 0
    if np_ > nb:
        print(f"VERDICT=CONTINUE_FLAT 高于基线 {np_-nb} 道但不显著(p={p:.3f})"
              f" —— **不能当增益报**")
        return 0
    print("VERDICT=CONTINUE_FLAT 与基线持平")
    return 0


if __name__ == "__main__":
    sys.exit(main())
