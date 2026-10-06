#!/usr/bin/env python3
"""s4 探针序列的**趋势**判据 —— 单点判不出来的东西,序列能判。

为什么要这个脚本(和 s2_verdict.py 分工):
  s2_verdict.py 回答「这一个 ckpt 比基线高吗」。它的检出下限是硬的:474 道题、
  p≈0.30 时 1σ = sqrt(474*0.3*0.7) ≈ 9.97 道,配对 McNemar 在 n_disc≈57 时
  MDE ≈ +17 道。**这个下限不会因为多训几步而下降** —— 它只取决于题量。
  10-06 的 step10 探针 145/474 比基线 134 高 11 道,p=0.1849,就死在这条线上。

  这个脚本回答「这条序列在往上走吗」。关键区别:k 个探针对**同一份固定基线**,
  基线自身的误差是一个常数偏移,**在斜率里抵消**。斜率的标准误按 OLS:
      SE(b) = s_resid / (sd(step) * sqrt(k))
  按 s_resid ≈ 9.97 道、10 步一个点、k=14 点(sd(step)≈40)估:SE ≈ 0.066 道/步,
  于是 z=2 只需要 0.13 道/步 = **「100 步涨 13 道」**。同样的 474 道题,单点要
  +17 才敢说话,序列只要 +13/100步。gsm8k 那条 z=5.70 走的就是这条路。
  ⇒ 所以 PROBE_EVERY 维持 10 不要拉长:多一个探针点比多训 5.5 步更值钱。

检验口径(汇报时别说错):
  * 分母用的是**实测残差**,不是二项公式。理由:探针是 temp=0.6 采样 + 固定
    474 道题,点到点的抖动既含采样噪声也含 ckpt 自身的抖动,这两样都是"不代表
    真实提升"的东西,正好就是该放进分母的;二项 σ 只是**规划**用的参考量。
    残差口径不需要假设噪声服从二项分布,df = k-2。
  * step 0 = 基线(s4 是从 SFT ep3 起训的,所以基线就是这条曲线的 step 0)。
    把它当锚点能把 x 跨度拉长、杠杆变大,但它也就成了 k 个点里的一个,不再是
    "抵消掉的常数"。所以两种都报:含锚点 / 只用探针点。结论要以**两种都同向**
    为准,只有一种显著就只能说"未被推翻"。
  * 逐题 id 必须完全一致才允许配对。换 harness / 换模板会在单题层面造出成片的
    伪翻转(s2_verdict.py:19-25 那条教训),所以这里硬查 id 集合相等。

用法:
  python3 s4_trend.py                      # 自动发现 val_s4_step*,基线 val_sft3_tmpl2
  python3 s4_trend.py --root DIR --baseline NAME --glob 'val_s4_step*'
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from math import comb

ROOT = "/workspace/agl-checkpoints/swe_smith_smoke"
BASELINE = "val_sft3_tmpl2"   # SFT ep3,当前模板下重测的那一份 = s4 的 step 0
PROBE_GLOB = "val_s4_step*"
N_EXPECT = 474


def load(d: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for f in glob.glob(os.path.join(d, "*", "result.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        out[os.path.basename(os.path.dirname(f))] = bool((r.get("eval") or {}).get("resolved"))
    return out


def load_dense(d: str) -> dict[str, float]:
    """逐题重建训练侧的稠密奖励 f2p_ratio,`smith_agent.py:768` 的原式:
        0.0 if (timed_out or n_p2p_ok != n_p2p) else n_f2p_pass / n_f2p
    val 评测本身记的是 binary(:1164 `reward = f2p_ratio if is_train else binary`),
    但 result.json 里 n_f2p/n_f2p_pass/n_p2p/n_p2p_ok 都在,所以能原样复算。

    **这是副指标,不是交付口径。** 交付口径是 resolved 率(SWE-bench 的标准口径)。
    而且实测它**并不降低检出下限**(10-06,step10 vs 基线):
      resolved 134→145;f2p 均值 0.4272→0.4562,配对差 +0.0290、sd 0.3816、
      SE 0.0175 → t=+1.65,换成等效道数 2σ 下限 = 16.6 道,二值口径是 ~17 道,**一样**。
    原因:稠密化让 34.0% 的题动起来(二值只有 12.0% 不一致),分子分母同时变大。
    ⇒ 它的用处只是**第二条趋势序列**做互相印证,不是「换个指标就显著了」的捷径。
    """
    out: dict[str, float] = {}
    for f in glob.glob(os.path.join(d, "*", "result.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        e = r.get("eval") or {}
        nf, nfp, np_, npo = e.get("n_f2p"), e.get("n_f2p_pass"), e.get("n_p2p"), e.get("n_p2p_ok")
        if e.get("timed_out") or not nf or (np_ is not None and npo != np_):
            ratio = 0.0
        else:
            ratio = (nfp or 0) / nf
        out[os.path.basename(os.path.dirname(f))] = ratio
    return out


def mcnemar(b: int, c: int) -> float:
    """双尾精确检验,和 s2_verdict.py:36 同一口径。b = 只有基线对,c = 只有探针对。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2**n)


def ols(xs: list[float], ys: list[float]):
    """返回 (斜率, 截距, 斜率标准误, t, 双尾 p, 残差 sd, df)。df<1 时 SE 给 nan。"""
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return (float("nan"),) * 5 + (float("nan"), 0)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    df = k - 2
    if df < 1:
        return b, a, float("nan"), float("nan"), float("nan"), float("nan"), df
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / df
    se = math.sqrt(s2 / sxx)
    t = b / se if se else float("nan")
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df))
    except Exception:                                     # 没 scipy 就退正态近似
        p = math.erfc(abs(t) / math.sqrt(2))
    return b, a, se, t, p, math.sqrt(s2), df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--glob", default=PROBE_GLOB)
    ap.add_argument("--allow-partial", action="store_true",
                    help="允许不足 474 道的目录参与(默认硬拒 —— 半份评测会把斜率带歪)")
    a = ap.parse_args()

    base_dir = os.path.join(a.root, a.baseline)
    base = load(base_dir)
    if not base:
        print(f"基线 {base_dir} 读不出任何 result.json", file=sys.stderr)
        return 2

    probes = []
    for d in sorted(glob.glob(os.path.join(a.root, a.glob))):
        if not os.path.isdir(d):
            continue
        m = re.search(r"step(\d+)", os.path.basename(d))
        if not m:
            continue
        probes.append((int(m.group(1)), d, load(d)))
    probes.sort()

    # --- 完整性 + 可配对性,先查后算 ---
    print(f"基线 {a.baseline}: {sum(base.values())}/{len(base)}")
    if len(base) != N_EXPECT:
        print(f"  !! 基线只有 {len(base)}/{N_EXPECT} 道")
        if not a.allow_partial:
            return 2
    usable = []
    for step, d, res in probes:
        name = os.path.basename(d)
        if len(res) != N_EXPECT and not a.allow_partial:
            print(f"跳过 {name}: 只有 {len(res)}/{N_EXPECT} 道(评测没跑完)")
            continue
        if set(res) != set(base):
            extra, miss = len(set(res) - set(base)), len(set(base) - set(res))
            print(f"跳过 {name}: 题目 id 和基线不一致(多 {extra} 少 {miss})—— 不能配对")
            continue
        usable.append((step, name, res))
    if not usable:
        print("没有可用探针点")
        return 2

    # --- 逐点表 ---
    nb = sum(base.values())
    print(f"\n{'step':>6} {'解出':>9} {'%':>7} {'Δ道':>6} "
          f"{'b(基线独对)':>12} {'c(探针独对)':>12} {'n_disc':>7} {'McNemar p':>10}")
    print(f"{0:>6} {f'{nb}/{len(base)}':>9} {nb/len(base)*100:>6.2f}% {'—':>6} "
          f"{'—':>12} {'—':>12} {'—':>7} {'(基线)':>10}")
    for step, name, res in usable:
        common = sorted(set(res) & set(base))
        n = sum(res[t] for t in common)
        b = sum(1 for t in common if base[t] and not res[t])
        c = sum(1 for t in common if res[t] and not base[t])
        print(f"{step:>6} {f'{n}/{len(common)}':>9} {n/len(common)*100:>6.2f}% "
              f"{n-nb:>+6} {b:>12} {c:>12} {b+c:>7} {mcnemar(b, c):>10.4f}")

    # --- 趋势 ---
    xs_p = [float(s) for s, _, _ in usable]
    ys_p = [float(sum(r.values())) for _, _, r in usable]
    p_hat = sum(ys_p) / len(ys_p) / N_EXPECT
    sig_binom = math.sqrt(N_EXPECT * p_hat * (1 - p_hat))

    print(f"\n二项参考量(只用于规划,不用于判决):p̄={p_hat:.4f} → 1σ={sig_binom:.2f} 道/次探针")
    for tag, xs, ys in (("只用探针点", xs_p, ys_p),
                        ("含 step0 锚点", [0.0] + xs_p, [float(nb)] + ys_p)):
        k = len(xs)
        if k < 3:
            print(f"{tag}: 只有 {k} 个点,df={k-2} 不足以给斜率标准误 —— 等更多探针")
            # 仍然把方向报出来,方便看着走
            if k == 2:
                print(f"  方向: {(ys[1]-ys[0])/(xs[1]-xs[0])*100:+.2f} 道/100步(无显著性)")
            continue
        b, a0, se, t, p, sres, df = ols(xs, ys)
        mde = 2 * se * 100 if se == se else float("nan")
        verdict = ("趋势显著上行" if (t == t and t >= 2 and p < 0.05) else
                   "趋势显著下行" if (t == t and t <= -2 and p < 0.05) else
                   "未被推翻(不能当增益报)")
        print(f"{tag}(k={k}, df={df}):")
        print(f"  斜率 {b*100:+.2f} 道/100步   截距 {a0:.1f} 道")
        print(f"  残差 sd {sres:.2f} 道(二项参考 {sig_binom:.2f});SE(斜率) {se*100:.3f} 道/100步")
        print(f"  t={t:+.2f}  双尾 p={p:.4g}  →  {verdict}")
        print(f"  当前检出下限(2σ): 斜率需 ≥ {mde:.2f} 道/100步")
    # --- 副指标:稠密 f2p_ratio 的同一套趋势(只作互相印证,见 load_dense 的长注)---
    bd = load_dense(base_dir)
    print("\n—— 副指标 f2p_ratio(训练侧稠密奖励口径;**不是交付口径**,"
          "实测不降低检出下限)——")
    mb = sum(bd.values()) / len(bd)
    print(f"{'step':>6} {'f2p均值':>9} {'Δ':>8} {'配对差 sd':>10} {'t':>7}")
    print(f"{0:>6} {mb:>9.4f} {'—':>8} {'—':>10} {'(基线)':>7}")
    xs_d, ys_d = [], []
    for step, name, _ in usable:
        pd_ = load_dense(os.path.join(a.root, name))
        common = sorted(set(pd_) & set(bd))
        if len(common) != len(bd):
            continue
        mp = sum(pd_[t] for t in common) / len(common)
        diffs = [pd_[t] - bd[t] for t in common]
        md = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((x - md) ** 2 for x in diffs) / len(diffs))
        se = sd / math.sqrt(len(diffs))
        print(f"{step:>6} {mp:>9.4f} {md:>+8.4f} {sd:>10.4f} {md/se if se else float('nan'):>+7.2f}")
        xs_d.append(float(step))
        ys_d.append(mp)
    if len(xs_d) >= 3:
        b, a0, se, t, p, sres, df = ols([0.0] + xs_d, [mb] + ys_d)
        print(f"趋势(含 step0 锚点, k={len(xs_d)+1}, df={df}): 斜率 {b*100:+.4f}/100步  "
              f"t={t:+.2f}  p={p:.4g}")
    else:
        print(f"(只有 {len(xs_d)} 个探针点,不给斜率显著性)")

    print("\n口径提醒:上行趋势显著 ≠ 某一个 ckpt 显著高于基线。要说「某个 ckpt 涨了」\n"
          "仍然得用 s2_verdict.py 的配对检验,并且 474 道的 MDE ≈ +17 道。\n"
          "副指标 f2p_ratio 只能作印证,汇报一律以 resolved 率为准。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
