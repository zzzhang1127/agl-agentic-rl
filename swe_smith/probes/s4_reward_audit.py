#!/usr/bin/env python
"""s4 训练侧奖励结构审计:「早交卷」在**组内**到底值不值钱?

§62.2 里我断言「跑满 40 轮没交卷 = 奖励 0;早交一个像样的补丁 = 大概率拿部分分,
于是早交的期望奖励更高」。那是**从奖励公式推出来的**,不是从 s4 自己的数据量出来的。
这个脚本用 s4 那 21 步的 rollout 日志把它验一遍。

为什么看「组内」而不是全局:GRPO 的优势是 (r - 组均值)/组标准差,梯度只认同一道题
那 4 条 rollout 之间的相对高低。全局相关会被「题目难度」整个吃掉。

**和 §62.4 那个方法错误的区别(重要,别又踩一次)**:
这里我**不**声称「少写 token 导致解不出」之类的因果。组内 corr(轮数, 奖励) < 0
只说明一件事:*梯度把概率质量推向更短的轨迹*。GRPO 不做因果推断 —— 它只是把
高优势 rollout 里出现过的 token 序列概率抬高。所以哪怕「简单题又快又对」是全部
原因,结论仍然成立:这个奖励结构在**机械地**奖励短轨迹。
这是对梯度方向的陈述,不是对「短是否有害」的陈述。

口径:
  * 只取 mode=train(val 不参与梯度)。
  * 组 = 同一个 instance,且起始时间相邻(间隔 > GAP 秒就切成新组)—— 同一步里
    同一道题的 4 条几乎同时起,跨步撞车很少但要切开。
  * 用 reward(整形后、真正进 advantage 的那个),同时并排打 raw_reward 做对照。
"""
from __future__ import annotations

import collections
import datetime as dt
import pathlib
import re
import statistics as st
import sys

D = pathlib.Path("/workspace/agl-checkpoints/smith_rollout_logs/new")
SINCE = dt.datetime(2026, 10, 6, 1, 30)   # s4 起训 01:39
GAP = 1200                                 # 20min 内算同一步

RE_START = re.compile(r"^([\d-]+ [\d:,]+) INFO SmithAgent start: instance=(\S+) max_turns=(\d+)")
RE_DONE = re.compile(
    r"done: mode=(\w+) submitted=(\w+) patch=(\d+)B reward=([-\d.]+) raw_reward=([-\d.]+) "
    r"binary=([\d.]+) f2p_ratio=([\d.]+) n_turns=(\d+) max_prompt_tokens=(\d+) reason=(.*)"
)
RE_REASON = re.compile(r"FAIL_TO_PASS (\d+)/(\d+) passed, PASS_TO_PASS (\d+)/(\d+) ok")


def parse(p: pathlib.Path) -> dict | None:
    inst = ts = None
    rec = None
    try:
        txt = p.read_text(errors="replace")
    except OSError:
        return None
    for line in txt.splitlines():
        if inst is None:
            m = RE_START.match(line)
            if m:
                ts = dt.datetime.strptime(m.group(1).split(",")[0], "%Y-%m-%d %H:%M:%S")
                inst = m.group(2)
                continue
        m = RE_DONE.search(line)
        if m:
            rec = m
    if inst is None or rec is None:
        return None
    r = {
        "inst": inst, "ts": ts, "mode": rec.group(1),
        "submitted": rec.group(2) == "True", "patch": int(rec.group(3)),
        "reward": float(rec.group(4)), "raw": float(rec.group(5)),
        "binary": float(rec.group(6)), "f2p": float(rec.group(7)),
        "turns": int(rec.group(8)), "ptok": int(rec.group(9)),
    }
    mr = RE_REASON.search(rec.group(10))
    if mr:
        r["f2p_pass"], r["f2p_tot"] = int(mr.group(1)), int(mr.group(2))
        r["p2p_ok"], r["p2p_tot"] = int(mr.group(3)), int(mr.group(4))
        r["p2p_broke"] = r["p2p_ok"] != r["p2p_tot"]
    else:
        r["p2p_broke"] = None
    return r


def groups(rows: list[dict]) -> list[list[dict]]:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["inst"]].append(r)
    out = []
    for inst, rs in by.items():
        rs.sort(key=lambda x: x["ts"])
        cur = [rs[0]]
        for a, b in zip(rs, rs[1:]):
            if (b["ts"] - a["ts"]).total_seconds() > GAP:
                out.append(cur); cur = [b]
            else:
                cur.append(b)
        out.append(cur)
    return out


def fe_slope(gs: list[list[dict]], xk: str, yk: str, sub=None):
    """组内去均值的最小二乘斜率(固定效应)+ t。sub 过滤单条 rollout。"""
    xs, ys = [], []
    for g in gs:
        rs = [r for r in g if sub is None or sub(r)]
        if len(rs) < 2:
            continue
        mx = st.mean(float(r[xk]) for r in rs)
        my = st.mean(float(r[yk]) for r in rs)
        for r in rs:
            xs.append(float(r[xk]) - mx); ys.append(float(r[yk]) - my)
    n = len(xs)
    if n < 3 or st.pstdev(xs) == 0:
        return None
    sxx = sum(v * v for v in xs)
    b = sum(x * y for x, y in zip(xs, ys)) / sxx
    resid = [y - b * x for x, y in zip(xs, ys)]
    s2 = sum(e * e for e in resid) / (n - 2)
    se = (s2 / sxx) ** 0.5
    return b, se, b / se if se else float("nan"), n


def paired_contrast(gs: list[list[dict]], flag, yk="reward"):
    """组内 E[y|flag] - E[y|not flag],只用同时含两类的组,配对 t。"""
    d = []
    for g in gs:
        a = [float(r[yk]) for r in g if flag(r)]
        b = [float(r[yk]) for r in g if not flag(r)]
        if a and b:
            d.append(st.mean(a) - st.mean(b))
    if len(d) < 3:
        return None
    m = st.mean(d); se = st.stdev(d) / len(d) ** 0.5
    return m, se, m / se if se else float("nan"), len(d)


def fmt(res, unit=""):
    if res is None:
        return "样本不足"
    b, se, t, n = res
    star = "**显著**" if abs(t) > 2 else "不显著"
    return f"{b:+.4f}{unit} (SE {se:.4f}, t={t:+.2f}, n={n}) {star}"


def main() -> int:
    files = [p for p in D.glob("agl-rollout-*.log")
             if dt.datetime.fromtimestamp(p.stat().st_mtime) >= SINCE]
    rows = [r for r in (parse(p) for p in files) if r and r["mode"] == "train" and r["ts"]]
    print(f"s4 窗口({SINCE:%m-%d %H:%M} 起)训练 rollout: {len(rows)} 条 / 扫了 {len(files)} 个日志")
    if not rows:
        return 1
    gs = [g for g in groups(rows) if len(g) >= 2]
    sizes = collections.Counter(len(g) for g in gs)
    print(f"组: {len(gs)} 个(大小分布 {dict(sorted(sizes.items()))});"
          f" 有奖励方差的组 {sum(1 for g in gs if st.pstdev([r['reward'] for r in g]) > 0)} 个")

    print("\n=== 1. 行为画像(全体训练 rollout)===")
    n = len(rows)
    sub = [r for r in rows if r["submitted"]]
    print(f"  交卷率 {len(sub)/n*100:.1f}%  顶满40轮 {sum(r['turns']>=40 for r in rows)/n*100:.1f}%  "
          f"轮数中位 {st.median(r['turns'] for r in rows):.0f}")
    print(f"  做对(binary=1) {sum(r['binary']>0 for r in rows)/n*100:.1f}%  "
          f"破P2P {sum(bool(r['p2p_broke']) for r in rows)/n*100:.1f}%  "
          f"拿到部分分(0<f2p<1) {sum(0<r['f2p']<1 for r in rows)/n*100:.1f}%")
    # 整形项到底开没开火:**必须逐条比 reward 和 raw**,不能看均值差。
    # 10-06 我第一版就是看了一眼均值(0.5943 vs 0.6073)写下「两者相等 ⇒ 整形没生效」,
    # 然后才发现真实阈值是 job-template-smith-s4.yaml 的 T0=32(不是代码默认 80),
    # 轮数罚一直活着 —— 而它正是组内唯一的长度信号。别再看均值了。
    shaped = [r for r in rows if abs(r["reward"] - r["raw"]) > 1e-9]
    print(f"  奖励均值 {st.mean(r['reward'] for r in rows):.4f}(raw {st.mean(r['raw'] for r in rows):.4f})")
    print(f"  整形项开火 {len(shaped)} 条 ({len(shaped)/n*100:.1f}%),平均扣 "
          f"{st.mean(r['raw']-r['reward'] for r in shaped) if shaped else 0:.4f},"
          f"最大扣 {max((r['raw']-r['reward'] for r in shaped), default=0):.4f};"
          f"全部落在做对的 rollout 上 {sum(r['binary']>0 for r in shaped)}/{len(shaped)}")

    print("\n=== 2. 交卷在组内值多少钱(这是梯度真正看到的)===")
    for yk, lab in (("reward", "奖励"), ("f2p", "f2p_ratio"), ("binary", "做对")):
        print(f"  {lab:10s} 交卷 − 没交卷 = {fmt(paired_contrast(gs, lambda r: r['submitted'], yk))}")

    print("\n=== 3. 轮数在组内值多少钱(每多跑 1 轮,奖励变化)===")
    print(f"  全体       {fmt(fe_slope(gs, 'turns', 'reward'), '/轮')}")
    print(f"  只看交卷的 {fmt(fe_slope(gs, 'turns', 'reward', sub=lambda r: r['submitted']), '/轮')}")
    print(f"  只看没交的 {fmt(fe_slope(gs, 'turns', 'reward', sub=lambda r: not r['submitted']), '/轮')}")

    print("\n=== 4. 「早交拿部分分」这条近路有多宽 ===")
    ps = [r for r in rows if r["submitted"] and 0 < r["f2p"] < 1]
    zs = [r for r in rows if not r["submitted"]]
    ok = [r for r in rows if r["binary"] > 0]
    print(f"  没交卷:        {len(zs):4d} 条,奖励均值 {st.mean(r['reward'] for r in zs) if zs else 0:.4f}")
    print(f"  交了但只拿部分分:{len(ps):4d} 条,奖励均值 {st.mean(r['reward'] for r in ps) if ps else 0:.4f}"
          f"  轮数中位 {st.median(r['turns'] for r in ps) if ps else 0:.0f}")
    print(f"  做对:          {len(ok):4d} 条,奖励均值 {st.mean(r['reward'] for r in ok) if ok else 0:.4f}"
          f"  轮数中位 {st.median(r['turns'] for r in ok) if ok else 0:.0f}")
    # 近路的宽度 = 部分分那一档比"没交卷"高出多少、又占多少比例
    if ps and zs:
        print(f"  ⇒ 近路落差 {st.mean(r['reward'] for r in ps) - st.mean(r['reward'] for r in zs):+.4f},"
              f" 覆盖 {len(ps)/n*100:.1f}% 的 rollout")

    print("\n=== 5. 组内方差来自哪里(GRPO 按组标准化 ⇒ 每个有方差的组权重相当)===")
    sd = lambda g, k: st.pstdev([r[k] for r in g])
    zero, shaped_only, task = [], [], []
    for g in gs:
        (zero if sd(g, "reward") == 0 else shaped_only if sd(g, "raw") == 0 else task).append(g)
    print(f"  全零方差(被动态采样丢掉)   {len(zero):4d} ({len(zero)/len(gs)*100:.1f}%)")
    print(f"  **只有整形项造出方差**      {len(shaped_only):4d} ({len(shaped_only)/len(gs)*100:.1f}%)"
          f" —— 这些组的梯度 100% 是「同样做对,少跑几轮」")
    print(f"  任务信号造出方差            {len(task):4d} ({len(task)/len(gs)*100:.1f}%)")
    if shaped_only:
        print(f"  纯整形组里「全做对」的 {sum(all(r['binary']>0 for r in g) for g in shaped_only)}"
              f"/{len(shaped_only)};若没有整形项,它们会被动态采样当零方差丢掉"
              f" ⇒ **整形项把它们捞回了训练**,占有梯度组的 {len(shaped_only)/max(1,len(gs)-len(zero))*100:.1f}%")
    print("  组内、只在**同样做对**的 rollout 之间,每多跑 1 轮:")
    print(f"    整形后 reward {fmt(fe_slope(gs, 'turns', 'reward', sub=lambda r: r['binary'] > 0), '/轮')}")
    print(f"    整形前 raw    {fmt(fe_slope(gs, 'turns', 'raw', sub=lambda r: r['binary'] > 0), '/轮')}"
          f"  ← 若为 0 ⇒ 这条长度信号**全部**来自轮数罚")
    print(f"  组内、只在同样没做对的之间: {fmt(fe_slope(gs, 'turns', 'reward', sub=lambda r: r['binary'] == 0), '/轮')}")

    print("\n=== 6. 破 P2P 的硬闸还在不在 ===")
    br = [r for r in rows if r["p2p_broke"]]
    print(f"  破 P2P {len(br)} 条,其中 f2p_ratio>0 的 {sum(r['f2p']>0 for r in br)} 条"
          f"({'硬闸生效' if not any(r['f2p']>0 for r in br) else '硬闸漏了!'})")

    print("\n口径提醒:这些都是**训练集**上的数,和交付口径(val 474 的 resolved)无关;"
          "\n第 3 节的斜率是对**梯度方向**的陈述,不是「短轨迹有害」的因果结论(见 §62.4)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
