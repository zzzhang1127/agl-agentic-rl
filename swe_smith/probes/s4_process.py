#!/usr/bin/env python
"""探针的「过程指标」对照 —— 不是交付物,是机理印证。

为什么要看它:resolved/474 的单点 1σ≈10 道,**结构上**没有判断力(见 s4_trend.py)。
但 submitted / n_turns / overflowed / timed_out 这些是每道题都有的过程量,
配对后方差小得多,能在 resolved 还看不出名堂时就告诉我「策略到底有没有往
我们训的方向动」。它**不能**替 resolved 当结论(没人会因为多提交了补丁就算解出来),
只能回答一个问题:RL 是在改变行为,还是根本没动。
"""
import json, pathlib, sys, math, statistics as st

ROOT = pathlib.Path("/workspace/agl-checkpoints/swe_smith_smoke")

def load(name):
    out = {}
    for f in sorted(ROOT.glob(f"{name}/*/result.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("n_turns") is None:      # 没跑起来的不算
            continue
        out[d["instance_id"]] = d
    return out

def pct(x, n):
    return f"{100*x/n:5.1f}%" if n else "    —"

def paired(base, cur, key, fn):
    """配对 t 检验。返回 (基线均值, 探针均值, 差, t)。"""
    ids = sorted(set(base) & set(cur))
    d = [fn(cur[i]) - fn(base[i]) for i in ids]
    b = [fn(base[i]) for i in ids]; c = [fn(cur[i]) for i in ids]
    if len(d) < 3 or st.pstdev(d) == 0:
        return st.mean(b), st.mean(c), st.mean(d), float("nan")
    se = st.stdev(d) / math.sqrt(len(d))
    return st.mean(b), st.mean(c), st.mean(d), st.mean(d) / se if se else float("nan")

BASE = sys.argv[1] if len(sys.argv) > 1 else "val_sft3_tmpl2"
others = sys.argv[2:] or sorted(
    p.name for p in ROOT.glob("val_s4_step*")
    if len(list(p.glob("*/result.json"))) == 474)

base = load(BASE)
print(f"基线 {BASE}: {len(base)} 道有效\n")
F = [
    ("提交了补丁",   lambda d: 1.0 if d.get("submitted") else 0.0, "%"),
    ("超时",         lambda d: 1.0 if d.get("timed_out") else 0.0, "%"),
    ("上下文溢出",   lambda d: 1.0 if d.get("overflowed") else 0.0, "%"),
    ("顶满 40 轮",   lambda d: 1.0 if (d.get("n_turns") or 0) >= 40 else 0.0, "%"),
    ("轮数",         lambda d: float(d.get("n_turns") or 0), ""),
    ("补丁字符数",   lambda d: float((d.get("eval") or {}).get("patch_chars") or 0), ""),
    ("生成 token",   lambda d: float(d.get("vllm_completion_tokens") or 0), ""),
    ("解出(resolved)", lambda d: 1.0 if (d.get("eval") or {}).get("resolved") else 0.0, "%"),
]
for name in others:
    cur = load(name)
    ids = set(base) & set(cur)
    print(f"=== {name}({len(cur)} 道有效,与基线配对 {len(ids)} 道)===")
    print(f"  {'指标':<14}{'基线':>10}{'探针':>10}{'差':>10}{'t':>8}")
    for label, fn, unit in F:
        mb, mc, md, t = paired(base, cur, label, fn)
        if unit == "%":
            print(f"  {label:<14}{100*mb:>9.1f}%{100*mc:>9.1f}%{100*md:>+9.1f}%{t:>8.2f}")
        else:
            print(f"  {label:<14}{mb:>10.1f}{mc:>10.1f}{md:>+10.1f}{t:>8.2f}")
    print("  |t|>2 才算动了;resolved 那行只作参照,判决一律用 s4_trend.py 的 McNemar/趋势。\n")
