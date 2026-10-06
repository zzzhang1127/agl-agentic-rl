import json, glob, hashlib, re, os, statistics as st
from collections import defaultdict

D   = "/workspace/agl-checkpoints/smith_rl_s1"
LOG = D + "/agl-logs/new/agl-rollout-%s.log"
RE_F2P = re.compile(r"FAIL_TO_PASS (\d+)/(\d+) passed")
RE_P2P = re.compile(r"PASS_TO_PASS (\d+)/(\d+) ok")
RE_TMO = re.compile(r"EVAL TIMEOUT after")
# job-template-smith.yaml:93-97
T0, LAM, MAXT = 32, 0.1, 40
SOFT, HARD, MAXP = 39000, 47104, 0.1

def turn_pen(r, n):
    if r < 1.0 or n <= T0: return r
    return 1.0 - LAM * min((n - T0) / max(1, MAXT - T0), 1.0)
def prompt_pen(r, pt, solved):
    if not solved or pt <= SOFT: return r
    if pt >= HARD: return r - MAXP
    return r - MAXP * min((pt - SOFT) / max(1, HARD - SOFT), 1.0)

def key(p):
    s = p.find("<|im_start|>user"); e = p.find("<|im_end|>", s)
    return hashlib.md5(p[s:e].encode()).hexdigest()

rows = []
for f in sorted(glob.glob(D + "/trajectories/step_*_train.jsonl")):
    for line in open(f):
        r = json.loads(line)
        lp = LOG % r["rollout_id"]
        if not os.path.exists(lp): continue
        txt = open(lp, errors="replace").read()
        m = RE_F2P.findall(txt)
        if not m: continue
        n, d = int(m[-1][0]), int(m[-1][1])
        p2p = RE_P2P.findall(txt)
        p2p_ok = (int(p2p[-1][0]) == int(p2p[-1][1])) if p2p else True
        tmo = bool(RE_TMO.search(txt))
        ratio = 0.0 if (tmo or not p2p_ok or not d) else n / d
        solved = ratio >= 1.0
        new = prompt_pen(turn_pen(ratio, r["n_turns"]), r["prompt_tokens"], solved)
        rows.append(dict(k=key(r["prompt"]), old=r["reward"], new=new, n=n, d=d,
                         turns=r["n_turns"], tok=r["resp_tokens"], solved=solved))

g = defaultdict(list)
for x in rows: g[x["k"]].append(x)
g4 = {k: v for k, v in g.items() if len(v) >= 2}
N = len(g4)
def zv(v): return len(set(round(x, 6) for x in v)) == 1

zo = sum(1 for v in g4.values() if zv([x["old"] for x in v]))
zn = sum(1 for v in g4.values() if zv([x["new"] for x in v]))
print(f"rollout {len(rows)}  可比组 {N}")
print(f"零方差组   二值: {zo} ({zo/N:.1%})   F2P比例: {zn} ({zn/N:.1%})")
print(f"有梯度组   {N-zo} ({1-zo/N:.1%}) -> {N-zn} ({1-zn/N:.1%})  = ×{(N-zn)/(N-zo):.2f}")
# 一致性检查:新口径不该把原本有梯度的组变成零方差
broke = [k for k,v in g4.items() if not zv([x["old"] for x in v]) and zv([x["new"] for x in v])]
print(f"被新口径「弄没」的组: {len(broke)}  (应为 0)")
af = [v for v in g4.values() if all(x["old"]==0.0 for x in v)]
rec = [v for v in af if not zv([x["new"] for x in v])]
print(f"四条全错组 {len(af)} -> 恢复方差 {len(rec)} ({len(rec)/len(af):.1%})")

fails = [x for x in rows if x["old"]==0.0]
part = [x for x in fails if x["new"]>0]
print(f"失败 rollout {len(fails)} 条, {len(part)} 条({len(part)/len(fails):.1%})拿到部分分, 均值 {st.mean([x['new'] for x in part]):.3f}")
print(f"奖励均值 {st.mean([x['old'] for x in rows]):.3f} -> {st.mean([x['new'] for x in rows]):.3f}")

# 长度捷径:组内高于均值 vs 低于均值的 response token 差(可和 -1208 对比)
def shortcut(f):
    hi, lo = [], []
    for v in g4.values():
        vals = [f(x) for x in v]
        if zv(vals): continue
        mu = st.mean(vals)
        for x in v: (hi if f(x) > mu else lo).append(x["tok"])
    return st.mean(hi) - st.mean(lo)
print(f"\n长度捷径(高分组-低分组 response token): 二值 {shortcut(lambda x:x['old']):+.0f}  ->  比例 {shortcut(lambda x:x['new']):+.0f}")

# 决定性问题:新恢复的全错组里,部分分高的那条是更长还是更短?
hi, lo, nh = [], [], 0
for v in af:
    vals = [x["new"] for x in v]
    if zv(vals): continue
    mu = st.mean(vals); nh += 1
    for x in v: (hi if x["new"] > mu else lo).append(x["tok"])
print(f"\n== 新恢复的 {nh} 个全错组 ==")
print(f"部分分高于组均值: {st.mean(hi):.0f} token   低于: {st.mean(lo):.0f} token   差 {st.mean(hi)-st.mean(lo):+.0f}")
# 失败 rollout 里 ratio 和长度的相关
fp = [x for x in rows if x["old"]==0.0 and x["d"]>0]
import math
xs=[x["n"]/x["d"] for x in fp]; ys=[x["tok"] for x in fp]
mx,my=st.mean(xs),st.mean(ys)
cov=sum((a-mx)*(b-my) for a,b in zip(xs,ys)); sx=math.sqrt(sum((a-mx)**2 for a in xs)); sy=math.sqrt(sum((b-my)**2 for b in ys))
print(f"失败 rollout 内 ratio~token 相关 r = {cov/(sx*sy):+.3f}  (n={len(fp)})")
z=[x for x in fp if x["n"]==0]; nz=[x for x in fp if x["n"]>0]
print(f"  0 分失败 {len(z)} 条 中位 {st.median([x['tok'] for x in z]):.0f} token / {st.median([x['turns'] for x in z]):.0f} 轮")
print(f"  有分失败 {len(nz)} 条 中位 {st.median([x['tok'] for x in nz]):.0f} token / {st.median([x['turns'] for x in nz]):.0f} 轮")
