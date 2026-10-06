"""按 s1 实测通过率筛训练集（离线 curriculum，DAPO dynamic sampling 的免费版）。

为什么离线做：DAPO 的 dynamic sampling 是在线重采到填满 batch，要多生成 1.67 倍
轨迹；而生成正是我们的瓶颈（一步 40–70 分钟）。离线筛一次，效果一样，不花 GPU。

难度标签来自 s1 的 90 步 × 16 题 × 4 条 = 5736 条 rollout。注意 s1 的策略几乎没动
（entropy 0.24842 → 0.25363），所以这批标签实际上是**基座策略**的难度，正是
curriculum 要的东西；但策略一旦真的提升，"从来做不对"的那批要重新筛。
"""
import glob, json, collections, sys

TRAJ = "/workspace/agl-checkpoints/smith_rl_s1/trajectories/step_*_train.jsonl"
SRC  = "train_dataset_mixed.jsonl"
OUT  = "train_dataset_screened.jsonl"
RATES = "task_pass_rates_s1.json"
MARK = "Consider the following PR description:"


def stmt_of_prompt(p):
    """从累积对话里取回第一条 user 消息中的 problem_statement 原文。"""
    s = p.find(MARK)
    if s < 0:
        return None
    s += len(MARK)
    e = p.find("</pr_description>", s)
    return p[s:e].strip() if e > 0 else None


# 1. 逐题通过率
pass_by_stmt = collections.defaultdict(list)
for f in sorted(glob.glob(TRAJ)):
    for ln in open(f):
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        st = stmt_of_prompt(d["prompt"])
        if st:
            pass_by_stmt[st].append(1 if d["reward"] > 0 else 0)

print("从轨迹里取回 %d 道不同的题" % len(pass_by_stmt))

# 2. 对齐到训练集行
rows = [json.loads(l) for l in open(SRC)]
print("训练集 %d 行" % len(rows))
by_stmt = {r["problem_statement"].strip(): r for r in rows}

matched, keep, rates = 0, [], {}
for st, v in pass_by_stmt.items():
    r = by_stmt.get(st)
    if r is None:
        continue
    matched += 1
    pr = sum(v) / len(v)
    rates[r["instance_id"]] = {"pass_rate": pr, "n": len(v)}
    if 0 < pr < 1:
        keep.append(r)

print("对齐成功 %d / %d  (%.1f%%)" % (matched, len(pass_by_stmt),
                                      100 * matched / max(1, len(pass_by_stmt))))
if matched < 0.9 * len(pass_by_stmt):
    print("!! 对齐率过低，先别用，去查 problem_statement 是否被改写过")
    sys.exit(1)

n0 = sum(1 for v in rates.values() if v["pass_rate"] == 0)
n1 = sum(1 for v in rates.values() if v["pass_rate"] == 1)
print("  pass=0   %4d 道 (%.1f%%)  —— 组内全错，梯度恒为 0" % (n0, 100 * n0 / matched))
print("  pass=1   %4d 道 (%.1f%%)  —— 组内全对，梯度恒为 0" % (n1, 100 * n1 / matched))
print("  0<pass<1 %4d 道 (%.1f%%)  —— 留下" % (len(keep), 100 * len(keep) / matched))

with open(OUT, "w") as fh:
    for r in keep:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(RATES, "w") as fh:
    json.dump(rates, fh, indent=1)
print("\n写出 %s  (%d 行)" % (OUT, len(keep)))
print("写出 %s  (%d 道题的通过率，供下一轮重筛)" % (RATES, len(rates)))
