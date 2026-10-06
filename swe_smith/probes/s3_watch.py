#!/usr/bin/env python3
"""s3 坍塌探针(Tier 1)。每 POLL 秒读一次 trainer 日志和 rollout 日志,越线就暂停训练。

为什么要这个:s2 坍塌时**我盯的那个"奖励"没报警** —— 而那是我盯错了指标。
10-05 复查,实测 5 步:

    step                1       2       3       4       5
    response_len      274.5   254.8   112.5    60.4    91.3   <- 坍塌
    training/reward   0.548   0.580   0.138   0.147   0.235   <- **同步掉 4.2 倍**
    critic/score/mean 0.434   0.563   0.396   0.432   0.439   <- 全程正常,骗了我

**`critic/score/mean` 不能当奖励看**,三条独立理由:

  a bf16 量化。s1 九十一步读数**无例外**全部落在 k/2^n 网格上(分母 128:32 次、
    256:32 次、512:27 次),0.6 附近分辨率只有 ~1/512;`training/reward` 是 float64
    (20 个读数里 18 个离网格,例如 0.6219277828052747)。两步读数相同可能只是
    同一个 bf16 桶,**不是指标冻住了**。
  b 行加权,不是 rollout 加权。轮数多的 rollout 占更大权重,而它们解得更差。
  c **过滤后选择偏差**。它只看得见"有方差"的组 —— 全对组和全错组被系统性排除,
    偏差随丢弃率增长。实测 score/mean − reward 中位:s1 +0.0479 / s2 **+0.2045**
    / s3 −0.0237。s2 那 +0.2045 就是把 4.2 倍跌幅压成 1.4 倍的原因。

但**反过来也不能只看 reward**:健康的 s1 自己就会掉到 0.2137~0.2768(第 3/8/12 步),
和 s2 的坍塌值 0.1375/0.1465 **区间重叠** —— 绝对门槛会在健康血统上误报。所以判据 7
用相对形式(见下),而**分得最开的仍然是长度**。两条独立开火、同一步命中,是冗余
不是替代。

阈值全部按 s1(健康 91 步)和 s2(坍塌 5 步)的实测值标定,下面每条都写了分离度。

判据(都要求**连续**命中,单步抖动不算 —— s1 九十一步里有 8 次单步 <0.80):

 1 长度绝对坍塌  rl < 0.50 × B,B = 前 2 步均值,连续 2 步
     s2: B=264.6 → 门槛 132.3;step3 112.5 ✓ step4 60.4 ✓ → **step4 触发**
     s1: B=177.8 → 门槛 88.9;91 步最低 115.1 → 不触发,余量 1.29×
 2 长度缓慢漂移  rl / 前5步均值 < 0.55,连续 2 步
     s1 九十一步单步最低 0.568,连一次都没破 0.55;s2 step4 是 0.282
 3 零方差组回升  zero_adv/seen > 0.60,连续 3 步
     s1 均值 41.0% 中位 43.8% 单步最高 62.5%;F2P 部分分后预期 ~30%,
     所以连续 3 步 >60% 说明部分分没生效或者模型在往同质化走
 4 采集率     capture < 0.90,连续 2 步(= AGL_MIN_CAPTURE_RATE 的值)
     s1 中位 1.000 最低 0.938(Go 镜像整组掉 4 条)
 5 训推偏差    rollout_is_mean 跑出 [0.85, 1.15],连续 2 步
     s1 实测 0.9903~0.9970,这项是安全网不是主动干预
 6 梯度爆炸    grad_norm 是 NaN/inf,或 > 10 × 前 10 步中位,连续 2 步
 7 奖励坍塌    training/reward < 0.40 × B,连续 2 步(B = 前 2 步均值)
     s2: B=0.564 → 门槛 0.226;step3 0.138 ✓ step4 0.147 ✓ → **step4 触发**,余量 1.54×
     s1: B=0.501 → 门槛 0.200;91 步里"连续两步中较高者"最低 0.3008(第 11-12 步)
         → 余量 1.50×,而且单步一次都没跌破过
     k 为什么取 0.40 不取 0.50:0.50 时 s1 侧余量只剩 1.20×(且有 2 次孤立单步跌破,
     全靠"连续"挡住),0.40 两侧 1.50/1.54 最对称 —— 比长度判据本身(1.66/1.18)还均衡。
     **原来写的是「score/mean == 0 连续 3 步」,结构上永不可能开火**:留存组按定义
     有奖励方差、分数非负,过滤后的均值几乎不可能恰好为 0;真要全组零方差,
     `_collect_train_batch` 返回 None,这一步根本不训练。属 §50 同一类静默失效 ——
     看着像覆盖,其实永远不响。
 8 进程没了    pgrep 找不到 trainer
10 日志路径错   trainer 活着但 TRAINER_LOG 不存在(换血统时忘了改路径 = 一路假 ok)
 9 合并断链   training/n_trace_merge_mismatch_rows >= MM_STOP(默认 4 行),单步即开火
91 合并断链(低档)  0 < mismatch_rows < MM_STOP —— **只告警,不停机,探针不退出**
     为什么要分档(2026-10-06 订正):下面"正常值恒为 0"这句被 s4 证伪了。
     s4 前 17 步 1152 条 rollout 全 0,第 18 步出了 1 行,残余率 1/1152 = 0.09%。
     查实:`diverge_kind='token_differs'`、分叉处 expected/actual 两段解码文本
     **逐字完全相同**、`turn_index=39`(40 轮上限的最后一轮)。把那段文本按 39 个
     切点拆开实测,`enc(A)+enc(B) != enc(A+B)` 在 24/39 = 62% 的切点成立;而
     `decode(encode(W))==W` 且 encode 幂等 ⇒ 不是有损往返、不是模板问题,是 BPE
     切分对接缝上下文敏感,长 episode 本就会偶发。这一行的影响面:那一步
     `max_rows_per_rollout=1`、48 行训练行里最多 1 行(2.1%)带错前缀,且只此一步。
     ⇒ 「>0 即停机」会为一个不改变"继续训"这个决定的观察把训练搁死
     (WATCH_ALARM.json 一在,s4_cycle.sh resume 就 exit 6),10-06 实际吃过这个亏。
     门槛没有把信号藏起来:每一步的断链数照旧打在 ok/warn 行里,巡检每次都读。
     要盯的是它**会不会长大**,不是它在不在。
     这一条不是统计门槛,是**结构不变量**:它只在 `rollout_adapter.py:756` 的
     「第 k+1 轮 prompt 以第 k 轮 (prompt+response) 为前缀」判据**不成立**时才累加
     (:816),合法的预算切分走的是另一条分支(:789 `n_budget_splits`)。所以
     健康值是 0 或个别零头(见 91 的订正),成**片**出现才说明 episode 被打碎成
     一轮一行、每行各自带整条 episode 的最终奖励 —— 多轮 credit assignment 没在跑,
     `enable_rollout_level_advantage` + `loss_mode=per_rollout_mean` 收到的不是合并序列。
     **s1 和 s2 两份日志上它都是满的(64/64,91 步 + 5 步无一例外),
     我盯了九十一步没看它一眼** —— 所以这一条在 selftest 里要求**两份都开火**
     (`STRUCTURAL`),而不像判据 1/7 那样要求"s1 不开火":s1 在坍塌意义上健康,
     在合并意义上从第一步就是坏的。病根见 §56/§57(AGL proxy 丢 `chat_template_kwargs`)。
     口径注:`n_unmerged_rollouts`(= 产出 >1 行的 rollout 数)会把合法预算切分
     也算进去,所以用 mismatch_rows 做判据、unmerged 只作参照。

命中后的动作:写 WATCH_ALARM.json + 给 trainer 发 SIGINT(**存盘后**退出,
resume_mode=auto 可原地续)。不改配方 —— 按巡检纪律第 7 条,指标异常一律
「停下、写清观察和判断、等人」。

安全:只对 cmdline 里同时含 train_smith_agent.py 和 --train-dataset-path 指向
s3 数据集的进程发信号,绝不 pkill 模式匹配,绝不碰邻居。

    python3 s3_watch.py                 # 跟随,默认会暂停
    S3_WATCH_PAUSE=0 python3 s3_watch.py  # 只告警不暂停
    python3 s3_watch.py --once          # 跑一次打印现状就退出
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import statistics as st
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

CK = Path(os.environ.get("S3_CKPT", "/workspace/agl-checkpoints/smith_rl_s3"))
# 血统名从 CK 目录名推出来(smith_rl_s4 → s4),可用 S3_LINEAGE 覆盖。
# **不能把 trainer_s3.log 写死**:换血统(s4)时写死的路径指向一个不存在的文件,
# 而 parse_trainer() 对不存在的路径返回 []、judge([]) 返回"无告警" —— 探针会
# **静默地一路报健康**。这正是 §50 那一类静默失效,和判据 7 原来的写法同病。
# 运行期还有一条兜底:trainer 活着但日志不存在时直接告警(见 main())。
LINEAGE = os.environ.get("S3_LINEAGE") or CK.name.rsplit("_", 1)[-1]
TRAINER_LOG = Path(os.environ.get("S3_TRAINER_LOG") or CK / f"trainer_{LINEAGE}.log")
ROLLOUT_GLOB = "agl-logs/new/agl-rollout-*.log"
ALARM = CK / "WATCH_ALARM.json"
HEALTH = CK / f"{LINEAGE}_health.jsonl"
DATASET_MARK = os.environ.get("S3_DATASET_MARK", "train_dataset_s3.jsonl")
POLL = int(os.environ.get("S3_WATCH_POLL", "120"))
DO_PAUSE = os.environ.get("S3_WATCH_PAUSE", "1") == "1"

# trainer 日志的一行形如 "<ray前缀> step:7 - timing/step_start_wall:... - training/reward:0.5 - ..."
RE_STEP = re.compile(r"step:(\d+) - timing/step_start_wall")
# rollout 日志里 smith_agent 收尾那行(agents/smith_agent.py:1035)
RE_DONE = re.compile(
    r"done: mode=(\w+).*?reward=([-\d.]+) raw_reward=([-\d.]+) binary=([\d.]+) "
    r"f2p_ratio=([\d.]+) n_turns=(\d+) max_prompt_tokens=(\d+) reason=(.*)"
)
RE_P2P_REASON = re.compile(r"PASS_TO_PASS (\d+)/(\d+) ok")

METRICS = [
    "response_length/mean",
    "critic/score/mean",
    "training/reward",
    "actor/grad_norm",
    "training/dynamic_sampling/n_groups_zero_adv",
    "training/dynamic_sampling/n_groups_seen",
    "training/dynamic_sampling/n_groups_valid",
    "training/dynamic_sampling/gen_rounds",
    "training/n_rollouts",
    "training/n_rollouts_w_trace",
    "rollout_corr/rollout_is_mean",
    "prompt_length/mean",
    "actor/entropy",
    "actor/kl_loss",
    "actor/pg_clipfrac",
    # 判据 9 的三个量。mismatch_rows 是判据本体,另两个只进快照作参照。
    "training/n_trace_merge_mismatch_rows",
    "training/n_unmerged_rollouts",
    "training/n_budget_splits",
]

# **verl 的部分指标是 repr 出来的 numpy 标量**:`actor/grad_norm:np.float64(0.00247)`、
# `rollout_corr/training_ppl:np.float64(1.776)`,而 critic/* 和 training/* 是裸浮点。
# 10-05 实测:只写 `([-\d.eE+]+)` 会让所有 actor/* 全部解析不到 —— 第 6 条
# 「梯度爆炸」判据就永远不会开火,而且**静默**(缺值被当成"没这项"跳过)。
# 所以这里必须同时吃两种形态,并且 `--selftest` 会断言 METRICS 每一项在两份历史
# 日志上都 100% 解析得出 —— 修之前那四条 actor/* 是 0/90,这个断言会直接红。
RE_NUM = r"(?:np\.float\d+\(|np\.int\d+\()?(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"

# selftest 的夹具:血统 s1(健康 90 步)和 s2(坍塌 5 步)的真实 trainer 日志。
# 阈值全部由它们标定,所以也只能由它们验收。值 = 「这份日志应该开火吗」。
FIXTURES = {
    "s1 健康": (Path("/workspace/agl-checkpoints/smith_rl_s1/trainer_s1.log"), False),
    "s2 坍塌": (Path("/workspace/agl-checkpoints/smith_rl_s2/trainer_s2.log"), True),
}
# 阈值在上面这两份日志上标定过的判据 —— selftest 要求它们**各自**在 s2 上开火。
# 其余判据(4 采集率 / 5 训推偏差 / 6 梯度爆炸 / 8 进程没了)这两份日志里没有正例,
# 是安全网而非标定项,不能要求它们开火,否则验收会变成逼人造假夹具。
CALIBRATED = (1, 7)
# 结构不变量类判据:正常值恒为某个定值(判据 9 是 0),不需要统计标定,也不该受
# 「s1 健康就不许开火」的约束 —— s1 在**合并**这件事上从第一步就是坏的。
# selftest 对它们的要求反过来:**两份夹具都必须开火**(两份都是已知的坏例)。
STRUCTURAL = (9,)
# 只告警、不停机、不写 WATCH_ALARM.json、探针自己不退出的判据。
WARN_ONLY = (91,)
# 判据 9 的停机门槛(单步断链行数)。为什么从「>0 即停」改成 ≥4 —— 见 judge() 里的长注。
MM_STOP = int(os.environ.get("S3_WATCH_MM_STOP", "4"))


def parse_trainer(path: Path) -> list[dict]:
    """按 step 读出每步指标。日志是 tee 追加的,整文件重读最省心(91 步才 ~500KB)。"""
    if not path.exists():
        return []
    rows: dict[int, dict] = {}
    with path.open(errors="replace") as fh:
        for line in fh:
            m = RE_STEP.search(line)
            if not m:
                continue
            rec = {"step": int(m.group(1))}
            for k in METRICS:
                mm = re.search(re.escape(k) + r":" + RE_NUM, line)
                if mm:
                    try:
                        rec[k] = float(mm.group(1))
                    except ValueError:
                        pass
            rows[rec["step"]] = rec  # 同 step 重跑时后写的赢
    return [rows[s] for s in sorted(rows)]


def scan_rollouts(limit: int = 400) -> dict:
    """读最近 limit 条 rollout 日志,算出 trainer 指标里没有的三个量:
    P2P 破坏率、白忙率(P2P 没破但 F2P 一条没过)、F2P 比例分布。

    10-05 修:原来写 `CK / "agl-logs" / "new"`,而 pod 的 hostPath 是
    **血统无关的** `smith_rollout_logs`(job-template-smith.yaml:141,那里特意
    注释了「写死某条血统的目录对另一条血统一定是错的」)。于是这个函数一直
    返回 {},三个量从起训起就没在测 —— 和 §50 一样的静默失效:日志里显示 `-`,
    而 `-` 又正好是「这步还没算完」的正常显示。
    现在找不到目录/找不到 train 行会**显式返回原因**,不再静默返回空。"""
    d = Path(os.environ.get("S3_ROLLOUT_LOGS", "/workspace/agl-checkpoints/smith_rollout_logs/new"))
    if not d.is_dir():
        return {"err": f"rollout 日志目录不存在: {d}"}
    files = sorted(d.glob("agl-rollout-*.log"), key=lambda p: p.stat().st_mtime)[-limit:]
    if not files:
        return {"err": f"rollout 日志目录是空的: {d}"}
    tr = []
    for p in files:
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        m = RE_DONE.search(txt)
        if not m or m.group(1) != "train":
            continue
        reason = m.group(8)
        pm = RE_P2P_REASON.search(reason)
        broke = bool(pm) and pm.group(1) != pm.group(2)
        tr.append(
            {
                "reward": float(m.group(2)),
                "ratio": float(m.group(5)),
                "turns": int(m.group(6)),
                "broke": broke,
                "idle": (not broke) and float(m.group(5)) == 0.0,
            }
        )
    if not tr:
        return {"err": f"{len(files)} 条日志里一条 train 的收尾行都没解析出来(格式变了?)"}
    return {
        "n": len(tr),
        "p2p_break_rate": sum(x["broke"] for x in tr) / len(tr),
        "idle_rate": sum(x["idle"] for x in tr) / len(tr),
        "partial_rate": sum(0.0 < x["ratio"] < 1.0 for x in tr) / len(tr),
        "solved_rate": sum(x["ratio"] >= 1.0 for x in tr) / len(tr),
        "ratio_mean": st.mean(x["ratio"] for x in tr),
        "turns_med": st.median(x["turns"] for x in tr),
    }


def trainer_pid() -> int | None:
    """只认我自己这条 s3 训练进程:cmdline 必须同时含脚本名和 s3 数据集。"""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "train_smith_agent.py"], capture_output=True, text=True, timeout=20
        ).stdout.split()
    except Exception:
        return None
    for pid in out:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "train_smith_agent.py" in cmd and DATASET_MARK in cmd:
            return int(pid)
    return None


def consec(flags: list[bool], n: int) -> bool:
    """末尾是否连续 n 个 True。"""
    return len(flags) >= n and all(flags[-n:])


def judge(rows: list[dict]) -> list[str]:
    """返回触发的告警条目(空 = 健康)。"""
    alarms: list[str] = []
    if not rows:
        return alarms

    # 判据 9 放在 `len(rows) < 3` 的早退**之前**:它是结构不变量,第 1 步就该判,
    # 而且 smoke(只跑 1 步)的验收全靠它。统计类判据需要前 2 步做基线,它不需要。
    # 原来写的是「mm > 0 即开火 + 停机」,按"结构不变量恒为 0"立的。
    # 2026-10-06 s4 step 18 证伪了那个"恒":前 17 步 1152 条 rollout 全 0,第 18 步
    # 出了 1 行。查实的事实:`diverge_kind='token_differs'`、分叉处两段解码文本
    # **逐字完全相同**、`turn_index=39`(40 轮上限的最后一轮)。实验:把那段文本按
    # 39 个切点拆开,`enc(A)+enc(B) != enc(A+B)` 在 24/39 = 62% 的切点成立
    # (文本里 `=` 18 次、`<<` 5 次、`→` 3 次)。而 `decode(encode(W))==W` 且 encode
    # 幂等 ⇒ 不是有损往返,是**切分对上下文极敏感**,长 episode 的接缝上本就会偶发。
    # ⇒ "恒为 0"是错的,真实残余率 1/1152 = 0.09%;但 64/64(修复前)仍必须停机。
    # 所以拆成两档:≥MM_STOP 行 → 判据 9(告警+停机);0<mm<MM_STOP → 判据 91(只告警)。
    # 门槛没有把信号藏起来:每一步的断链数照旧打在 ok/ALARM 行里(`断链 N`)。
    mm = rows[-1].get("training/n_trace_merge_mismatch_rows")
    if mm is not None and mm > 0:
        um = rows[-1].get("training/n_unmerged_rollouts")
        nr = rows[-1].get("training/n_rollouts")
        bs = rows[-1].get("training/n_budget_splits")
        common = (
            f"n_trace_merge_mismatch_rows={mm:.0f}"
            f";未合并 rollout {um if um is None else '%.0f' % um}/"
            f"{nr if nr is None else '%.0f' % nr},预算切分 {bs if bs is None else '%.0f' % bs}。"
            f"看 {CK}/trajectories/step_*_merge_mismatch.jsonl 的 expected/actual_window。"
        )
        if mm >= MM_STOP:
            alarms.append(
                f"[9 合并断链] {common}"
                f"达到停机门槛 {MM_STOP} 行 —— 大面积 episode 被打碎成一轮一行、"
                f"每行带整条 episode 的最终奖励,credit assignment 没在跑,这一步的梯度不可用。"
            )
        else:
            alarms.append(
                f"[91 合并断链(低于停机门槛 {MM_STOP})] {common}"
                f"已知残余机理:长 episode 接缝上的 token 切分不稳定(实测同一段文本"
                f"62% 的切点 enc(A)+enc(B)!=enc(A+B))。单行影响 ≤1/48 训练行、仅此一步,"
                f"**只告警不停机**;要盯的是它会不会长大。"
            )

    if len(rows) < 3:
        return alarms

    def series(k):
        return [r.get(k) for r in rows]

    rl = [r.get("response_length/mean") for r in rows]
    have = [x for x in rl if x is not None]
    if len(have) >= 3:
        base = st.mean(have[:2])  # 前 2 步 = 还没有哪次更新能真正落地
        hits = [(x is not None and x < 0.50 * base) for x in rl]
        if consec(hits, 2):
            alarms.append(
                f"[1 长度绝对坍塌] response_length/mean 连续 2 步 < 0.50×前2步均值"
                f"({0.50 * base:.1f});最近两步 {have[-2]:.1f} / {have[-1]:.1f}。"
                f"s2 就是这个签名(274→112→60),判据 7 同步开火。"
            )
        drift = []
        for i, x in enumerate(rl):
            w = [y for y in rl[max(0, i - 5) : i] if y is not None]
            drift.append(bool(w and x is not None and x / st.mean(w) < 0.55))
        if consec(drift, 2):
            alarms.append(
                f"[2 长度缓慢漂移] 连续 2 步 < 前5步均值的 0.55;s1 九十一步单步最低 0.568,从没破过。"
            )

    za, se = series("training/dynamic_sampling/n_groups_zero_adv"), series(
        "training/dynamic_sampling/n_groups_seen"
    )
    zf = [(a / b if (a is not None and b) else None) for a, b in zip(za, se)]
    if consec([(x is not None and x > 0.60) for x in zf], 3):
        got = [x for x in zf if x is not None][-3:]
        alarms.append(
            f"[3 零方差组回升] 连续 3 步 >60%:{['%.1f%%' % (x * 100) for x in got]}。"
            f"s1 均值 41.0%,F2P 部分分后预期 ~30% —— 回到 60% 说明部分分没生效或模型在同质化。"
        )

    nr, nt = series("training/n_rollouts"), series("training/n_rollouts_w_trace")
    cap = [(t / n if (t is not None and n) else None) for t, n in zip(nt, nr)]
    if consec([(x is not None and x < 0.90) for x in cap], 2):
        alarms.append(f"[4 采集率] 连续 2 步 <0.90,最近 {cap[-1]:.3f}。轨迹在丢,梯度算在残缺数据上。")

    ism = series("rollout_corr/rollout_is_mean")
    if consec([(x is not None and not (0.85 <= x <= 1.15)) for x in ism], 2):
        alarms.append(f"[5 训推偏差] rollout_is_mean 连续 2 步出界,最近 {ism[-1]:.4f}(s1 实测 0.990~0.997)。")

    gn = [x for x in series("actor/grad_norm") if x is not None]
    if gn:
        bad = [(x is None or math.isnan(x) or math.isinf(x)) for x in series("actor/grad_norm")]
        if any(bad[-2:]) and len(gn) >= 2:
            alarms.append("[6 梯度爆炸] grad_norm 出现 NaN/inf。")
        elif len(gn) >= 12:
            med = st.median(gn[-12:-2])
            if med > 0 and consec([x > 10 * med for x in gn[-2:]], 2):
                alarms.append(f"[6 梯度爆炸] grad_norm 连续 2 步 >10× 前10步中位({med:.4g}),最近 {gn[-1]:.4g}。")

    # 用 training/reward(float64、rollout 加权、过滤前),**不是** critic/score/mean
    # (bf16 量化 + 行加权 + 过滤后选择偏差,s2 上整体偏高 +0.2045,把 4.2 倍跌幅
    # 压成 1.4 倍)。相对形式是因为 s1 健康值和 s2 坍塌值区间重叠,绝对门槛会误报。
    rw = series("training/reward")
    hv = [x for x in rw if x is not None]
    if len(hv) >= 3:
        rb = st.mean(hv[:2])
        if rb <= 0:
            alarms.append(f"[7 奖励坍塌] 前 2 步 training/reward 均值 {rb:.4g} ≤ 0 —— 从一开始就没有奖励信号。")
        elif consec([(x is not None and x < 0.40 * rb) for x in rw], 2):
            alarms.append(
                f"[7 奖励坍塌] training/reward 连续 2 步 < 0.40×前2步均值({0.40 * rb:.4f});"
                f"最近两步 {hv[-2]:.4f} / {hv[-1]:.4f}。"
                f"s2 就是这个签名(0.580→0.138,与长度同步);s1 九十一步余量 1.50×。"
            )
    return alarms


def pause_trainer(pid: int) -> str:
    """SIGINT -> verl 会存盘再退;给足时间,绝不升级到 SIGKILL(那会丢 ckpt)。"""
    os.kill(pid, signal.SIGINT)
    for _ in range(120):  # 最多等 10 分钟
        time.sleep(5)
        try:
            os.kill(pid, 0)
        except OSError:
            return f"trainer pid {pid} 已退出(SIGINT,应已存盘)"
    return f"trainer pid {pid} 收到 SIGINT 后 10 分钟仍在,**没有升级信号**(避免丢 ckpt),需要人工看"


def snapshot(rows, roll, pid):
    last = rows[-1] if rows else {}
    za = last.get("training/dynamic_sampling/n_groups_zero_adv")
    se = last.get("training/dynamic_sampling/n_groups_seen")
    return {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": pid,
        "step": last.get("step"),
        "response_length": last.get("response_length/mean"),
        # reward 在前、score_mean 保留在后:前者是判据 7 真正用的量,后者只作参照
        # (bf16 量化 + 过滤后偏差,见模块 docstring 的 a/b/c)。两个都留着,因为
        # **两者之差本身是个有用的量** —— 它就是丢弃组带来的选择偏差。
        "reward": last.get("training/reward"),
        "score_mean": last.get("critic/score/mean"),
        "zero_adv_frac": (za / se) if (za is not None and se) else None,
        "gen_rounds": last.get("training/dynamic_sampling/gen_rounds"),
        "grad_norm": last.get("actor/grad_norm"),
        "rollout_is_mean": last.get("rollout_corr/rollout_is_mean"),
        "merge_mismatch_rows": last.get("training/n_trace_merge_mismatch_rows"),
        "unmerged_rollouts": last.get("training/n_unmerged_rollouts"),
        **{f"roll_{k}": v for k, v in (roll or {}).items()},
    }


def line(s: dict) -> str:
    def f(k, fmt="%.3f"):
        v = s.get(k)
        return (fmt % v) if isinstance(v, (int, float)) else "-"

    if s.get("roll_err"):   # 不让 rollout 侧的失败藏在一串 `-` 里
        return f"step {s.get('step')}  len {f('response_length','%.1f')}  reward {f('reward','%.4f')}  [rollout扫描失败: {s['roll_err']}]"
    return (
        f"step {s.get('step')}  len {f('response_length','%.1f')}  "
        f"reward {f('reward','%.4f')}(score {f('score_mean')})  "
        f"零方差 {('%.1f%%' % (s['zero_adv_frac'] * 100)) if s.get('zero_adv_frac') is not None else '-'}  "
        f"gen轮 {f('gen_rounds','%.0f')}  "
        f"断链 {f('merge_mismatch_rows','%.0f')}  "
        f"破P2P {('%.1f%%' % (s['roll_p2p_break_rate'] * 100)) if s.get('roll_p2p_break_rate') is not None else '-'}  "
        f"白忙 {('%.1f%%' % (s['roll_idle_rate'] * 100)) if s.get('roll_idle_rate') is not None else '-'}  "
        f"部分分 {('%.1f%%' % (s['roll_partial_rate'] * 100)) if s.get('roll_partial_rate') is not None else '-'}  "
        f"做对 {('%.1f%%' % (s['roll_solved_rate'] * 100)) if s.get('roll_solved_rate') is not None else '-'}"
    )


def selftest() -> int:
    """验收两件事,都只用真实历史日志,不用编造的夹具:

    A 指标真的解析得出吗 —— 这是第 5/6 条判据的**前提**。缺值在 judge() 里是被
      静默跳过的,所以解析失败不会报错,只会让判据永远不开火。10-05 实测:
      `([-\\d.eE+]+)` 对所有 actor/*(np.float64 repr)解析率 0/90。
    B 阈值在真实历史上分得开吗 —— 增量回放 judge(rows[:i]),s2 必须开火、
      s1 九十步必须一次都不开火。只要有人改动门槛就会在这里红。
    """
    bad = []
    for tag, (path, want_fire) in FIXTURES.items():
        print(f"=== {tag}: {path.name}")
        if not path.exists():
            print("    日志不在,跳过(不算失败 —— 可能在别的机器上跑)")
            continue
        rows = parse_trainer(path)
        print(f"    解析出 {len(rows)} 步")
        if not rows:
            bad.append(f"{tag}: 一步都没解析出来")
            continue

        # A
        for k in METRICS:
            n = sum(1 for r in rows if k in r)
            if n != len(rows):
                bad.append(f"{tag}: {k} 只解析出 {n}/{len(rows)} 步")

        # B 全量增量回放,并且**逐条**记下每个判据首次开火于哪一步。
        # 为什么不能只记"首次有告警":judge() 返回的是合并列表,只要第 1 条开火,
        # 整个验收就绿 —— 判据 7 原来写成 `score/mean == 0`(结构上永不可能开火)
        # 在这套验收下一直是绿的。这正是 §50 那类静默失效,所以这里必须逐条看。
        per = {}
        for i in range(1, len(rows) + 1):
            for a in judge(rows[:i]):
                c = int(a[1:a.index(" ")])
                per.setdefault(c, rows[i - 1]["step"])
        # 坍塌类和结构类要分开看:want_fire 说的是「这份日志坍塌了吗」,
        # 而结构判据(9)在两份日志上都该开火 —— 混在一起会让 s1 的"不该开火"误红。
        # WARN_ONLY(91)也要排掉:它只告警不停机,出现在 s1 上不代表 s1 坍塌。
        collapse = {c: s for c, s in per.items() if c not in STRUCTURAL and c not in WARN_ONLY}
        fired_at = min(collapse.values()) if collapse else None
        for c in WARN_ONLY:
            if c in per:
                print(f"    [只告警判据 {c}] 首次出现 step {per[c]}(不停机,不计入坍塌判定)")
        for c in STRUCTURAL:
            if c in per:
                print(f"    [结构判据 {c}] 首次开火 step {per[c]}(两份夹具都该开 —— 已知的坏例)")
            else:
                bad.append(f"{tag}: 结构判据 {c} 没开火,可它是已知的坏例 —— 判据是死的")
        rl = [r["response_length/mean"] for r in rows]
        rw = [r["training/reward"] for r in rows]
        # 判据 1/7 都要求**连续 2 步**跌破,所以余量的分子不是"单步最低",而是
        # 「所有相邻两步里、较高那个的最小值」—— 这个数跌到门槛下才会开火。
        # 混用两种分子会得出不可比的余量(单步口径下长度是 1.29×、奖励 1.07×)。
        pair = lambda v: min(max(v[i], v[i + 1]) for i in range(len(v) - 1))
        print(f"    response_length/mean 首2步 {rl[0]:.1f} / {rl[1]:.1f}  "
              f"单步最低 {min(rl):.1f}  连续2步可达最低 {pair(rl):.1f}")
        print(f"    training/reward      首2步 {rw[0]:.4f} / {rw[1]:.4f}  "
              f"单步最低 {min(rw):.4f}  连续2步可达最低 {pair(rw):.4f}")
        if want_fire:
            if fired_at is None:
                bad.append(f"{tag}: 应该开火却全程没开 —— 探针对已知坍塌是瞎的")
            else:
                print(f"    >>> 坍塌判据首次开火于 step {fired_at};逐条 {dict(sorted(per.items()))}")
                for a in judge(rows):
                    print("        " + a)
            # 在这两份日志上标定过阈值的判据,必须**各自**在已知坍塌上开火。
            # 没开 = 这条判据是死的(哪怕整体验收因为别的判据而绿)。
            for c in CALIBRATED:
                if c not in per:
                    bad.append(f"{tag}: 判据 {c} 在已知坍塌上没开火 —— 这条判据是死的,"
                               f"整体绿是被别的判据盖住了(§50 静默失效)")
        else:
            if fired_at is not None:
                bad.append(f"{tag}: 坍塌判据不该开火却在 step {fired_at} 开了(逐条 {collapse})"
                           f" —— 门槛太紧,会把健康训练停掉")
            else:
                print(f"    >>> 全程未开火。连续2步口径余量:长度 "
                      f"{pair(rl) / (0.50 * st.mean(rl[:2])):.2f}×  "
                      f"奖励 {pair(rw) / (0.40 * st.mean(rw[:2])):.2f}×")

    # C rollout 日志路径。路径写错 = bug(必须红);目录空 = 刚起训,合法(只警告)。
    # 这一条是 10-05 补的:scan_rollouts 一直指着血统目录 CK/agl-logs/new,
    # 而 pod 其实写到血统无关的 smith_rollout_logs/new,三个量从头就没在测。
    rd = Path(os.environ.get("S3_ROLLOUT_LOGS", "/workspace/agl-checkpoints/smith_rollout_logs/new"))
    if not rd.is_dir():
        bad.append(f"rollout 日志目录不存在: {rd}(路径写错了?pod 侧见 job-template-smith.yaml 的 hostPath)")
    else:
        r = scan_rollouts()
        if r.get("err"):
            print(f"=== rollout 扫描\n    警告: {r['err']}(刚起训没日志是正常的)")
        else:
            print(f"=== rollout 扫描\n    {r['n']} 条:破P2P {r['p2p_break_rate']*100:.1f}%  "
                  f"白忙 {r['idle_rate']*100:.1f}%  部分分 {r['partial_rate']*100:.1f}%  "
                  f"做对 {r['solved_rate']*100:.1f}%  f2p均值 {r['ratio_mean']:.3f}  轮数中位 {r['turns_med']:.0f}")

    if bad:
        print("\n验收:失败")
        for b in bad:
            print("  - " + b)
        return 1
    print("\n验收:通过")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="用 s1/s2 真实日志验收解析和门槛")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    seen_dead = 0
    while True:
        rows = parse_trainer(TRAINER_LOG)
        roll = scan_rollouts()
        pid = trainer_pid()
        snap = snapshot(rows, roll, pid)
        alarms = judge(rows)

        if pid is None:
            seen_dead += 1
            if seen_dead >= 2:  # 两轮都看不到,排除重启窗口
                alarms.append(f"[8 进程没了] 找不到 {LINEAGE} trainer 进程。看 {TRAINER_LOG} 尾部。")
        else:
            seen_dead = 0
            # trainer 活着却读不到它的日志 = 路径配错。不报就等于一路静默报健康。
            if not TRAINER_LOG.exists():
                alarms.append(
                    f"[10 日志路径错] trainer pid {pid} 在跑,但 {TRAINER_LOG} 不存在 ——"
                    f"探针什么都没在读,之前每一行 ok 都是假的。用 S3_TRAINER_LOG 指对。"
                )

        # 告警分两档。WARN_ONLY 的那几条只打印、进 health 日志,**不**写
        # WATCH_ALARM.json、不发信号、探针自己也不退出 —— 否则一条"知道了但不改变
        # 决定"的观察就会把训练搁死(WATCH_ALARM.json 一在,s4_cycle.sh resume 会
        # exit 6),而且探针 return 1 之后就没人看着了。10-06 step 18 吃过这个亏。
        warns = [a for a in alarms if int(a[1:a.index(" ")]) in WARN_ONLY]
        stops = [a for a in alarms if a not in warns]

        snap["alarms"] = alarms  # 两档都进 health 日志,不藏信号
        with HEALTH.open("a") as fh:
            fh.write(json.dumps(snap, ensure_ascii=False) + "\n")
        print(("ALARM " if stops else "warn  " if warns else "ok    ") + line(snap), flush=True)

        for a in warns:
            print("  " + a, flush=True)

        if stops:
            for a in stops:
                print("  " + a, flush=True)
            act = ""
            if DO_PAUSE and pid and not any(a.startswith("[8") for a in stops):
                act = pause_trainer(pid)
                print("  动作: " + act, flush=True)
            ALARM.write_text(
                json.dumps(
                    {**snap, "action": act or "未暂停(S3_WATCH_PAUSE=0 或进程已不在)"},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1

        if args.once:
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
