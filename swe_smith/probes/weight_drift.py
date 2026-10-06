#!/usr/bin/env python3
"""量「模型到底动了多少」—— 训练前后权重的全局相对 L2 漂移。

为什么要有这个脚本:在线指标回答不了「学到东西了吗」(那只能靠全量 474 配对
McNemar),但**「参数动了吗、动了多少」是能离线定死的**,而且它一次性区分两种
完全不同的失败:

    算法有偏、学歪了   →  漂移大,但评测跌
    优化量不够、没动   →  漂移小到可以忽略

s1 就是个反例教材:在线看 `actor/kl_loss` 0.63→0.70、`pg_clipfrac` 1~2%,
「更新了」这一层明明过了,可全量 474 平线 —— 所以**「更新了」不等于「学到了」**,
这个数只能当排除项用,不能当成绩。

口径和 §39.1 完全一致(改了口径就没法和历史比):

    逐张量   d = ||W_new - W_old||_2 ,  n = ||W_old||_2    (fp32 累加)
    全局     sqrt(Σ d²) / sqrt(Σ n²)

历史参照(同一口径):

    v2 step284   4.316e-04  = 0.043%   欠训(284 步只喂了 2272 个题目实例)
    v4 step62    1.74e-03   = 0.174%   欠训
    SFT(全参)   3.29e-03   = 0.329%   有效(474 上 123→136)

用法(第二个参数默认就是 s3 的起点 SFT ep3):

    python3 weight_drift.py /workspace/models/s3_step5_hf
    python3 weight_drift.py <new_hf_dir> <old_hf_dir> --top 15

FSDP 分片的 ckpt 要先 merge 成 HF(和 s3_cycle.sh probe 用的是同一条路):
    .venv/bin/python -m verl.model_merger merge --backend fsdp \
        --local_dir <CK>/global_step_<S>/actor --target_dir <new_hf_dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

BASE = "/workspace/models/MiniCPM5-2B-sft-v3-ep3"
# 同口径的历史值,打印在结论旁边 —— 一个裸数字没法解读。
REFERENCE = [
    ("v2 step284", 4.316e-04, "欠训"),
    ("v4 step62", 1.74e-03, "欠训"),
    ("SFT 全参", 3.29e-03, "有效(474 上 123→136)"),
]


def shard_map(d: Path) -> dict[str, Path]:
    """张量名 -> 所在 safetensors 文件。单文件和分片 index 两种布局都吃。"""
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return {k: d / v for k, v in wm.items()}
    out: dict[str, Path] = {}
    for p in sorted(d.glob("*.safetensors")):
        with safe_open(str(p), framework="pt") as f:
            for k in f.keys():
                out[k] = p
    if not out:
        raise SystemExit(f"{d} 里没有 safetensors")
    return out


def norms(new_d: Path, old_d: Path) -> tuple[list[tuple[str, float, float]], list[str], list[str]]:
    """返回 [(name, d, n)] 以及两边各自独有的张量名。"""
    nmap, omap = shard_map(new_d), shard_map(old_d)
    only_new = sorted(set(nmap) - set(omap))
    only_old = sorted(set(omap) - set(nmap))
    common = sorted(set(nmap) & set(omap))

    # 按文件分组打开,避免一个张量开一次文件(3 个分片 × 381 张量会很慢)
    rows: list[tuple[str, float, float]] = []
    handles: dict[Path, object] = {}

    def get(path: Path, name: str) -> torch.Tensor:
        if path not in handles:
            handles[path] = safe_open(str(path), framework="pt")
        return handles[path].get_tensor(name)  # type: ignore[union-attr]

    for k in common:
        a = get(nmap[k], k).to(torch.float32)
        b = get(omap[k], k).to(torch.float32)
        if a.shape != b.shape:
            only_new.append(f"{k}(形状不同 {tuple(a.shape)} vs {tuple(b.shape)})")
            continue
        rows.append((k, float((a - b).norm().item()), float(b.norm().item())))
    return rows, only_new, only_old


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new", help="训练后的 HF 目录")
    ap.add_argument("old", nargs="?", default=BASE, help=f"起点 HF 目录(默认 {BASE})")
    ap.add_argument("--top", type=int, default=10, help="列出漂移最大的 N 个张量")
    args = ap.parse_args()

    new_d, old_d = Path(args.new), Path(args.old)
    for d in (new_d, old_d):
        if not d.is_dir():
            raise SystemExit(f"目录不存在: {d}")

    rows, only_new, only_old = norms(new_d, old_d)
    if not rows:
        raise SystemExit("没有可比的同名同形张量")

    sd = math.sqrt(sum(d * d for _, d, _ in rows))
    sn = math.sqrt(sum(n * n for _, _, n in rows))
    glob = sd / sn if sn else float("nan")

    print(f"新 {new_d}")
    print(f"旧 {old_d}")
    print(f"比了 {len(rows)} 个张量" + (f";仅新有 {len(only_new)}、仅旧有 {len(only_old)}" if (only_new or only_old) else ""))
    for n in only_new[:5]:
        print(f"   仅新: {n}")
    for n in only_old[:5]:
        print(f"   仅旧: {n}")

    print(f"\n全局相对漂移 = {glob:.3e}  = {glob * 100:.3f}%")

    per = sorted(((d / n if n else 0.0), k, d, n) for k, d, n in rows)
    print(f"\n漂移最大的 {args.top} 个张量(相对):")
    for rel, k, d, n in per[::-1][: args.top]:
        print(f"  {rel:.3e}  {k}   (d={d:.4g} n={n:.4g})")

    # layernorm 是否一起动了 —— 全 0 说明只有 proj 在学(§39.1 当时就是这个样子)
    ln = [(k, d) for k, d, _ in rows if "norm" in k.lower()]
    if ln:
        mx = max(d for _, d in ln)
        print(f"\nlayernorm 类 {len(ln)} 个张量,最大绝对漂移 {mx:.3e}" + ("  (全 0:只有 proj 在动)" if mx == 0 else ""))

    print("\n同口径历史参照:")
    for name, v, note in REFERENCE:
        print(f"  {v:.3e} = {v * 100:.3f}%  {name:12s} {note}")
    print("\n注意:漂移大只说明「动了」,**不说明「学到了」**。s1 在线 kl_loss 0.63→0.70、")
    print("clipfrac 1~2%,照样全量 474 平线。能回答「学到了吗」的只有全量配对 McNemar。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
