#!/usr/bin/env python3
"""起训前的 chat 模板一致性断言。退出码非 0 就别起训。

为什么需要它(§56 → §57 → §58 三次踩同一类坑):

  * §56 改了模板,但训练侧的 prompt 根本不是模板渲染的 → 白改。
  * §57 修了 proxy 丢 `chat_template_kwargs`,但 agl-server 是长命进程、
    `_TOKENIZERS` 是进程内缓存 → **改磁盘文件不算改**。
  * §58 机理定位对了,补丁却打在 base 目录那份模板上,而 proxy 实际加载的是
    **actor 目录**那份(agent 发 `model="auto"`,proxy 解析成上游真实模型名)
    → 又白改一轮 smoke。

这三次的共同点是:"我以为在用的那份"和"真正在用的那份"不是一个东西。
人是记不住的,所以写成断言。

检查项:
  1. 每个候选目录里的两处副本(`chat_template.jinja` 和
     `tokenizer_config.json["chat_template"]`)逐字相同。
  2. 所有候选目录之间逐字相同。
  3. 显式传给 vLLM 的 `--chat-template` 文件(如果给了)也相同。
  4. **往返不变式**:用每个目录的真实 tokenizer 渲染,
     `render(msgs[:k+1], add_generation_prompt=True) + reply`
     必须是 `render(msgs[:k+2], add_generation_prompt=True)` 的前缀 ——
     字符串层和 token 层都验。多轮 RL 的轨迹合并判据就是这个
     (`rollout_adapter.py:756 ids_startswith`)。
  5. **agl-server 不比模板老**:进程启动时间早于任一模板 mtime 就报错,
     因为它缓存的是旧的那份。

用法:
    .venv/bin/python examples/swe_smith/preflight_templates.py \
        --dirs /workspace/models/MiniCPM5-2B-sft-v3-ep3 /workspace/models/MiniCPM5-2B \
        --explicit-template /workspace/models/MiniCPM5-2B/chat_template.jinja \
        --agl-port 18082
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# 探针回复。第 2 条是 §58 的病例:正文里带一个孤立的 `</think>`(SFT 教师轨迹
# 教出来的习惯),旧模板会按它把回复重切成 reasoning+content。
# 第 3 条带 bash 围栏里的 `</think>`,第 4 条带 `<think>` 开标签。
_PROBE_REPLIES = [
    "THOUGHT: plain reply, no markers.\n\n```bash\nls /testbed\n```",
    "THOUGHT: reasoning here\nmore reasoning\n</think>\n\n```bash\ngrep -n x f.py\n```",
    "THOUGHT: the file contains\n\n```bash\ncat <<'EOF' > /tmp/a\n</think>\nEOF\n```",
    "THOUGHT: <think> appears here too\n</think>\n\n```bash\npytest -x\n```",
    "THOUGHT: trailing newlines\n\n\n```bash\necho hi\n```\n\n",
]

_FAIL: list[str] = []


def fail(msg: str) -> None:
    _FAIL.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def collect_copies(d: Path) -> dict[str, str]:
    """一个目录里所有的模板副本。两处都要查 —— 只改一处是 §58 的坑。"""
    out: dict[str, str] = {}
    jinja = d / "chat_template.jinja"
    if jinja.exists():
        out[str(jinja)] = jinja.read_text()
    cfg = d / "tokenizer_config.json"
    if cfg.exists():
        try:
            emb = json.loads(cfg.read_text()).get("chat_template")
        except Exception as exc:  # noqa: BLE001
            fail(f"{cfg} 读不出来:{exc}")
            emb = None
        if isinstance(emb, str):
            out[f"{cfg}::chat_template"] = emb
    return out


def check_roundtrip(d: Path) -> None:
    """用真实 tokenizer 验往返不变式 —— 这是合并判据本体,不是近似。"""
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        fail(f"import transformers 失败:{exc}")
        return
    try:
        tk = AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        fail(f"{d}: tokenizer 加载失败:{exc}")
        return

    base = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the bug in drange()."},
    ]
    # agent 实际发的就是 enable_thinking=False(smith_agent.py:1071)。
    # True 也验一遍:哪天开 thinking 不该再踩一次。
    for enable_thinking in (False, True):
        for i, reply in enumerate(_PROBE_REPLIES):
            msgs = list(base)
            gen = tk.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
            msgs = msgs + [{"role": "assistant", "content": reply}, {"role": "user", "content": "<tool_response>\nout\n</tool_response>"}]
            hist = tk.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
            expected = gen + reply
            if not hist.startswith(expected):
                n = 0
                while n < min(len(hist), len(expected)) and hist[n] == expected[n]:
                    n += 1
                fail(
                    f"{d.name} 往返不变式(字符串层)失败 probe#{i} enable_thinking={enable_thinking}\n"
                    f"        共同前缀尾 {expected[max(0, n - 28):n]!r}\n"
                    f"        expected 续 {expected[n:n + 24]!r}\n"
                    f"        actual   续 {hist[n:n + 24]!r}"
                )
                continue
            # token 层:合并判据跑在 token id 上,字符串过了也可能在边界合并。
            e_ids = tk(expected, add_special_tokens=False)["input_ids"]
            h_ids = tk(hist, add_special_tokens=False)["input_ids"]
            if h_ids[: len(e_ids)] != e_ids:
                k = 0
                while k < min(len(h_ids), len(e_ids)) and h_ids[k] == e_ids[k]:
                    k += 1
                fail(
                    f"{d.name} 往返不变式(token 层)失败 probe#{i} "
                    f"enable_thinking={enable_thinking} 首个分歧 index={k}"
                )
    if not _FAIL:
        ok(f"{d.name}: 往返不变式 {len(_PROBE_REPLIES) * 2} 条探针全过(字符串层+token 层)")


def check_server_not_stale(port: int, newest_mtime: float, newest_path: str) -> None:
    """agl-server 比模板老 = 它缓存的是旧模板(§57)。"""
    try:
        pids = subprocess.run(
            ["pgrep", "-f", "agl-server"], capture_output=True, text=True, check=False
        ).stdout.split()
    except Exception as exc:  # noqa: BLE001
        fail(f"pgrep 失败,无法检查 agl-server 新鲜度:{exc}")
        return
    found = False
    for pid in pids:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except Exception:  # noqa: BLE001
            continue
        if "agl-server" not in cmd or f"port={port}" not in cmd:
            continue
        found = True
        started = os.stat(f"/proc/{pid}").st_mtime
        if started < newest_mtime:
            fail(
                f"agl-server pid {pid} 启动于 {started:.0f},早于模板 {newest_path} "
                f"的 mtime {newest_mtime:.0f} —— 进程内 _TOKENIZERS 缓存的是旧模板,"
                f"请先 bash examples/swe_smith/start_agl_server.sh"
            )
        else:
            ok(f"agl-server pid {pid} 启动于模板改动之后(新鲜)")
    if not found:
        fail(f"端口 {port} 上没找到 agl-server —— 先起它")


def check_dp_group_patch() -> None:
    """verl 的 dp_actor 默认不往 rearrange_micro_batches 传 dp_group,而 dp_group=None
    会让 seqlen_balancing.py:402 的 all_reduce(MAX) 被静默跳过 —— 各 rank 于是算出
    **不同的 micro-batch 条数**,下一个 allgather 直接对不上,表现为挂死而不是报错
    ([[s1-nccl-allgather-hang]]:s1 三步挂两次,靠功耗 123W≠300W 认出来的)。

    补丁打在 site-packages 里,`uv sync` / `pip install -U verl` 会无声地把它冲掉。
    这条跑要连轴 40 小时、nccl_timeout 又是 3600 秒,丢了补丁的代价是每次挂死烧掉
    一小时还查不出原因。所以起训前机检,而且判 FAIL(不是 warn):
    这个缺陷没有任何"降级也能跑"的形态。

    不导入 verl:import 会连带拉起 torch,在共享卡上多开一个 CUDA 上下文纯属找事,
    而这个判据本来就是纯文本比对。按路径定位即可。
    """
    import sysconfig  # noqa: PLC0415

    root = Path(__file__).resolve().parents[2]  # examples/swe_smith/x.py -> repo 根
    cands = sorted(root.glob(".venv/lib/python3*/site-packages/verl/workers/actor/dp_actor.py"))
    if not cands:
        cands = [Path(sysconfig.get_paths()["purelib"]) / "verl/workers/actor/dp_actor.py"]
    src = cands[0]
    if not src.exists():
        fail(f"找不到 verl 的 dp_actor.py(查过 {src}）—— 无法确认 dp_group 补丁")
        return
    text = src.read_text()
    n_call = text.count("dp_group=self._dp_group()")
    has_def = "def _dp_group(self)" in text
    if has_def and n_call >= 2:
        ok(f"{src}:补丁在位(_dp_group 定义 + {n_call} 处调用点)")
    else:
        fail(f"{src}:dp_group 补丁丢了(定义={has_def} 调用点={n_call},应为 True/2)"
             f" —— verl 被重装过?重打补丁再起训,否则 allgather 会挂死")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="所有可能被加载 tokenizer 的模型目录")
    ap.add_argument("--explicit-template", default=None, help="显式传给 vLLM 的 --chat-template 路径")
    ap.add_argument("--agl-port", type=int, default=18082)
    ap.add_argument("--skip-server-check", action="store_true")
    args = ap.parse_args()

    print("== 1/5 每个目录内部的两处副本是否一致 ==")
    all_copies: dict[str, str] = {}
    newest_mtime, newest_path = 0.0, ""
    for raw in args.dirs:
        d = Path(raw)
        if not d.is_dir():
            fail(f"{d} 不是目录")
            continue
        copies = collect_copies(d)
        if not copies:
            fail(f"{d} 里没有任何模板副本")
            continue
        hashes = {_md5(v) for v in copies.values()}
        if len(hashes) > 1:
            fail(f"{d} 内部副本不一致:" + ", ".join(f"{k}={_md5(v)[:8]}" for k, v in copies.items()))
        else:
            ok(f"{d.name}: {len(copies)} 处副本一致 md5={next(iter(hashes))[:10]}")
        all_copies.update(copies)
        for p in (d / "chat_template.jinja", d / "tokenizer_config.json"):
            if p.exists() and p.stat().st_mtime > newest_mtime:
                newest_mtime, newest_path = p.stat().st_mtime, str(p)

    if args.explicit_template:
        p = Path(args.explicit_template)
        if not p.exists():
            fail(f"显式模板 {p} 不存在")
        else:
            all_copies[f"{p}(显式传给 vLLM)"] = p.read_text()
            if p.stat().st_mtime > newest_mtime:
                newest_mtime, newest_path = p.stat().st_mtime, str(p)

    print("== 2/5 所有候选副本之间是否一致 ==")
    groups: dict[str, list[str]] = {}
    for k, v in all_copies.items():
        groups.setdefault(_md5(v), []).append(k)
    if len(groups) > 1:
        fail(f"共 {len(groups)} 个不同版本在用(必须只有 1 个):")
        for h, ks in groups.items():
            print(f"        md5={h[:10]}  ×{len(ks)}")
            for k in ks:
                print(f"            {k}")
    else:
        ok(f"{len(all_copies)} 处副本全部逐字相同 md5={next(iter(groups))[:10]}")

    print("== 3/5 往返不变式(合并判据本体) ==")
    for raw in args.dirs:
        d = Path(raw)
        if d.is_dir():
            check_roundtrip(d)

    print("== 4/5 agl-server 新鲜度 ==")
    if args.skip_server_check:
        ok("按要求跳过")
    elif newest_mtime:
        check_server_not_stale(args.agl_port, newest_mtime, newest_path)

    print("== 5/5 site-packages 里的 dp_group 补丁还在不在 ==")
    check_dp_group_patch()

    print()
    if _FAIL:
        print(f"preflight 不通过:{len(_FAIL)} 项失败 —— 不要起训")
        return 1
    print("preflight 全过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
