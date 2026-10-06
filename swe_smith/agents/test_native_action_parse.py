"""Unit + acceptance tests for accepting MiniCPM5's native tool-call syntax.

Run: .venv/bin/python examples/swe_smith/test_native_action_parse.py

The acceptance test at the bottom replays the *real* distribution: every tool call
in the 858 SFT training rows, rendered through chat_template.jinja exactly as the
SFT loss saw it. That is what the policy was optimized to emit, so it is the only
honest measure of how much the widened parser recovers.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents"))

from smith_agent import (  # noqa: E402
    FormatError,
    UnavailableToolError,
    parse_action,
    strip_turn_delims,
)

FAILED = []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        FAILED.append(f"{name}: {exc}")
        print(f"  FAIL {name}: {exc}")
    else:
        print(f"  ok   {name}")


def expect_action(content, want):
    got = parse_action(content)
    assert got == want, f"got {got!r} want {want!r}"


def expect_raises(content, exc_type, **attrs):
    try:
        got = parse_action(content)
    except exc_type as exc:
        for k, v in attrs.items():
            assert getattr(exc, k) == v, f"{k}={getattr(exc, k)!r} want {v!r}"
        return
    except FormatError as exc:
        raise AssertionError(f"raised {type(exc).__name__}, want {exc_type.__name__}")
    raise AssertionError(f"returned {got!r}, want {exc_type.__name__}")


print("--- 原有行为不能变 ---")
check("fence", lambda: expect_action("THOUGHT: look\n```bash\nls -la /testbed\n```", "ls -la /testbed"))
check("fence mswea", lambda: expect_action("```mswea_bash_command\npwd\n```", "pwd"))
check("no action", lambda: expect_raises("I think the bug is in foo.py", FormatError, n_actions=0))
check("two fences", lambda: expect_raises("```bash\na\n```\n```bash\nb\n```", FormatError, n_actions=2))

print("--- 原生格式要被接受 ---")
check(
    "native bash",
    lambda: expect_action(
        '<function name="bash"><param name="command">ls -la /testbed</param></function>', "ls -la /testbed"
    ),
)
check(
    "native + think",
    lambda: expect_action(
        '<think>\nlook around\n</think>\n\n<function name="bash"><param name="command">pwd</param></function>', "pwd"
    ),
)
check(
    "native CDATA",
    lambda: expect_action(
        '<function name="bash"><param name="command"><![CDATA[grep -rn "a<b" /testbed]]></param></function>',
        'grep -rn "a<b" /testbed',
    ),
)
check(
    "native CDATA multiline",
    lambda: expect_action(
        '<function name="bash"><param name="command"><![CDATA[cd /testbed && python - <<\'PY\'\nprint(1)\nPY]]></param></function>',
        "cd /testbed && python - <<'PY'\nprint(1)\nPY",
    ),
)
check(
    "native bare CDATA (param 壳丢了)",
    lambda: expect_action('<function name="bash"><![CDATA[cd /testbed && ls]]></function>', "cd /testbed && ls"),
)
check(
    "native 别名 shell/exec",
    lambda: expect_action('<function name="shell"><param name="cmd">whoami</param></function>', "whoami"),
)

print("--- 本 harness 没有的工具:要报工具名,不是报格式 ---")
check(
    "native read",
    lambda: expect_raises(
        '<function name="read"><param name="filePath">/testbed/a.py</param></function>',
        UnavailableToolError,
        tools=["read"],
    ),
)
check(
    "native grep",
    lambda: expect_raises(
        '<function name="grep"><param name="pattern">foo</param><param name="path">/testbed</param></function>',
        UnavailableToolError,
        tools=["grep"],
    ),
)
check(
    "退化:命令塞进了工具名那一格",
    lambda: expect_raises(
        '<function name="ls -la /testbed"></function>', UnavailableToolError, tools=["ls -la /testbed"]
    ),
)
check(
    "两个 read 只算一次工具名去重后仍报错",
    lambda: expect_raises(
        '<function name="read"><param name="filePath">a</param></function>'
        '<function name="edit"><param name="filePath">b</param></function>',
        UnavailableToolError,
        tools=["read", "edit"],
    ),
)

print("--- 歧义必须拒绝 ---")
check(
    "两个原生 bash",
    lambda: expect_raises(
        '<function name="bash"><param name="command">a</param></function>'
        '<function name="bash"><param name="command">b</param></function>',
        FormatError,
        n_actions=2,
    ),
)
check(
    "围栏 + 原生混用",
    lambda: expect_raises(
        '```bash\na\n```\n<function name="bash"><param name="command">b</param></function>', FormatError, n_actions=2
    ),
)
check(
    "bash 但没有命令参数",
    lambda: expect_raises(
        '<function name="bash"><param name="timeout">30</param></function>', UnavailableToolError, tools=["bash"]
    ),
)

print("--- 分隔符剥除 ---")


def expect_strip(content, want):
    got = strip_turn_delims(content)
    assert got == want, f"got {got!r} want {want!r}"


check(
    "尾部 im_end 剥掉后仍能解析",
    lambda: expect_action(
        strip_turn_delims('<function name="bash"><param name="command">ls</param></function><|im_end|>\n'), "ls"
    ),
)
check("只有 im_end", lambda: expect_strip("<|im_end|>", ""))
check("im_end 之后的内容全丢", lambda: expect_strip("done<|im_end|>\n<|im_start|>user\nhi", "done"))
check("不碰正常文本里的尖括号", lambda: expect_strip("if a<b and c>d: pass", "if a<b and c>d: pass"))

# ---------------------------------------------------------------- acceptance
print("\n--- 验收:SFT 真实分布(858 行,12528 个学习轮)---")
TRAIN = "/workspace/agl-checkpoints/swe_smith_smoke/sft/data/train.parquet"
SFTDIR = "/workspace/agl-checkpoints/swe_smith_smoke/sft"
if not os.path.exists(TRAIN):
    print(f"  跳过:{TRAIN} 不在")
else:
    sys.path.insert(0, SFTDIR)
    import pandas as pd
    from transformers import AutoTokenizer

    from swe_sft_dataset import learnable_indices, normalize_messages, render

    tk = AutoTokenizer.from_pretrained("/workspace/models/MiniCPM5-2B", trust_remote_code=True)
    df = pd.read_parquet(TRAIN)
    ok = unavail = fmt = 0
    tools_seen = {}
    for i in range(len(df)):
        r = df.iloc[i]
        msgs = normalize_messages(json.loads(r["messages_json"]))
        tl = json.loads(r["tools_json"])
        for j in learnable_indices(msgs, r["segment"]):
            before = render(tk, msgs[:j], tl, True)
            after = render(tk, msgs[: j + 1], tl, False)
            span = after[len(before) :]
            if "<function name=" not in span:
                continue
            try:
                parse_action(span)
                ok += 1
            except UnavailableToolError as exc:
                unavail += 1
                for t in exc.tools:
                    tools_seen[t] = tools_seen.get(t, 0) + 1
            except FormatError:
                fmt += 1
    total = ok + unavail + fmt
    print(f"  带原生调用的学习轮 = {total}")
    print(f"    解析成 shell 动作      = {ok:5d}  ({ok / max(total, 1):.1%})")
    print(f"    判「本 harness 无此工具」= {unavail:5d}  ({unavail / max(total, 1):.1%})")
    print(f"    仍判格式错(多动作等)   = {fmt:5d}  ({fmt / max(total, 1):.1%})")
    print(f"    无此工具的名字: {sorted(tools_seen.items(), key=lambda kv: -kv[1])}")
    assert ok > 0, "一条都没解析出来,补丁没生效"

print()
if FAILED:
    print(f"FAILED {len(FAILED)}:")
    for f in FAILED:
        print("  -", f)
    sys.exit(1)
print("全部通过")
