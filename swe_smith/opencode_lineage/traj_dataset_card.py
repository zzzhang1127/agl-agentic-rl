#!/usr/bin/env python3
"""Build the HF dataset card (README.md) for an assembled (segment-aware) teacher-trajectory jsonl.

usage: traj_dataset_card.py TRAJ.jsonl SWEEP_ROOT PROXY_EVENTS.jsonl OUT_README.md \
          --teacher deepseek-v4-flash --opencode-cfg opencode.teacher-dsv4flash.18034.json \
          [--ts-window START:END ...] [--spend-usd 17.2] [--run-note "batch1: ...|rc: ..."]
SWEEP_ROOT may be several roots joined by ','. Prints the stats it computed as JSON on stdout as well.
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_repo_sweep as sweep  # noqa: E402


def pct(a, b):
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else "n/a"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def dist(xs):
    return {"mean": st.mean(xs) if xs else 0, "p50": q(xs, .5), "p90": q(xs, .9), "max": max(xs) if xs else 0, "n": len(xs)}


def fmt(d, f="{:.0f}"):
    return f"mean {f.format(d['mean'])}, p50 {d['p50']}, p90 {d['p90']}, max {d['max']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("sweep_root"); ap.add_argument("events"); ap.add_argument("out")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--opencode-cfg", required=True)
    ap.add_argument("--key-spend-start", type=float, default=None)
    ap.add_argument("--key-spend-end", type=float, default=None)
    ap.add_argument("--spend-usd", type=float, default=None, help="total gateway spend (overrides key-spend delta)")
    ap.add_argument("--ts-start", type=float, default=0.0, help="only count proxy events with ts >= this")
    ap.add_argument("--ts-end", type=float, default=1e18)
    ap.add_argument("--ts-window", action="append", default=[], help="START:END epoch windows of proxy events to count (repeatable)")
    ap.add_argument("--run-note", default="", help="'|'-separated notes on the runs contained")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.traj, encoding="utf-8") if l.strip()]
    cfg = json.load(open(a.opencode_cfg))
    mcfg = next(iter(cfg["provider"]["agl"]["models"].values()))
    ctx, outl = mcfg["limit"]["context"], mcfg["limit"]["output"]
    windows = [tuple(map(float, w.split(":"))) for w in a.ts_window] or [(a.ts_start, a.ts_end)]
    ev = [json.loads(l) for p in a.events.split(",") for l in open(p, encoding="utf-8") if l.strip()]  # comma-separated: one per proxy dir
    ev = [r for r in ev if any(s <= r.get("ts", 0) <= e for s, e in windows)]
    ok = [r for r in ev if r.get("status") == 200 and (r.get("response") or {}).get("usage")]
    ptok = sum(r["response"]["usage"].get("prompt_tokens", 0) for r in ok)
    ctok = sum(r["response"]["usage"].get("completion_tokens", 0) for r in ok)
    rtok = sum(((r["response"]["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens") or 0) for r in ok)
    cached = sum(((r["response"]["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0) for r in ok)
    n429 = sum(1 for r in ev if r.get("status") == 429)
    nerr = sum(1 for r in ev if r.get("status") != 200)

    tk = "total_tokens" if "total_tokens" in rows[0] else "total_tokens_est"
    # rollout = (run, instance_id); its segments are the rows
    roll = defaultdict(list)
    for r in rows:
        roll[(r.get("run"), r["instance_id"])].append(r)
    rollouts = list(roll.values())
    res_roll = [s for s in rollouts if s[0]["resolved"]]
    clean_roll = [s for s in rollouts if s[0]["resolved"] and not any(r["flags"] for r in s if r["segment"] != "compact")]
    complete_roll = [s for s in rollouts if not any("post_segments_missing" in r["flags"] for r in s)]
    seg = Counter(r["segment"] for r in rows)
    runs = Counter(r.get("run") for r in rows)
    ncomp = Counter("unknown(>=1)" if s[0]["n_compactions"] is None else s[0]["n_compactions"] for s in rollouts)
    flags = Counter(f for r in rows for f in r["flags"])
    row_res = [r for r in rows if r["resolved"]]
    row_clean = [r for r in rows if r["resolved"] and not r["flags"]]
    tools = Counter()
    for r in rows:
        for m in r["messages"]:
            if m.get("role") == "assistant" and not m.get("synthetic"):
                for tc in m.get("tool_calls") or []:
                    tools[tc["function"]["name"]] += 1
    turns_roll = [sum(r["n_assistant_turns"] for r in s if r["segment"] != "compact") for s in rollouts]
    turns_roll_complete = [sum(r["n_assistant_turns"] for r in s if r["segment"] != "compact") for s in complete_roll]
    seg_tok = {k: dist([r[tk] for r in rows if r["segment"] == k]) for k in ("main", "post", "compact")}
    seg_turns = {k: dist([r["n_assistant_turns"] for r in rows if r["segment"] == k]) for k in ("main", "post")}
    mturn = [r["max_turn_tokens"] for r in rows if r["segment"] != "compact"]
    walls = [s[0]["wall_seconds"] or 0 for s in rollouts]
    images = Counter(s[0]["image_name"] for s in rollouts)
    problems = {s[0]["instance_id"] for s in rollouts}
    spend = a.spend_usd if a.spend_usd is not None else (
        (a.key_spend_end - a.key_spend_start) if (a.key_spend_start is not None and a.key_spend_end is not None) else None)

    stats = {
        "n_rows": len(rows), "n_rollouts": len(rollouts), "n_problems": len(problems), "runs": dict(runs), "segments": dict(seg),
        "n_resolved_rollouts": len(res_roll), "n_clean_resolved_rollouts": len(clean_roll), "n_complete_rollouts": len(complete_roll),
        "n_resolved_rows": len(row_res), "n_clean_resolved_rows": len(row_clean),
        "compactions_per_rollout": dict(sorted(ncomp.items(), key=lambda kv: (isinstance(kv[0], str), kv[0] if not isinstance(kv[0], str) else 0))), "flags": dict(flags), "tool_calls": dict(tools),
        "turns_per_rollout": dist(turns_roll), "turns_per_complete_rollout": dist(turns_roll_complete),
        "segment_tokens": seg_tok, "segment_turns": seg_turns, "max_turn_tokens": dist(mturn),
        "wall_mean": st.mean(walls), "wall_p90": q(walls, .9), "images": len(images),
        "api_requests_ok": len(ok), "api_prompt_tokens": ptok, "api_cached_prompt_tokens": cached, "api_completion_tokens": ctok,
        "api_reasoning_tokens": rtok, "api_429": n429, "api_non200": nerr, "spend_usd": spend,
    }
    print(json.dumps(stats, indent=1))

    prompt_tmpl = sweep.prompt_text({"problem_statement": "{problem_statement}"})
    main0 = next(r for r in rows if r["segment"] == "main")
    sysmsg = next((m for m in main0["messages"] if m["role"] == "system"), None)
    sys_chars = len(sysmsg["content"]) if sysmsg and isinstance(sysmsg.get("content"), str) else 0
    tool_names = [t["function"]["name"] for t in (main0.get("tool_schemas") or [])]
    est = "" if tk == "total_tokens" else " (est. chars/3.7)"

    flag_doc = {
        "edits_tests": "(rollout) patch.diff touches test files — test tampering; never use for SFT",
        "never_ran_tests": "(rollout) no bash tool call that looks like running tests (pytest/unittest/tox...)",
        "post_segments_missing": "(rollout) the run compacted but a post-compaction segment is not in the dataset. For `batch1` this is "
                                 "the pre-fix proxy bug (fixed 2026-09-20): only the `main` segment exists and `n_compactions` is null "
                                 "(count unknown, >= 1). 207 of these 210 problems were re-run (run `rc`)",
        "over_turn_cap": "(rollout) more than 60 assistant turns over all segments (student turn cap)",
        "long_turn": "(segment) some assistant turn (reasoning + text + tool-call args) exceeds 6144 MiniCPM tokens (student per-turn output limit)",
        "over_context": "(segment) the segment's context exceeds 38912 MiniCPM tokens (student compaction threshold)",
        "no_final_answer": "(last segment / compact rows) final assistant message empty or finish_reason != stop",
    }
    lines = [
        "---", "license: mit", "task_categories:", "- text-generation", "language:", "- en", "tags:",
        "- swe-smith", "- agentic", "- sft", "- trajectories", "- opencode", "- context-compaction", f"- {a.teacher}", "size_categories:",
        "- n<1K" if len(rows) < 1000 else "- 1K<n<10K", "---", "",
        f"# SWE-smith agent trajectories from `{a.teacher}` (opencode harness, compaction-aware)", "",
        f"Teacher trajectories collected on the SWE-smith **train** split used for MiniCPM5-2B agentic RL/SFT. "
        f"One opencode rollout = one problem attempt. opencode auto-compacts the context when it grows past the model's budget; "
        f"the student is meant to **learn compaction too**, so a compacted rollout is stored as several rows (*segments*), each a complete "
        f"OpenAI-style chat request as the teacher saw it, so each can be trained on independently:", "",
        "| `segment` | what the row contains | what the student learns from it |", "|---|---|---|",
        "| `main` | system prompt, task prompt, assistant/tool turns up to the last request before the 1st compaction (or the whole rollout if it never compacted) | normal agent turns |",
        "| `compact` (K) | opencode's summarizer request: summarization system prompt + the rendered conversation so far; the final assistant message is the structured summary | writing the compaction summary |",
        "| `post` (K) | the rebuilt context after compaction K: system prompt + user `What did we do so far?` + assistant summary (marked `synthetic: true`, it is *input* here) + the new turns | continuing from a summary |",
        "",
        f"Rows are emitted for **all** attempts (resolved or not); filter on `resolved`, `segment` and `flags`. "
        f"Runs contained: {'; '.join(f'`{k}` {v} rows' for k, v in runs.items())}. " + (a.run_note.replace('|', ' ') if a.run_note else ""), "",
        "## Numbers", "",
        "| metric | value |", "|---|---|",
        f"| distinct problems | {len(problems)} (stratified round-robin over {len(images)} SWE-smith docker images, seed 20260920) |",
        f"| rollouts (problem attempts) | {len(rollouts)} ; rows (segments) {len(rows)}: " + ", ".join(f"`{k}` {v}" for k, v in seg.items()) + " |",
        f"| resolved rollouts (all F2P pass, no P2P regression) | {pct(len(res_roll), len(rollouts))} |",
        f"| resolved **and** no quality flag on any main/post segment (SFT-ready rollouts) | {pct(len(clean_roll), len(rollouts))} |",
        f"| SFT-ready rows (resolved, no flags) | {len(row_clean)} of {len(rows)} |",
        f"| compactions per rollout | " + ", ".join(f"{k}: {v}" for k, v in sorted(ncomp.items(), key=lambda kv: (isinstance(kv[0], str), kv[0] if not isinstance(kv[0], str) else 0))) + " |",
        f"| assistant turns per rollout (all main+post segments) | {fmt(stats['turns_per_rollout'], '{:.1f}')} (complete rollouts only: {fmt(stats['turns_per_complete_rollout'], '{:.1f}')}) |",
        f"| assistant turns per segment | main {fmt(seg_turns['main'], '{:.1f}')}; post {fmt(seg_turns['post'], '{:.1f}')} |",
        f"| segment length, MiniCPM tokens{est} | main {fmt(seg_tok['main'])}; post {fmt(seg_tok['post'])}; compact {fmt(seg_tok['compact'])} |",
        f"| longest single assistant turn (MiniCPM tokens) | {fmt(stats['max_turn_tokens'])} |",
        f"| wall clock per rollout | mean {stats['wall_mean']:.0f} s, p90 {stats['wall_p90']:.0f} s |",
        f"| API requests (HTTP 200) | {len(ok)} ; 429 rate-limited {n429}, other non-200 {nerr - n429} (retried by opencode) |",
        f"| API prompt tokens (teacher tokenizer) | {ptok:,} (cached {cached:,}) |",
        f"| API completion tokens | {ctok:,} (reasoning {rtok:,}) |",
        f"| gateway spend | {'$%.2f (%.4f $/rollout)' % (spend, spend / len(rollouts)) if spend is not None else 'n/a'} |",
        "", "Tool calls over all main/post rows: " + ", ".join(f"`{k}` {v}" for k, v in tools.most_common()) + ".", "",
        "Quality flags (a row may carry several; rollout-level flags are copied onto every segment of the rollout):", "",
        "| flag | rows | meaning |", "|---|---|---|",
        *[f"| `{k}` | {flags.get(k, 0)} | {v} |" for k, v in flag_doc.items()],
        "", "## Harness", "",
        f"- **Agent harness**: [opencode](https://opencode.ai) 1.18.28, `opencode run --pure --auto --format json`, run *inside* the SWE-smith "
        f"problem container (`/testbed`, git metadata removed, network disabled except the model endpoint). Driver: `run_repo_sweep.py` from "
        f"our agent-lightning SWE-smith setup; the harness runs the hidden F2P/P2P tests *after* opencode exits, the teacher never sees the verdict.",
        f"- **Teacher model**: `{a.teacher}` via an OpenAI-compatible gateway (litellm), reasoning content visible and stored per turn as "
        f"`reasoning_content` (opencode sends it back in the history on later turns). No server-side tools, no web search.",
        f"- **Tools** (opencode built-ins, same set the student MiniCPM5-2B sees; schemas in `tool_schemas`): {', '.join(f'`{t}`' for t in tool_names)}. "
        f"`webfetch` cannot succeed (network disabled); `task`/`skill`/`todowrite` were never used.",
        f"- **System prompt**: opencode's default agent system prompt ({sys_chars:,} chars), stored verbatim as the first message of every `main`/`post` row.",
        f"- **Context / output limits and compaction**: opencode 1.18.28 triggers auto-compaction when `tokens.total >= context − max_output` "
        f"(the `compaction.reserved` setting is *not* applied for models without an explicit input limit). Student config: context 45,056, "
        f"output 6,144 → threshold **38,912**. Run `batch1` used context 52,480 / output 16,384 → threshold 36,096 (teacher tokenizer); "
        f"run `rc` (config in `opencode.teacher.json`) uses context {ctx:,} / output {outl:,} → threshold {ctx - outl:,}, aligned with the student. "
        f"Teacher and student tokenizers differ, so the `over_context` flag re-measures every segment with the MiniCPM tokenizer.",
        "- **Sampling**: gateway defaults (opencode sends no temperature); `max_tokens` = output limit.",
        "- **Per-problem limits**: 1800 s wall clock for the agent (`AGL_SAMPLE_TIMEOUT`), 600 s for the hidden test run; no explicit turn cap "
        "for the teacher (the student's 60-turn cap is reported as the `over_turn_cap` flag instead).",
        "- **Task prompt** (`prompt_text`, the first user message; `{problem_statement}` = SWE-smith problem statement):", "",
        "```", prompt_tmpl.rstrip(), "```", "",
        "## Row schema", "",
        "`instance_id, run, segment, segment_index, n_segments, n_compactions, repo, image_name, resolved, reward, n_f2p_pass, n_f2p, n_p2p_ok, n_p2p, "
        "reason, opencode_rc, wall_seconds, compacted, n_assistant_turns, n_tool_calls, tool_names, chars, " + tk + ", max_turn_tokens, "
        "n_turns_with_reasoning, reasoning_chars, patched_files, test_files_edited, ran_tests, flags, teacher_finish, teacher_model, messages, tool_schemas`.", "",
        "`messages` is OpenAI chat format: `system`, `user`, then alternating `assistant` (with `reasoning_content`, `content`, `tool_calls`) "
        "and `tool` messages (tool output as opencode returned it, truncated by opencode's own limits), ending with the final assistant message "
        "of that segment. In `post` rows the first assistant message carries `synthetic: true` (the summary opencode injected) — mask it from the loss. "
        "`compact` rows have no tools; their messages are `system` (summarizer prompt), `user` (`Here is the conversation so far: <conversation>…</conversation> <template>…`), "
        "`assistant` (summary, with `reasoning_content`).", "",
        "Rollout-level fields (`resolved`, `reward`, `patched_files`, `ran_tests`, `n_compactions`, `wall_seconds`, …) are identical on all segments of a rollout; "
        "`n_assistant_turns`, token counts and segment flags are per row.", "",
        "## How it was built", "",
        "```", "run_repo_sweep.py         (train split, AGL_SWEEP_IDS_FILE=<ids>, 8 shards x 10 workers)",
        "teacher_proxy.py          (logs every request/response; files main/compactK/postK segments per rollout)",
        "assemble_teacher_traj.py  SWEEP_ROOT PROXY_DIR -o traj.jsonl --tokenizer MiniCPM5-2B --run-name <run>", "```", "",
        "## Caveats", "",
        "- Only `resolved=true` rows are verified-correct; unresolved rows are kept for analysis / rejection-sampling baselines.",
        "- Tool outputs are what the teacher saw, including opencode's truncation of long outputs.",
        "- `batch1` rows with `post_segments_missing` are the pre-compaction part of a rollout whose continuation was lost; they are still "
        "valid `main` segments (the teacher's turns are genuine) but the rollout's final answer is not in the dataset. Which batch1 rollouts "
        "compacted was established from the proxy's captured compaction requests (`batch1_true_compacted_ids.json`, 210 problems), NOT from "
        "the sweep's `compacted` field: that field was a substring heuristic on the opencode event log and counted repo content such as "
        "funcy's `compact()` (28 of the 235 problems selected for the `rc` rerun, `rc_ids.json`, had in fact not compacted; 3 that did were "
        "not re-run). `rc` therefore gives 235 second rollouts (different samples, not copies) with the aligned threshold and all segments.",
        "- Val-split problems were **not** included; a separate val sweep of the same teacher is reported in the model card of "
        "`<hf-user>/MiniCPM5-2B-SWE-smith-GRPO` for difficulty comparison only.",
    ]
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
