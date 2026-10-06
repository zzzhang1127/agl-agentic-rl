#!/usr/bin/env python3
"""Join teacher-proxy session files with sweep results into SFT candidates (segment-aware, 2026-09-20 v2).

usage: assemble_teacher_traj.py SWEEP_ROOT PROXY_LOG_DIR [PROXY_LOG_DIR ...] -o OUT.jsonl
                                [--tokenizer /path/to/MiniCPM] [--max-turn-tokens 6144]
                                [--max-turns 60] [--max-context-tokens 38912] [--run-name batch1]

One opencode rollout = one problem.  opencode compacts the context when it exceeds
context-output tokens; every compaction splits the rollout into segments that the proxy
(since the 2026-09-20 fix) files separately:
  <task>.json           pre-compaction part (or the whole rollout when it never compacted)
  <task>.compactK.json  K-th compaction request: summarizer system prompt + rendered
                        conversation -> structured summary (this IS the compaction skill)
  <task>.postK.json     continuation after compaction K: system + "What did we do so far?"
                        + assistant(summary, synthetic) + new turns
Each segment becomes one row.  The student is trained on assistant turns of every segment
(mask the synthetic summary message in post segments: it is the *input* there, and it is the
*target* in the compact row).

Row fields: instance_id, run, segment (main|compact|post), segment_index, n_segments,
  n_compactions, resolved, reward, n_f2p_pass/n_f2p, n_p2p_ok/n_p2p, reason, opencode_rc,
  wall_seconds, compacted, n_assistant_turns, n_tool_calls, tool_names, chars,
  total_tokens(_est), max_turn_tokens, n_turns_with_reasoning, reasoning_chars,
  patched_files, test_files_edited, ran_tests (rollout-level), flags, teacher_finish,
  teacher_model, messages, tool_schemas.
Flags (all rows are emitted; filter downstream):
  rollout-level: edits_tests, never_ran_tests, post_segments_missing (compacted before the
                 proxy fix: continuation context lost), over_turn_cap (total assistant turns
                 across segments > --max-turns)
  segment-level: long_turn (> --max-turn-tokens), over_context (> --max-context-tokens),
                 no_final_answer (only for the LAST segment of a rollout: it must end with a
                 text answer, finish_reason == stop)
Token counts use the given tokenizer (MiniCPM) when --tokenizer is set, else chars/3.7.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_repo_sweep as sweep  # noqa: E402
os.environ.setdefault("TEACHER_BASE_URL", "http://x"); os.environ.setdefault("TEACHER_API_KEY", "x")
sys.argv_backup, sys.argv = sys.argv, ["teacher_proxy.py", "0", "/tmp/claude-0/teacher_proxy_import"]
import teacher_proxy as proxy  # noqa: E402
sys.argv = sys.argv_backup

TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)(test_[^/]*\.py|[^/]*_tests?\.py|conftest\.py)$")
RAN_TESTS_RE = re.compile(r"\b(pytest|py\.test|unittest|tox|nose2?|make\s+test|npm\s+test|cargo\s+test|go\s+test)\b")
SEG_RE = re.compile(r"^([0-9a-f]{32})(?:\.(compact|post)(\d+))?\.json$")


def _text(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


def make_counter(tok_path):
    if not tok_path:
        return lambda s: int(len(s) / 3.7 + 0.5)
    from transformers import AutoTokenizer  # noqa: WPS433
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    return lambda s: len(tok.encode(s, add_special_tokens=False)) if s else 0


def turn_text(m):
    """Everything the student would have to generate for this assistant turn."""
    parts = [m.get("reasoning_content") or "", _text(m)]
    for tc in m.get("tool_calls") or []:
        parts.append(tc["function"]["name"] + tc["function"]["arguments"])
    return "\n".join(parts)


def msg_text(m):
    if m.get("role") == "assistant":
        return turn_text(m)
    return _text(m)


def patched_files(diff_text):
    files = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                files.append(parts[3][2:] if parts[3].startswith("b/") else parts[3])
        elif line.startswith("+++ ") and not files:
            p = line[4:].strip()
            files.append(p[2:] if p.startswith("b/") else p)
    return sorted(set(files))


def load_sessions(proxy_dirs, since=0.0, until=1e18, main_only=False):
    """key -> {"main": path, "compact": {k: path}, "post": {k: path}}; key = md5(norm first user msg)
    for main files (works for pilot-era raw-md5 names too), stem prefix for segment files.
    since/until: only session files whose request ts lies in the window (runs share the proxy dir)."""
    sessions = {}
    for d in proxy_dirs:
        for p in Path(d).glob("*.json"):
            m = SEG_RE.match(p.name)
            if not m:
                continue
            stem, kind, k = m.group(1), m.group(2), m.group(3)
            if main_only and kind is not None:
                continue
            if since > 0 or until < 1e18:
                mt = p.stat().st_mtime
                if mt < since - 3600 or mt > until + 3600:      # cheap pre-filter, then exact ts
                    continue
                try:
                    ts = json.loads(p.read_text(encoding="utf-8")).get("ts") or mt
                except Exception:
                    continue
                if not (since <= ts <= until):
                    continue
            if kind is None:
                try:
                    s = json.loads(p.read_text(encoding="utf-8"))
                    u = next(x for x in s["request"]["messages"] if x.get("role") == "user")
                    key = hashlib.md5(proxy.norm_prompt(_text(u)).encode("utf-8")).hexdigest()
                except Exception:
                    continue
                sessions.setdefault(key, {"main": None, "compact": {}, "post": {}})["main"] = p
            else:
                sessions.setdefault(stem, {"main": None, "compact": {}, "post": {}})[kind][int(k)] = p
    return sessions


def load_turn_capped(proxy_dirs, since=0.0, until=1e18):
    """Task keys of rollouts the proxy stopped at TEACHER_MAX_TURNS (events.jsonl `turn_cap` rows, run ra+)."""
    capped = set()
    for d in proxy_dirs:
        p = Path(d) / "events.jsonl"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if '"turn_cap"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("event") == "turn_cap" and since <= float(r.get("ts") or 0) <= until:
                    capped.add(str(r.get("stem") or "")[:32])
    return capped


def segment_messages(path):
    s = json.loads(path.read_text(encoding="utf-8"))
    msgs = list(s["request"]["messages"])
    final = (s.get("response") or {}).get("message")
    if final:
        msgs.append(final)
    return s, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_root")
    ap.add_argument("proxy_dirs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer dir (MiniCPM); default = chars/3.7 estimate")
    ap.add_argument("--max-turn-tokens", type=int, default=6144)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-context-tokens", type=int, default=38912)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--since", type=float, default=0.0, help="only session files with request ts >= this epoch")
    ap.add_argument("--until", type=float, default=1e18)
    ap.add_argument("--main-only", action="store_true", help="ignore compact/post segment files (pre-fix runs)")
    ap.add_argument("--compacted-ids", default=None,
                    help="JSON list of instance_ids known (from the proxy's compaction requests) to have compacted in a "
                         "pre-fix run: they get post_segments_missing and n_compactions=null (count unknown)")
    args = ap.parse_args()
    known_compacted = set(json.loads(Path(args.compacted_ids).read_text())) if args.compacted_ids else set()
    root = Path(args.sweep_root)
    ntok = make_counter(args.tokenizer)
    tok_field = "total_tokens" if args.tokenizer else "total_tokens_est"
    sessions = load_sessions(args.proxy_dirs, args.since, args.until, args.main_only)
    turn_capped = load_turn_capped(args.proxy_dirs, args.since, args.until)

    n_rollouts = n_match = n_resolved = n_rows = 0
    flag_counts, seg_counts = Counter(), Counter()
    clean_rollouts = 0
    missing = []
    with open(args.out, "w", encoding="utf-8") as out:
        for res_path in sorted(root.glob("*/result.json")):
            n_rollouts += 1
            res = json.loads(res_path.read_text(encoding="utf-8"))
            inst = json.loads((res_path.parent / "instance.json").read_text(encoding="utf-8"))
            key = hashlib.md5(sweep.prompt_text(inst).strip().encode("utf-8")).hexdigest()
            sess = sessions.get(key)
            ev = res.get("eval") or {}
            if sess is None or sess["main"] is None:
                missing.append(res["instance_id"])
                continue
            n_match += 1
            resolved = bool(ev.get("resolved"))
            n_resolved += resolved
            # compaction ground truth = the proxy's compactK segment files (result.json's compact_events was a
            # substring heuristic on repo content and is NOT used); pre-fix runs pass --compacted-ids instead
            n_comp = len(sess["compact"])
            if res["instance_id"] in known_compacted and not sess["compact"]:
                n_comp = None  # compacted, count unknown (pre-fix proxy)
            compacted = n_comp is None or n_comp > 0
            # ordered segments: main, compact1, post1, compact2, post2, ...
            segs = [("main", 0, sess["main"])]
            ks = sorted(set(sess["compact"]) | set(sess["post"]))
            for k in ks:
                if k in sess["compact"]:
                    segs.append(("compact", k, sess["compact"][k]))
                if k in sess["post"]:
                    segs.append(("post", k, sess["post"][k]))
            loaded = [(kind, k, *segment_messages(p)) for kind, k, p in segs]
            for kind, k, s, msgs in loaded:
                if kind == "post":
                    # the summary is the 1st assistant message right after "What did we do so far?":
                    # input to the student here (target only in the compact row)
                    for m in msgs:
                        if m.get("role") == "assistant":
                            m["synthetic"] = True
                            break
            # rollout-level facts
            diff_path = res_path.parent / "patch.diff"
            files = patched_files(diff_path.read_text(encoding="utf-8", errors="replace")) if diff_path.exists() else []
            test_files = [f for f in files if TEST_PATH_RE.search(f)]
            all_asst = [m for kind, k, s, msgs in loaded if kind != "compact" for m in msgs if m.get("role") == "assistant" and not m.get("synthetic")]
            all_tc = [tc for m in all_asst for tc in (m.get("tool_calls") or [])]
            ran_tests = any(RAN_TESTS_RE.search(tc["function"]["arguments"]) for tc in all_tc if tc["function"]["name"] == "bash")
            rflags = []
            if test_files:
                rflags.append("edits_tests")
            if not ran_tests:
                rflags.append("never_ran_tests")
            # a compaction request that failed (no finish_reason; gateway 429/timeout) is retried by opencode as
            # the next compactK, so it legitimately has no postK: only a SUCCESSFUL compaction without its post
            # segment means the continuation was lost
            ok_compact = {k for kind, k, s, msgs in loaded if kind == "compact" and (s.get("response") or {}).get("finish_reason") == "stop"}
            if n_comp is None or (ok_compact - set(sess["post"])):
                rflags.append("post_segments_missing")
            # > cap by counting, or stopped BY the proxy's cap (run ra+: the 61st model call was refused, so the
            # rollout has <= 60 assistant turns but did not finish on its own)
            if len(all_asst) > args.max_turns or key in turn_capped:
                rflags.append("over_turn_cap")
            rollout_clean = resolved and not rflags
            for si, (kind, k, s, msgs) in enumerate(loaded):
                # batch1 layout: the rollout continued after compaction but that part was lost, so the
                # main segment is NOT the end of the rollout -> no_final_answer must not be judged on it
                is_last = si == len(loaded) - 1 and "post_segments_missing" not in rflags
                asst =[m for m in msgs if m.get("role") == "assistant" and not m.get("synthetic")]
                tool_calls = [tc for m in asst for tc in (m.get("tool_calls") or [])]
                turn_tokens = [ntok(turn_text(m)) for m in asst]
                total_tokens = sum(ntok(msg_text(m)) for m in msgs)
                finish = (s.get("response") or {}).get("finish_reason")
                final = msgs[-1] if msgs and msgs[-1].get("role") == "assistant" else None
                flags = list(rflags)
                if turn_tokens and max(turn_tokens) > args.max_turn_tokens:
                    flags.append("long_turn")
                if total_tokens > args.max_context_tokens:
                    flags.append("over_context")
                if is_last and kind != "compact" and (not final or not _text(final).strip() or finish != "stop"):
                    flags.append("no_final_answer")
                if kind == "compact" and (not final or not _text(final).strip() or finish != "stop"):
                    flags.append("no_final_answer")
                if flags and kind != "compact":
                    rollout_clean = rollout_clean and not [f for f in flags if f not in rflags]
                flag_counts.update(flags)
                seg_counts[kind] += 1
                n_rows += 1
                row = {
                    "instance_id": res["instance_id"],
                    "run": args.run_name,
                    "segment": kind,
                    "segment_index": k,
                    "n_segments": len(loaded),
                    "n_compactions": n_comp,
                    "repo": inst.get("repo"),
                    "image_name": inst.get("image_name"),
                    "resolved": resolved,
                    "reward": ev.get("reward"),
                    "n_f2p_pass": ev.get("n_f2p_pass"),
                    "n_f2p": ev.get("n_f2p"),
                    "n_p2p_ok": ev.get("n_p2p_ok"),
                    "n_p2p": ev.get("n_p2p"),
                    "reason": ev.get("reason"),
                    "opencode_rc": res.get("opencode_rc"),
                    "wall_seconds": res.get("wall_seconds"),
                    "compacted": compacted,
                    "n_assistant_turns": len(asst),
                    "n_tool_calls": len(tool_calls),
                    "tool_names": sorted(set(tc["function"]["name"] for tc in tool_calls)),
                    "chars": sum(len(msg_text(m)) for m in msgs),
                    tok_field: total_tokens,
                    "max_turn_tokens": max(turn_tokens) if turn_tokens else 0,
                    "n_turns_with_reasoning": sum(1 for m in asst if m.get("reasoning_content")),
                    "reasoning_chars": sum(len(m.get("reasoning_content") or "") for m in asst),
                    "patched_files": files,
                    "test_files_edited": test_files,
                    "ran_tests": ran_tests,
                    "flags": flags,
                    "teacher_finish": finish,
                    "teacher_model": s.get("model"),
                    "messages": msgs,
                    "tool_schemas": s["request"].get("tool_schemas"),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            clean_rollouts += rollout_clean
    print(f"rollouts={n_rollouts} matched={n_match} resolved={n_resolved} rows={n_rows} segments={dict(seg_counts)} "
          f"clean_resolved_rollouts(no flags in any non-compact segment)={clean_rollouts} unmatched={len(missing)} -> {args.out}")
    print("flags:", dict(flag_counts.most_common()))
    if missing:
        print("unmatched e.g.", missing[:5])


if __name__ == "__main__":
    main()
