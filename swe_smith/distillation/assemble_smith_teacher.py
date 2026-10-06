#!/usr/bin/env python3
"""Assemble teacher trajectories collected with the OFFICIAL smith harness (run_smith_sweep.py ->
smith_rollout.py/smith_agent.py, plain-text bash-block protocol, no tools, no compaction) into SFT rows.

2026-10-06, user: 「蒸馏同步进行,增大并发度,使用AGL的简易harness」+「deepseekv4flash做错的题,就没必要参与训练了」
=> rejection sampling: ONLY rollouts whose patch RESOLVED the instance (eval.resolved == True) are kept.

Sources
  <root>/<instance_id>/{instance.json,result.json,status.json}   (harness; result has eval.resolved)
  <proxy>/<md5(first user msg)>.json                              (teacher_proxy_smith.py: LAST request of the
                                                                    rollout = full history, + its response)
  <proxy>/events.jsonl                                            (per-request finish_reason, retries, errors)
The full conversation = session.request.messages + [session.response.message] (content only; the teacher's
reasoning_content is dropped - the student has no thinking channel in this protocol).

Output: jsonl rows {instance_id, run, segment:"main", segment_index:0, resolved, flags, messages, tool_schemas:[],
n_turns, n_tokens, reward}, consumable by sft/prep_sft_data.py (-> train.parquet) and sft/swe_sft_dataset.py.

Drops (counted in the summary): unresolved / not submitted / timed_out / overflowed / harness error / session
missing / prompt mismatch (system or instance prompt not byte-identical to the student's) / role pattern broken /
empty assistant message (an upstream HTTP error burned the turn) / n_assistant != n_turns / any turn finished
with 'length' (truncated output) / no submit in the last turn / longer than --max-tokens (MiniCPM tokenizer,
chat template rendered; falls back to chars/3 when transformers is unavailable).
"""
import argparse, hashlib, json, os, re, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smith_agent  # the smoke copy is byte-identical to examples/swe_smith/agents/smith_agent.py

SUBMIT_MARKER = smith_agent.SUBMIT_MARKER  # 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' (a \bsubmit\b regex misses it)
FORMAT_ERR_PREFIX = "Format error:"
# deepseek-v4-flash quirk (1.3% of turns, finish=stop): the whole answer incl. the bash block lands in
# reasoning_content and content is ''. The harness then appends the empty assistant turn + a
# "Format error: ... found 0" user turn and the model simply redoes the action next turn. Such
# (malformed assistant, Format error user) pairs are spliced out so the SFT target never teaches
# "emit nothing"; the remaining conversation stays coherent (next action answers the previous observation).
MAX_SPLICE = 3

# deepseek-v4-flash output hygiene (measured on 12k batch-5 turns, 2026-10-06 16:5x):
#   * ~35% of turns carry pseudo-XML closers after the bash block ("</parameter>", "</invoke>",
#     "</｜｜DSML｜｜ calls>", "</format_example>", ...), 4% continue with a hallucinated next step after it;
#   * the harness only ever executes parse_action(content) == the single ```bash fence; everything after the
#     closing fence is ignored and pure-tag lines before it are noise.
# normalize_assistant() keeps THOUGHT text (minus pure-tag lines) + the fence and drops the tail. It is
# applied only when parse_action() of the result equals parse_action() of the original, so the executed
# command - and therefore the observation that follows - is unchanged.
_FENCE_RE = re.compile(r"```(?:bash|mswea_bash_command)[^\S\n]*\n.*?\n?```", re.DOTALL)
_TAG_LINE_RE = re.compile(r"^\s*(?:</?[A-Za-z_][A-Za-z0-9_:-]*(?:\s[^>\n]*)?>|</?｜｜DSML｜｜[^>\n]*>|</?｜[^>\n]*>)\s*$")
_THOUGHT_TAG_RE = re.compile(r"</?THOUGHT>|<parameter name=\"THOUGHT\">")
_JUNK_IN_CMD_RE = re.compile(r"</?parameter|DSML|</?invoke|</?antml")


def parse_or_none(content):
    try:
        return smith_agent.parse_action(content)
    except Exception:  # FormatError / UnavailableToolError
        return None


def normalize_assistant(content):
    """-> (normalized_content, n_chars_dropped_after_fence, n_tag_lines_dropped). Never changes the action."""
    c = content.strip()
    spans = [m.span() for m in _FENCE_RE.finditer(c)]
    if len(spans) != 1:
        return c, 0, 0
    a, b = spans[0]
    keep, n_tag = [], 0
    for ln in _THOUGHT_TAG_RE.sub("", c[:a]).split("\n"):
        if _TAG_LINE_RE.match(ln):
            n_tag += 1
        else:
            keep.append(ln)
    pre = re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()
    out = (pre + "\n\n" if pre else "") + c[a:b]
    tail = c[b:].strip()
    if parse_or_none(out) != parse_or_none(c):
        return c, 0, 0
    return out, len(tail), n_tag


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def text(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return "" if c is None else str(c)


def load_events(path):
    """session -> dict(finish=[...], n_retry, n_err)"""
    out = collections.defaultdict(lambda: {"finish": [], "n_retry": 0, "n_err": 0, "n_req": 0})
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            s = r.get("session")
            if not s:
                continue
            d = out[s]
            ev = r.get("event")
            if ev == "retry":
                d["n_retry"] += 1
                continue
            if ev == "upstream_error" or (r.get("status") and r["status"] != 200):
                d["n_err"] += 1
                continue
            d["n_req"] += 1
            resp = r.get("response") or {}
            if isinstance(resp, dict):
                d["finish"].append(resp.get("finish_reason"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="smith_b5")
    ap.add_argument("--tokenizer", default="/workspace/models/MiniCPM5-2B-sft-v3-ep3")
    ap.add_argument("--max-tokens", type=int, default=45000)
    ap.add_argument("--keep-unsubmitted", action="store_true", help="keep resolved rollouts that never submitted")
    args = ap.parse_args()

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    except Exception as e:  # noqa
        print(f"WARN tokenizer unavailable ({str(e)[:80]}); using chars/3 estimate", file=sys.stderr)

    def n_tokens(messages):
        if tok is None:
            return sum(len(text(m)) for m in messages) // 3
        s = tok.apply_chat_template(messages, tools=None, add_generation_prompt=False, tokenize=False)
        return len(tok(s, add_special_tokens=False)["input_ids"])

    events = load_events(os.path.join(args.proxy, "events.jsonl"))
    drops = collections.Counter()
    kept = []
    n_dirs = 0
    for d in sorted(os.listdir(args.root)):
        p = os.path.join(args.root, d)
        rp = os.path.join(p, "result.json")
        if not os.path.isfile(rp):
            continue
        n_dirs += 1
        res = json.load(open(rp, encoding="utf-8"))
        inst = json.load(open(os.path.join(p, "instance.json"), encoding="utf-8"))
        try:
            st = json.load(open(os.path.join(p, "status.json"), encoding="utf-8"))
        except Exception:
            st = {}
        ev = res.get("eval") or {}
        if not ev.get("resolved"):
            drops["unresolved"] += 1
            continue
        if res.get("timed_out") or ev.get("timed_out"):
            drops["timed_out"] += 1
            continue
        if res.get("overflowed") or st.get("overflowed"):
            drops["overflowed"] += 1
            continue
        if st.get("error"):
            drops["harness_error"] += 1
            continue
        if not st.get("submitted") and not args.keep_unsubmitted:
            drops["not_submitted"] += 1
            continue
        # smith_agent.main() does AGL_TASK_INPUT.strip() before formatting -> same here, else md5 differs
        user0 = smith_agent.INSTANCE_PROMPT.format(problem_statement=inst["problem_statement"].strip())
        key = md5(user0.strip())
        sp = os.path.join(args.proxy, f"{key}.json")
        if not os.path.isfile(sp):
            drops["session_missing"] += 1
            continue
        sess = json.load(open(sp, encoding="utf-8"))
        msgs = list(sess["request"]["messages"] or [])
        last = (sess.get("response") or {}).get("message") or {}
        msgs.append({"role": "assistant", "content": last.get("content") or ""})
        msgs = [{"role": m["role"], "content": text(m)} for m in msgs]
        if len(msgs) < 3 or msgs[0]["role"] != "system" or msgs[0]["content"] != smith_agent.SYSTEM_PROMPT \
                or msgs[1]["role"] != "user" or msgs[1]["content"] != user0:
            drops["prompt_mismatch"] += 1
            continue
        ok = True
        for i, m in enumerate(msgs[2:], start=2):
            want = "assistant" if i % 2 == 0 else "user"
            if m["role"] != want:
                ok = False
                break
        if not ok or msgs[-1]["role"] != "assistant":
            drops["role_pattern"] += 1
            continue
        n_assist = sum(1 for m in msgs if m["role"] == "assistant")
        if st.get("n_turns") is not None and n_assist != st["n_turns"]:
            drops["n_turns_mismatch"] += 1
            continue
        # splice out (unparseable assistant, harness "Format error:" user) pairs; the harness replies with a
        # format error iff parse_action() raised, so both sides must agree or the transcript is not what we think
        spliced = 0
        clean = msgs[:2]
        i = 2
        bad = None
        while i < len(msgs):
            m = msgs[i]
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            if m["role"] == "assistant":
                action = parse_or_none(m["content"])
                nxt_is_fmt = nxt is not None and nxt["content"].lstrip().startswith(FORMAT_ERR_PREFIX)
                if action is None and nxt_is_fmt:
                    spliced += 1
                    i += 2
                    continue
                if action is None:
                    bad = "unparseable_unanswered"  # last turn or a reply we do not understand
                    break
                if nxt_is_fmt:
                    bad = "format_error_on_parsed_turn"
                    break
            clean.append(m)
            i += 1
        if bad:
            drops[bad] += 1
            continue
        if spliced > MAX_SPLICE:
            drops["too_many_format_errors"] += 1
            continue
        msgs = clean
        if any(m["role"] == "assistant" and not m["content"].strip() for m in msgs):
            drops["empty_assistant_remaining"] += 1
            continue
        # output hygiene (action-preserving, see normalize_assistant)
        trimmed_chars = tag_lines = junk_cmd = 0
        for m in msgs:
            if m["role"] != "assistant":
                continue
            m["content"], t, g = normalize_assistant(m["content"])
            trimmed_chars += t
            tag_lines += g
            if _JUNK_IN_CMD_RE.search(parse_or_none(m["content"]) or ""):
                junk_cmd += 1
        n_assist = sum(1 for m in msgs if m["role"] == "assistant")
        evs = events.get(key)
        flags = [f"spliced{spliced}"] if spliced else []
        if trimmed_chars:
            flags.append(f"trimmed{trimmed_chars}")
        if tag_lines:
            flags.append(f"taglines{tag_lines}")
        if junk_cmd:
            flags.append(f"junkcmd{junk_cmd}")
        if evs:
            if any(f == "length" for f in evs["finish"]):
                drops["length_finish"] += 1
                continue
            if evs["n_err"]:
                flags.append("upstream_err")
            if evs["n_retry"]:
                flags.append("retried")
        if SUBMIT_MARKER not in (parse_or_none(msgs[-1]["content"]) or ""):
            drops["no_submit_last"] += 1
            continue
        nt = n_tokens(msgs)
        if nt > args.max_tokens:
            drops["too_long"] += 1
            continue
        kept.append({
            "instance_id": inst["instance_id"], "run": args.run, "segment": "main", "segment_index": 0,
            "resolved": True, "flags": flags, "messages": msgs, "tool_schemas": [],
            "n_turns": n_assist, "n_tokens": nt, "reward": ev.get("reward"),
            "repo": inst.get("repo"), "image_name": inst.get("image_name"),
        })
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    toks = sorted(r["n_tokens"] for r in kept)
    turns = sorted(r["n_turns"] for r in kept)
    print(f"dirs_with_result={n_dirs} kept={len(kept)} drops={dict(drops)}")
    if kept:
        def nrows(prefix):
            return sum(1 for r in kept if any(f.startswith(prefix) for f in r["flags"]))
        print(f"n_tokens p50={toks[len(toks)//2]} p90={toks[int(len(toks)*.9)]} max={toks[-1]} "
              f"turns p50={turns[len(turns)//2]} max={turns[-1]} flagged={sum(1 for r in kept if r['flags'])} "
              f"spliced_rows={nrows('spliced')} trimmed_rows={nrows('trimmed')} tagline_rows={nrows('taglines')} "
              f"junkcmd_rows={nrows('junkcmd')} total_turns={sum(turns)} "
              f"spliced_turns={sum(int(f[7:]) for r in kept for f in r['flags'] if f.startswith('spliced'))}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
