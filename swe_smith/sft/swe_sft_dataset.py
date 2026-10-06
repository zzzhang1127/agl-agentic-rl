"""Custom verl SFT dataset for the SWE-smith teacher trajectories (MiniCPM5-2B chat template).

Why not verl's MultiTurnSFTDataset: it renders every message separately and concatenates, which
breaks with the MiniCPM template (bos token, grouped <tool_response> blocks, <think> handling).
Here the whole conversation is rendered once with apply_chat_template (exactly what vLLM fed the
student during GRPO), and the loss mask is computed from character spans: for assistant message i,
learn = [render(messages[:i], add_generation_prompt=True), render(messages[:i+1]) minus the trailing
newline). Synthetic messages (post-compaction summary re-injected by opencode) are never learned.

Parquet columns: messages_json (str), tools_json (str), instance_id, segment, run.
"""
import json
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

TRAIL = "<|im_end|>\n"


def normalize_messages(messages):
    """OpenAI-style messages -> what the MiniCPM template expects (arguments as dict, content str)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function") or {}
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a)
                    except Exception:
                        fn["arguments"] = {"_raw": a}
                tc["function"] = fn
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def learnable_indices(messages, segment):
    """Indices of assistant messages to learn. Post segments: the leading run of assistant messages
    after 'What did we do so far?' is opencode's re-injected summary (sometimes split in two: reasoning
    part + text part); only the last message of that run is the student's real first action."""
    idx = [i for i, m in enumerate(messages) if m["role"] == "assistant" and not m.get("synthetic")]
    if segment == "post":
        # leading run of assistant messages starting at index 2 (system, user, assistant...)
        j = 2
        run = []
        while j < len(messages) and messages[j]["role"] == "assistant":
            run.append(j)
            j += 1
        drop = set(run[:-1])  # keep only the last one (the real action)
        idx = [i for i in idx if i not in drop]
    return idx


def render(tokenizer, messages, tools, add_generation_prompt):
    return tokenizer.apply_chat_template(
        messages, tools=tools or None, add_generation_prompt=add_generation_prompt, tokenize=False
    )


def build_example(tokenizer, messages, tools, segment, max_length=None):
    """Return dict(input_ids, loss_mask, position_ids) for one conversation; raises on inconsistency."""
    messages = normalize_messages(messages)
    full = render(tokenizer, messages, tools, False)
    spans = []
    for i in learnable_indices(messages, segment):
        before = render(tokenizer, messages[:i], tools, True)
        after = render(tokenizer, messages[: i + 1], tools, False)
        if not full.startswith(before) or not full.startswith(after):
            raise ValueError(f"prefix mismatch at assistant message {i}")
        b = len(after)
        if after.endswith(TRAIL):
            b -= 1  # keep <|im_end|>, drop the newline after it
        spans.append((len(before), b))
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    loss = [0] * len(ids)
    k = 0
    spans.sort()
    for t, (s, e) in enumerate(offs):
        while k < len(spans) and spans[k][1] <= s:
            k += 1
        if k < len(spans) and spans[k][0] <= s < spans[k][1]:
            loss[t] = 1
    if max_length is not None and len(ids) > max_length:
        raise ValueError(f"sequence_length={len(ids)} > max_length={max_length}")
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "loss_mask": torch.tensor(loss, dtype=torch.long),
        "position_ids": torch.arange(len(ids), dtype=torch.long),
    }


class SWESmithSFTDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if not isinstance(parquet_files, (list, tuple)):
            parquet_files = [parquet_files]
        self.df = pd.concat([pd.read_parquet(p) for p in parquet_files]).reset_index(drop=True)
        if max_samples > 0:
            self.df = self.df.iloc[:max_samples]
        self.tokenizer = tokenizer
        self.max_length = int(config.get("max_length", 49152))
        assert config.get("pad_mode", "no_padding") == "no_padding"
        print(f"SWESmithSFTDataset: {len(self.df)} rows from {parquet_files}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return build_example(
            self.tokenizer, json.loads(r["messages_json"]), json.loads(r["tools_json"]), r["segment"], self.max_length
        )


if __name__ == "__main__":  # validation: python swe_sft_dataset.py <tokenizer_dir> <parquet> [n]
    import sys
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    df = pd.read_parquet(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else len(df)
    lens, ltoks, bad = [], [], 0
    for i in range(n):
        r = df.iloc[i]
        try:
            ex = build_example(tok, json.loads(r["messages_json"]), json.loads(r["tools_json"]), r["segment"])
        except Exception as e:
            bad += 1
            print("BAD", r["instance_id"], r["segment"], e)
            continue
        lens.append(len(ex["input_ids"]))
        ltoks.append(int(ex["loss_mask"].sum()))
        if i == 0:
            ids = ex["input_ids"].tolist()
            m = ex["loss_mask"].tolist()
            # show the first learned span and the boundary around it
            s = m.index(1)
            e = s
            while e < len(m) and m[e] == 1:
                e += 1
            print("ctx  <<", repr(tok.decode(ids[max(0, s - 12) : s])))
            print("learn<<", repr(tok.decode(ids[s:e]))[:400])
            print("next <<", repr(tok.decode(ids[e : e + 8])))
    lens.sort()
    print(f"rows={n} bad={bad} tokens: total={sum(lens)} max={lens[-1]} p50={lens[len(lens)//2]} "
          f"p90={lens[int(len(lens)*.9)]} >40960={sum(l>40960 for l in lens)} >49152={sum(l>49152 for l in lens)}; "
          f"loss tokens total={sum(ltoks)} ({sum(ltoks)/sum(lens):.1%})")
