"""Behavior logprobs must drive response token ids, not parsed assistant text."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PATH = Path(__file__).resolve().parents[3] / "agentlightning/server/proxy.py"
tree = ast.parse(PATH.read_text(encoding="utf-8"))
names = [
    "_behavior_logprob_token_ids",
    "_token_ids_from_vocab_logprobs",
    "_parse_sampled_token_id",
    "_sampled_token_ids",
]
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
ns: dict = {"Any": object, "log": SimpleNamespace(warning=lambda *a, **k: None), "_TOKENIZERS": {}}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), "exec"), ns)
behavior = ns["_behavior_logprob_token_ids"]
vocab = ns["_token_ids_from_vocab_logprobs"]


class _Tok:
    unk_token_id = 0

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        mapping = {"I": 10, "'ll": 11, "Ġlocate": 12}
        if token not in mapping:
            return 0
        return mapping[token]


def test_vocab_logprobs_match_content_length():
    choice = {
        "logprobs": {
            "content": [
                {"token": "I", "logprob": -0.1},
                {"token": "'ll", "logprob": -0.2},
                {"token": "Ġlocate", "logprob": -0.3},
            ]
        },
        "token_ids": [999],
    }
    before = copy.deepcopy(choice)
    with patch.dict(ns["_TOKENIZERS"], {"m": _Tok()}, clear=False):
        assert vocab(choice, "m") == [10, 11, 12]
    assert choice == before


def test_behavior_prefers_vocab_when_no_token_id_prefix():
    choice = {
        "message": {"content": "short parsed text"},
        "logprobs": {"content": [{"token": "I", "logprob": -0.1}]},
    }
    with patch.dict(ns["_TOKENIZERS"], {"m": _Tok()}, clear=False):
        assert behavior(choice, "m") == [10]
