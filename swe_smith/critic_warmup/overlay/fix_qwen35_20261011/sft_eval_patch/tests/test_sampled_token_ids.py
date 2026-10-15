"""Sampled IDs must come from the same ordered records as behavior logprobs."""
import ast
import copy
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / 'agentlightning/server/proxy.py'
tree = ast.parse(PATH.read_text(encoding='utf-8'))
nodes = [
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name in ("_parse_sampled_token_id", "_sampled_token_ids")
]
ns = {'Any': object}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), ns)
extract = ns['_sampled_token_ids']


def choice():
    return {'message': {'content': 'parsed text, not raw XML'}, 'logprobs': {'content': [
        {'token': 'token_id:12', 'logprob': -0.1},
        {'token': 'token_id:0', 'logprob': -0.2},
        {'token': 'token_id:345', 'logprob': -0.3}]}}


def test_exact_sample_order_including_zero():
    c = choice()
    before = copy.deepcopy(c)
    assert extract(c) == [12, 0, 345]
    assert c == before
    c['token_ids'] = [12, 0, 345]
    assert extract(c) == c['token_ids']


@pytest.mark.parametrize('token', ['parsed', 'token_id:-1', 'token_id:1.5', 'token_id:', 'token_id:１２', None])
def test_no_guessed_or_malformed_tokens(token):
    c = choice()
    c['logprobs']['content'][0]['token'] = token
    with pytest.raises(ValueError):
        extract(c)


def test_existing_reencoded_ids_cannot_override_sampled_ids():
    c = choice()
    c['token_ids'] = [999]
    with pytest.raises(ValueError, match='disagree'):
        extract(c)


@pytest.mark.parametrize('c', [{}, {'logprobs': None}, {'logprobs': {'content': []}}])
def test_missing_logprobs_rejected(c):
    with pytest.raises(ValueError):
        extract(c)
