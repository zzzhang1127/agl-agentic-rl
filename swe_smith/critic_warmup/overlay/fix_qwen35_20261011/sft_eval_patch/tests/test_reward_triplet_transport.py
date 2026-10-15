"""Triplet projection must retain evaluator approval and explicit rejection."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('transport_reward_guard', ROOT / 'agentlightning/verl/reward_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
source = ROOT / 'agentlightning/server/routes/events.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_trim_reward')
ns = {'Any': object}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
trim = ns['_trim_reward']


def rollout(data):
    return SimpleNamespace(rollout_id='fixture', final_reward=0.0, error_message=None,
                           triplet_events=[{'event_type': 'reward', 'data': trim(data)}])


def test_approved_reward_survives_triplet_projection():
    data = {'value': 0.0, 'reward_eligible': True, 'evaluation_state': 'complete',
            'trajectory_terminal': True, 'termination_reason': 'stop',
            'grader_protocol': 'pytest-structured-v1', 'p2p_scope': 'f2p_same_files',
            'report_sha256': 'fixture-hash', 'unneeded_large_payload': 'omit'}
    projected = trim(data)
    assert 'unneeded_large_payload' not in projected
    assert all(projected[key] == value for key, value in data.items() if key != 'unneeded_large_payload')
    guard.require_explicit_training_rewards([rollout(data)], require_evaluator_approval=True)


@pytest.mark.parametrize('eligible', [False, None, 'true'])
def test_rejection_is_not_lost_or_coerced(eligible):
    data = {'value': 0.0, 'reward_eligible': eligible, 'evaluation_state': 'unavailable'}
    assert trim(data)['reward_eligible'] is eligible
    with pytest.raises(ValueError, match='rejected'):
        guard.require_explicit_training_rewards([rollout(data)], require_evaluator_approval=True)


def test_legacy_scalar_is_not_given_synthetic_approval():
    assert trim({'value': 0.0}) == {'value': 0.0}
    with pytest.raises(ValueError, match='approval'):
        guard.require_explicit_training_rewards([rollout({'value': 0.0})], require_evaluator_approval=True)
