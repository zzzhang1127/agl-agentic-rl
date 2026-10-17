import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('swanlab_health', Path(__file__).parents[1] / 'swanlab_health.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_secret_values_are_flagged_without_exporting_them():
    result = module.inspect_redaction({'config': {'nested': [{'AGL_KEY': 'private-value'}], 'max_tokens': 6144}})
    assert result['sensitive_field_count'] == 1
    assert result['unsafe_field_names'] == ['config.nested[0].AGL_KEY']
    assert 'private-value' not in json.dumps(result)


def test_exact_redaction_markers_and_empty_fields_are_safe():
    result = module.inspect_redaction({'config': json.dumps({'api_key': '[REDACTED]', 'password': '', 'secret': '***'})})
    assert not result['unsafe_field_names']
    assert module.inspect_redaction({'config': {'api_key': 'not-redacted-secret'}})['unsafe_field_names']


def test_readback_selects_metrics_and_never_exports_profile_values():
    class Frame:
        def tail(self, count):
            assert count == 2
            return self
        def to_json(self, orient):
            assert orient == 'index'
            return '{"3":{"actor/lr":0.000002},"4":{"actor/lr":0.000002}}'
    def metrics(keys):
        assert keys == list(module.METRIC_KEYS)
        return Frame()
    run = SimpleNamespace(id='run-id', state='RUNNING', metrics=metrics,
                          profile=SimpleNamespace(config={'api_key': 'never-export-this'}, metadata={'host': 'private-host'}))
    report = module.read_run(run, 2)
    encoded = json.dumps(report)
    assert len(report['metrics']) == 2
    assert 'never-export-this' not in encoded and 'private-host' not in encoded
    assert report['redaction']['unsafe_field_names'] == ['config.api_key']


def test_unparsed_profile_is_not_claimed_safe():
    assert module.inspect_redaction({'config': 'not-json'})['unsafe_field_names'] == ['config.unparsed_profile']

def test_missing_profile_section_is_explicitly_unverified():
    result = module.inspect_redaction({'config': None, 'metadata': {}})
    assert result['missing_sections'] == ['config']
