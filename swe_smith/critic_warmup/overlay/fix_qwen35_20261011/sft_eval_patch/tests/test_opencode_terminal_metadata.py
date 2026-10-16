import importlib.util
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[3]/'examples/swe_smith/agents/opencode_terminal.py'
spec=importlib.util.spec_from_file_location('terminal_metadata',PATH)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('agent,summary,finish,expected',[
    ('build',False,'stop','task_stop'),('compaction',True,'stop','compaction'),
    ('build',True,'stop','compaction'),('compaction',False,'stop','compaction'),
    ('title',False,'stop','unknown'),('build',False,'tool-calls','unknown')])
def test_message_agent_not_step_stop(agent,summary,finish,expected):
    event={'type':'step_finish','sessionID':'ses_a','part':{'messageID':'msg_a','reason':'stop'}}
    info={'id':'msg_a','sessionID':'ses_a','role':'assistant','agent':agent,'summary':summary,'finish':finish}
    assert module.classify_exported_message(event,{'messages':[{'info':info}]})==expected
    info['sessionID']='ses_other'
    assert module.classify_exported_message(event,{'messages':[{'info':info}]})=='unknown'


def test_missing_or_duplicate_metadata_fails_closed():
    event={'type':'step_finish','sessionID':'ses_a','part':{'messageID':'msg_a','reason':'stop'}}
    assert module.classify_exported_message(event,{'messages':[]})=='unknown'
    info={'id':'msg_a','sessionID':'ses_a','role':'assistant','agent':'build','finish':'stop'}
    assert module.classify_exported_message(event,{'messages':[{'info':info},{'info':info}]})=='unknown'


def test_export_failure_fails_closed(monkeypatch):
    def fail(*args,**kwargs):
        raise module.subprocess.TimeoutExpired('export',5)
    monkeypatch.setattr(module.subprocess,'run',fail)
    event={'type':'step_finish','sessionID':'ses_a','part':{'messageID':'msg_a','reason':'stop'}}
    assert module.classify_terminal_event(event,binary='opencode',env={},cwd='.')=='unknown'


@pytest.mark.parametrize('summary',[False,True])
def test_long_export_uses_regular_file(monkeypatch,tmp_path,summary):
    from types import SimpleNamespace
    import json
    event={'type':'step_finish','sessionID':'ses_a','part':{'messageID':'msg_a','reason':'stop'}}
    payload={'messages':[{'info':{'id':'msg_a','sessionID':'ses_a','role':'assistant',
                                 'agent':'compaction' if summary else 'build',
                                 'summary':summary,'finish':'stop'},'parts':['x'*100000]}]}
    def export(*args,**kwargs):
        assert kwargs['stdin'] is module.subprocess.DEVNULL
        assert kwargs['stdout'] is not module.subprocess.PIPE
        kwargs['stdout'].write(json.dumps(payload).encode())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',export)
    assert module.classify_terminal_event(event,binary='opencode',env={'TMPDIR':str(tmp_path)},cwd='.')==('compaction' if summary else 'task_stop')
