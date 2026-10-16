"""Disambiguate OpenCode task completion from auxiliary compaction steps."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from typing import Any


def classify_exported_message(event: dict[str, Any], exported: dict[str, Any]) -> str:
    part = event.get('part', {})
    sid, mid = event.get('sessionID'), part.get('messageID')
    if event.get('type') != 'step_finish' or part.get('reason') != 'stop' or not sid or not mid:
        return 'unknown'
    matches = [x.get('info', {}) for x in exported.get('messages', [])
               if isinstance(x, dict) and x.get('info', {}).get('id') == mid]
    if len(matches) != 1:
        return 'unknown'
    info = matches[0]
    if info.get('sessionID') != sid or info.get('role') != 'assistant':
        return 'unknown'
    if info.get('summary') is True or info.get('agent') == 'compaction':
        return 'compaction'
    if info.get('agent') == 'build' and info.get('finish') == 'stop' and not info.get('error'):
        return 'task_stop'
    return 'unknown'


def classify_terminal_event(event: dict[str, Any], *, binary: str, env: dict[str, str], cwd: str) -> str:
    sid = event.get('sessionID')
    if not isinstance(sid, str) or not re.fullmatch(r'ses_[A-Za-z0-9_-]+', sid):
        return 'unknown'
    try:
        # Pinned OpenCode may exit before flushing long exports to a pipe.
        # A regular anonymous file avoids the 16 KiB truncation. Prefer the
        # runner's mounted log directory, so host storage remains on data.
        directory = env.get('AGL_LOG_DIR') or env.get('TMPDIR') or cwd
        with tempfile.TemporaryFile(dir=directory) as output:
            result = subprocess.run([binary, 'export', '--pure', sid], env=env, cwd=cwd,
                                    stdin=subprocess.DEVNULL, stdout=output,
                                    stderr=subprocess.PIPE, timeout=5)
            if result.returncode:
                return 'unknown'
            output.seek(0)
            return classify_exported_message(event, json.load(output))
    except (subprocess.TimeoutExpired, OSError, ValueError, TypeError, AttributeError):
        return 'unknown'
