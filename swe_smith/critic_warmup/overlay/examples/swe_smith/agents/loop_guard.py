#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Conservative no-progress loop detection for the SWE-smith agent loop.

A "stuck loop" is NOT command repetition alone. The agent legitimately re-runs
``pytest`` every turn while it iterates on a fix; that is progress only if the
workspace (or the observation) is actually changing. So a loop is only declared
when the *same* normalized action produces an *equivalent* observation while the
workspace fingerprint is *unchanged*, and that whole triple repeats periodically
``min_repeats`` times (period ``1..max_period``).

Supported semantics — conservative by construction:

* Default disabled — callers opt in explicitly (``SMITH_LOOP_GUARD=1``). The
  guard is NOT auto-enabled: it stays off until replay calibration confirms the
  false-positive rate is acceptable on the actual model+repo distribution.
* ``min_repeats`` full periods are required before a verdict, so a single
  repeated poll is never flagged.
* ``observation`` must be byte-identical and ``fingerprint`` unchanged, so a
  poll whose output ticks (a counter, a timestamp, a build artifact growing) or
  a workspace that is being edited is never flagged.
* A legitimate wait/poll that keeps returning the *same* output with the *same*
  workspace is indistinguishable, by this signal alone, from a stuck loop. We
  deliberately do not claim such a triple "proves" the agent is stuck — it is
  only the supported conservative trigger, tuned by ``min_repeats``/``max_period``
  and kept opt-in. There is no guarantee of zero false positives; the design
  only errs toward fewer of them.
* Any fingerprint error, a bounded-scan limit being exceeded, or an unreadable
  file yields ``None``, which ``observe`` treats as "unknown" and clears the
  cycle buffer — the verdict is disabled, never a false positive.

Control-flow reality (see ``run_agent_loop`` in ``smith_agent``): the guard
observes at executed-tool boundaries, i.e. AFTER a completed model generation
and the corresponding shell command has already returned. When a loop fires there
is therefore usually NO in-flight request left to cancel — firing just ends the
episode cleanly. A wall-clock timeout or disconnect that happens *during* an
awaited generation (while ``_query`` is blocked) cannot be aborted by this module;
cancelling a backend generation still requires a cancellation proxy, which is
NOT implemented here. Nothing in this module frees backend resources.

This module is stdlib-only and importable both as a package
(``from examples.swe_smith.agents import loop_guard``) and top-level (the SWE
image copies it next to ``smith_agent.py`` for the smoke path).
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import time
from collections import deque
from typing import Any

# Directory names that never contribute to a workspace fingerprint. ``.git`` is
# relocated out of the worktree by ``relocate_git()`` before the loop runs, but
# is kept here so the guard also works against an un-relocated tree. Caches are
# pure noise and change under the agent's feet even when it is stuck.
_IGNORE_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".nox",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "venv",
    }
)

_READ_CHUNK = 65536

# Bounded-scan limits so a dead loop guard never scans a huge repo forever.
# Every limit is configurable via ``workspace_fingerprint``; exceeding any of
# them makes the fingerprint ``None`` (no verdict), which is the sensible
# fallback — the guard disables itself rather than hang or guess.
_DEFAULT_GIT_TIMEOUT = 20.0  # seconds, per git subprocess call
_DEFAULT_MAX_FILES = 20000
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024  # 256 MiB


def normalize_action(action: str) -> str:
    """Normalize an action for cycle comparison — exact strip only.

    Leading/trailing whitespace is trimmed, but interior whitespace is preserved
    verbatim. Collapsing interior whitespace would change shell semantics inside
    quoted strings (``echo "a  b"`` is not the same command as ``echo "a b"``),
    so a stuck-cycle detector must not merge them: a false negative is
    acceptable, a false positive is not.
    """
    return (action or "").strip()


def _hash_file_into(hasher: Any, path: str, remaining_bytes: int) -> tuple[bool, int]:
    """Stream ``path`` into ``hasher``. Returns (ok, bytes_read).

    On Linux, opens with ``O_NONBLOCK`` and confirms ``S_ISREG`` via ``fstat``
    before reading to avoid blocking on FIFOs or special files. Each chunk read
    is counted against ``remaining_bytes``.
    """
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        fd = os.open(path, flags)
    except OSError:
        return False, 0
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return False, 0
        if st.st_size > remaining_bytes:
            return False, 0
        total = 0
        fh = os.fdopen(fd, "rb", closefd=False)
        try:
            for chunk in iter(lambda: fh.read(_READ_CHUNK), b""):
                total += len(chunk)
                if total > remaining_bytes:
                    return False, total
                hasher.update(chunk)
            return True, total
        finally:
            fh.close()
    except OSError:
        return False, 0
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _walk_fingerprint(root: str, ignore_dirs: frozenset[str], max_files: int, max_bytes: int, deadline: float) -> str | None:
    """Hash every file's relative path + content under ``root`` (untracked included).

    Returns ``None`` if ``root`` is missing, the shared ``deadline`` expires,
    an unsafe symlink or special file is encountered, any file cannot be read,
    or a bounded-scan limit is exceeded.
    """
    if not os.path.isdir(root):
        return None
    if time.monotonic() > deadline:
        return None
    hasher = hashlib.blake2b(digest_size=16)
    nfiles = 0
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in ignore_dirs)
        for name in sorted(filenames):
            if time.monotonic() > deadline:
                return None
            if nfiles >= max_files:
                return None
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                return None
            try:
                st = os.lstat(full)
            except OSError:
                return None
            if not stat.S_ISREG(st.st_mode):
                return None
            size = st.st_size
            if total_bytes + size > max_bytes:
                return None
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            hasher.update(rel.encode("utf-8", "replace"))
            ok, read_bytes = _hash_file_into(hasher, full, max_bytes - total_bytes)
            if not ok:
                return None
            nfiles += 1
            total_bytes += read_bytes
    return hasher.hexdigest()


def _git_fingerprint(
    root: str,
    git_base: list[str],
    ignore_dirs: frozenset[str],
    timeout_s: float,
    max_files: int,
    max_bytes: int,
) -> str | None:
    """Fast path: hash git status + diff HEAD + HEAD tree + untracked bytes.

    ``status --porcelain -uall`` captures tracked modifications AND untracked
    names; ``diff HEAD`` captures tracked content; ``rev-parse HEAD^{tree}``
    captures the committed tree identity so a commit cycle (edit -> commit ->
    clean tree) still changes the fingerprint; the untracked-file loop captures
    untracked content (which a name-only hash would miss). Every git call is
    bounded by ``timeout_s`` and the untracked scan by ``max_files``/``max_bytes``.
    Returns ``None`` on any git/read error or exceeded limit.
    """
    del ignore_dirs  # git path does not need the dir-name filter
    try:
        status = subprocess.run(
            [*git_base, "status", "--porcelain=v1", "-z", "-uall"],
            cwd=root,
            capture_output=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if status.returncode != 0:
        return None
    try:
        diff = subprocess.run(
            [*git_base, "-c", "core.fileMode=false", "diff", "HEAD", "--no-ext-diff"],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if diff.returncode != 0:
        return None
    try:
        head_tree = subprocess.run(
            [*git_base, "rev-parse", "HEAD^{tree}"],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if head_tree.returncode != 0:
        return None

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(b"status\n")
    hasher.update(status.stdout)
    hasher.update(b"diff\n")
    hasher.update(diff.stdout.encode("utf-8", "replace"))
    hasher.update(b"head_tree\n")
    hasher.update(head_tree.stdout.encode("utf-8", "replace"))

    nfiles = 0
    total_bytes = 0
    entries = status.stdout.split(b"\0")
    i = 0
    while i < len(entries):
        entry = entries[i]
        if len(entry) < 3 or entry[2:3] != b" ":
            i += 1
            continue
        code = entry[0:2]
        path_bytes = entry[3:]
        if b"R" in code or b"C" in code:
            i += 2
        else:
            i += 1
        if code != b"??":
            continue
        path = os.fsdecode(path_bytes)
        full = os.path.join(root, path)
        if not os.path.isfile(full):
            # Untracked directory or symlink: its name is already in ``status``.
            continue
        if nfiles >= max_files:
            return None
        try:
            size = os.path.getsize(full)
        except OSError:
            return None
        if total_bytes + size > max_bytes:
            return None
        ok, read_bytes = _hash_file_into(hasher, full, max_bytes - total_bytes)
        if not ok:
            return None
        nfiles += 1
        total_bytes += read_bytes
    return hasher.hexdigest()


def workspace_fingerprint(
    root: str,
    *,
    git_base: list[str] | None = None,
    ignore_dirs: frozenset[str] = _IGNORE_DIR_NAMES,
    timeout_s: float = _DEFAULT_GIT_TIMEOUT,
    max_files: int = _DEFAULT_MAX_FILES,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> str | None:
    """Content fingerprint of the worktree (tracked AND untracked), or ``None``.

    Tries the git fast path first (when ``git_base`` is supplied, e.g. the
    harness's ``_git_base()``), then falls back to a full directory walk. Any
    error — missing root, failed git, unreadable file, exceeded bounded-scan
    limit — returns ``None`` so the loop verdict is disabled rather than guessed.
    """
    try:
        if git_base is not None:
            fp = _git_fingerprint(root, git_base, ignore_dirs, timeout_s, max_files, max_bytes)
            if fp is not None:
                return fp
        return _walk_fingerprint(root, ignore_dirs, max_files, max_bytes, time.monotonic() + timeout_s)
    except Exception:
        return None


class LoopGuard:
    """Detects periodic stuck cycles of (action, observation, fingerprint).

    ``observe`` records one executed tool step and returns ``True`` only when the
    last ``min_repeats`` periods form an exact periodic cycle (period between 1
    and ``max_period``). Period 1 is "same step three times in a row"; period 2
    is an alternating A/B cycle repeated three times. A ``None`` fingerprint
    clears history and never triggers.
    """

    def __init__(self, *, min_repeats: int = 3, max_period: int = 2) -> None:
        self.min_repeats = max(2, int(min_repeats))
        self.max_period = max(1, int(max_period))
        self._history: deque[tuple[str, str, str]] = deque(maxlen=self.min_repeats * self.max_period)
        self.detected = False
        self.period = 0

    def reset(self) -> None:
        self._history.clear()
        self.detected = False
        self.period = 0

    def _signature(self, action: str, observation: str, fingerprint: str) -> tuple[str, str, str]:
        obs_hash = hashlib.blake2b((observation or "").encode("utf-8", "replace"), digest_size=8).hexdigest()
        return (normalize_action(action), obs_hash, fingerprint)

    def observe(self, action: str, observation: str, fingerprint: str | None) -> bool:
        """Record one step. Returns True when a stuck cycle is detected."""
        if fingerprint is None:
            # Workspace state unknown: we cannot confirm "unchanged", so the
            # verdict is disabled (and stale history discarded) — never guess.
            self._history.clear()
            return False
        self._history.append(self._signature(action, observation, fingerprint))
        return self._detect()

    def _detect(self) -> bool:
        history = list(self._history)
        n = len(history)
        for period in range(1, self.max_period + 1):
            need = self.min_repeats * period
            if n < need:
                continue
            tail = history[-need:]
            block = tail[:period]
            if all(tail[i] == block[i % period] for i in range(need)):
                self.detected = True
                self.period = period
                return True
        return False
