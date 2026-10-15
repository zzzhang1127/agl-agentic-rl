# Copyright (c) Microsoft. All rights reserved.

"""Online bridge to the same structured grader and independent evidence gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(smith, meta, timeout, *, f2p_only):
    here = Path(__file__).resolve().parent
    grader_path = here / "eval_inside.py"
    gate_path = here / "validate_results.py"
    if not grader_path.exists():
        grader_path = here.parent / "opencode_smoke/eval_inside.py"
    if not gate_path.exists():
        gate_path = here.parents[2] / "fix_qwen35_20261011/sft_eval_patch/validate_results.py"
    # Missing dependencies must fail, never silently fall back to legacy zero rewards.
    grader = _load("online_structured_grader", grader_path)
    gate = _load("online_structured_gate", gate_path)
    if not f2p_only:
        raise ValueError("structured online grader only supports the versioned F2P-same-file P2P scope")
    grader.TESTBED = smith.TESTBED
    grader.git_base = smith._git_base
    result = grader.evaluate(meta, timeout)
    try:
        if result.get("grader_protocol") != "pytest-structured-v1":
            raise gate.InvalidResults("missing structured grader protocol")
        gate.validate_eval({"eval": result, "smith_rc": 0, "timed_out": False,
                            "n_f2p": len(meta.get("FAIL_TO_PASS") or [])})
    except gate.InvalidResults as exc:
        result["reward_eligible"] = False
        if result.get("evaluation_state") != "timeout":
            result["evaluation_state"] = "unavailable"
        result["approval_error"] = str(exc)
    return result
