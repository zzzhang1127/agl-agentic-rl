# Copyright (c) Microsoft. All rights reserved.
"""Place critic runtime files on an explicit data filesystem, never /tmp."""
from __future__ import annotations

import hashlib
from pathlib import Path


def runtime_environment(project_dir: str, run_name: str, *, data_root: str = "/data") -> dict[str, str]:
    """Return storage overrides; reject root storage and symlink escapes.

    Apply these before importing Ray, Torch, vLLM or SwanLab. This function
    creates only directories under the selected data root and returns no secrets.
    """
    root = Path(data_root).resolve()
    project = Path(project_dir).resolve()
    if root == Path(root.anchor) or project == root or not project.is_relative_to(root):
        raise ValueError("project_dir must be inside the data filesystem")
    if not run_name or "/" in run_name or "\\" in run_name or run_name in {".", ".."}:
        raise ValueError("invalid run identity")
    identity = hashlib.sha256((str(project) + "/" + run_name).encode()).hexdigest()[:12]
    paths = {
        "RAY_TMPDIR": root / ("r" + identity),
        "TMPDIR": root / ("t" + identity),
        "SWANLAB_LOG_DIR": project / "swanlog" / run_name,
        "AGL_COMPLETED_ROLLOUT_ARCHIVE": project / "rollout_archive" / run_name,
        "AGL_TRAJ_DIR": project / "trajectory_debug" / run_name,
    }
    for key, name in {
        "XDG_CACHE_HOME": "xdg", "HF_HOME": "huggingface", "TORCH_HOME": "torch",
        "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TORCH_EXTENSIONS_DIR": "extensions", "VLLM_CACHE_ROOT": "vllm",
    }.items():
        paths[key] = project / "runtime_cache" / name
    # Validate every final target before creating anything, including preexisting symlinks.
    for path in paths.values():
        if not path.resolve().is_relative_to(root):
            raise ValueError("runtime directory escapes the data filesystem")
    # Ray appends a timestamp/session suffix and sockets/plasma_store (about 70 bytes).
    if len(str(paths["RAY_TMPDIR"]).encode()) + 70 >= 108:
        raise ValueError("data_root is too long for Ray Unix-domain socket paths")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return {key: str(path) for key, path in paths.items()}
