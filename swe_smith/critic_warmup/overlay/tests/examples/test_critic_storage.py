import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "critic_storage", Path(__file__).resolve().parents[2] / "examples/swe_smith/critic_storage.py"
)
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


def test_root_storage_is_rejected():
    with pytest.raises(ValueError, match="data filesystem"):
        storage.runtime_environment("/tmp/critic", "run")


def test_long_run_keeps_socket_path_short(monkeypatch):
    root = Path("/data").resolve()
    project = root / "project"
    created = []
    monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: created.append(self))
    env = storage.runtime_environment(str(project), "run" * 70, data_root=str(root))
    assert len(Path(env["RAY_TMPDIR"]).name) == 13
    assert all(Path(value).is_relative_to(root) and Path(value) in created for value in env.values())
    assert {"HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "AGL_COMPLETED_ROLLOUT_ARCHIVE",
            "CUDA_CACHE_PATH", "FLASHINFER_WORKSPACE_BASE"} <= env.keys()


def test_symlink_escape_rejected_before_creation(tmp_path):
    root = tmp_path / "data"
    project = root / "project"
    project.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (project / "runtime_cache").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink privilege unavailable")
    with pytest.raises(ValueError, match="escapes"):
        storage.runtime_environment(str(project), "run", data_root=str(root))
    assert not (project / "swanlog").exists()
