"""Tracking must redact secrets without breaking live gateway authentication."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "tracking_config_under_test",
    Path(__file__).resolve().parents[2] / "agentlightning/verl/tracking_config.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_redaction_preserves_runtime_credentials_and_hyperparameters():
    live = {
        "agentlightning": {"agl_key": "gateway-credential", "critic_only": True},
        "runtime_env": {"env_vars": {"SWANLAB_API_KEY": "logger-credential"}},
        "data": {"max_tokens": 10240},
        "trainer": {"save_freq": 5},
    }
    logged = module.redact_tracking_config(live)
    assert logged["agentlightning"]["agl_key"] == "<redacted>"
    assert logged["runtime_env"]["env_vars"]["SWANLAB_API_KEY"] == "<redacted>"
    assert live["agentlightning"]["agl_key"] == "gateway-credential"
    assert live["runtime_env"]["env_vars"]["SWANLAB_API_KEY"] == "logger-credential"
    assert logged["data"]["max_tokens"] == 10240
    assert logged["agentlightning"]["critic_only"] is True
    logged["trainer"]["save_freq"] = 99
    assert live["trainer"]["save_freq"] == 5


def test_nested_containers_and_case_insensitive_secret_keys():
    logged = module.redact_tracking_config({"items": [{"API_KEY": "one"}, ({"password": "two"},)]})
    assert logged == {"items": [{"API_KEY": "<redacted>"}, ({"password": "<redacted>"},)]}
