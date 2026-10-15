# Copyright (c) Microsoft. All rights reserved.

"""OpenCode SWE-smith GRPO trainer (k3s rollouts, 4-GPU FSDP + offload)."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from pprint import pprint
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_smith_agent import DEFAULT_MODEL, EXAMPLE_DIR, load_split_file, log

TRAIN_BACKEND = "fsdp"


def verl_default_config() -> dict[str, Any]:
    ckpt_dir = os.environ.get(
        "AGL_CKPT_DIR",
        str(Path("/data/agl-example/agl-checkpoints/swe_smith_opencode")),
    )
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 4,
            "max_prompt_length": 32768,
            "max_response_length": 6144,
            "truncation": "left",
        },
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "rollout": {
                "mode": "async",
                "tensor_model_parallel_size": 2,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 40960,
                "log_prob_use_dynamic_bsz": True,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                # TP=2 → 2 vLLM replicas (2 GPUs each); util/seqs tuned from smoke.
                "gpu_memory_utilization": 0.8,
                "max_model_len": 40960,
                "max_num_seqs": 8,
                "max_num_batched_tokens": 8192,
                "free_cache_engine": True,
                "enforce_eager": True,
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                        "chat_template": str(EXAMPLE_DIR / "swe_smith_chat_template.jinja"),
                    }
                },
                "temperature": 1,
                "val_kwargs": {"temperature": 0.7, "do_sample": True},
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "logprobs_mode": None,
                "checkpoint_engine": {"update_weights_bucket_megabytes": 4096},
            },
            "actor": {
                "ppo_mini_batch_size": 4,
                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_max_token_len_per_gpu": 40960,
                "use_dynamic_bsz": True,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                    "reshard_after_forward": True,
                    "fsdp_size": 4,
                    "entropy_from_logits_with_chunking": True,
                },
                "loss_agg_mode": "token-mean",
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 40960,
                "log_prob_use_dynamic_bsz": True,
                "fsdp_config": {"param_offload": True, "fsdp_size": 4},
            },
            "model": {
                "path": DEFAULT_MODEL,
                "use_remove_padding": True,
                "use_fused_kernels": True,
                "fused_kernel_options": {"impl_backend": "torch"},
                "enable_gradient_checkpointing": True,
                "override_config": {"attn_implementation": "flash_attention_2"},
            },
        },
        "trainer": {
            "n_gpus_per_node": 4,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console", "swanlab"],
            "project_name": "agentlightning",
            "experiment_name": "swe_smith_opencode",
            "nnodes": 1,
            "nccl_timeout": 3600,
            # 0 disables periodic and end-of-run validation; run val separately when needed.
            "test_freq": 0,
            "save_freq": 2,
            "total_epochs": 4,
            "total_training_steps": 1500,
            "resume_mode": "disable",
            "resume_from_path": None,
            "default_local_dir": ckpt_dir,
            "max_actor_ckpt_to_keep": 1,
            "del_local_ckpt_after_load": True,
        },
        "agentlightning": {
            "agl_base_url": "http://localhost:8080",
            "agl_key": "",
            "rollout_timeout_seconds": 1500,
            "reward_fillna_value": 0.0,
            "max_ppo_update_times": 2,
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 32768,
                "trajectory_max_response_length": 6144,
            },
            "async_rollout": {
                "enabled": False,
                "async_train_batch_size": 50,
            },
            "k8s": {
                "job_template_path": str(EXAMPLE_DIR / "job-template-opencode.yaml"),
            },
        },
    }


def build_config(
    *,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
    critic_warmup: bool = False,
) -> DictConfig:
    import importlib.resources

    verl_pkg = importlib.resources.files("agentlightning.verl")
    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
        base_cfg = compose(config_name="config")

    overrides = verl_default_config()
    if model:
        overrides["actor_rollout_ref"]["model"]["path"] = model
    if agl_base_url:
        overrides["agentlightning"]["agl_base_url"] = agl_base_url
    if agl_key is not None:
        overrides["agentlightning"]["agl_key"] = agl_key

    rollout_mode = overrides["actor_rollout_ref"]["rollout"]["mode"]
    model_path = overrides["actor_rollout_ref"]["model"]["path"]
    overrides["trainer"]["experiment_name"] = f"swe_smith_opencode_{rollout_mode}_{model_path.split('/')[-1]}_{TRAIN_BACKEND}"
    if run_name:
        overrides["trainer"]["experiment_name"] = f"{overrides['trainer']['experiment_name']}_{run_name}"

    override_conf = OmegaConf.create(overrides)
    cli_override_conf = OmegaConf.from_dotlist(list(config_overrides))
    OmegaConf.set_struct(base_cfg, False)
    layers = [base_cfg, override_conf]
    if critic_warmup:
        if not model:
            raise ValueError("critic warmup requires an explicit starting model")
        layers.append(OmegaConf.load(Path(__file__).with_name("critic_warmup.yaml")))
        layers.append(OmegaConf.create({"critic": {"model": {"path": model, "tokenizer_path": model}}}))
    config = OmegaConf.merge(*layers, cli_override_conf)
    OmegaConf.set_struct(config, False)
    if critic_warmup:
        if (config.agentlightning.critic_only is not True
                or config.agentlightning.freeze_critic_attention is not True
                or config.agentlightning.require_explicit_reward is not True
                or config.algorithm.adv_estimator != "gae"
                or config.algorithm.enable_rollout_level_advantage is not False
                or config.actor_rollout_ref.rollout.n != 1):
            raise ValueError("critic warmup overrides must preserve the fixed-actor GAE interlocks")
        if config.trainer.resume_mode != "disable":
            if config.trainer.resume_mode != "resume_path" or config.trainer.del_local_ckpt_after_load:
                raise ValueError("critic warmup resume requires explicit resume_path and source preservation")
            from agentlightning.verl.critic_checkpoint import validate_critic_checkpoint
            validate_critic_checkpoint(
                config.trainer.resume_from_path,
                world_size=config.trainer.n_gpus_per_node * config.trainer.nnodes,
                model_path=model,
            )
        if not any(item.startswith("trainer.default_local_dir=") for item in config_overrides):
            raise ValueError("critic warmup requires an explicit independent trainer.default_local_dir")
    return config


def train(
    *,
    train_dataset_path: str,
    val_dataset_path: str,
    max_val_instances: int | None = None,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
    critic_warmup: bool = False,
) -> None:
    if critic_warmup:
        from critic_data_guard import validate_critic_smoke_data

        if max_val_instances is not None:
            raise ValueError("critic smoke must use its entire prepared validation split")
        validate_critic_smoke_data(train_dataset_path, val_dataset_path)

    from agentlightning.verl.entrypoint import run_ppo

    if not agl_key:
        raise RuntimeError("AGL_KEY is required")

    train_dataset = load_split_file(train_dataset_path)
    if max_val_instances == 0:
        val_dataset: list[dict[str, Any]] = []
    else:
        val_dataset = load_split_file(val_dataset_path, max_instances=max_val_instances)
    instances = train_dataset + val_dataset
    distinct_repos = sorted({row["repo"] for row in instances})

    log("=== Preflight ===")
    log(f"  Agent Lightning: {agl_base_url or 'http://localhost:8080'}")
    log(f"  model:        {model or DEFAULT_MODEL}")
    log(f"  train file:   {train_dataset_path}")
    log(f"  val file:     {val_dataset_path}")
    log(f"  instances:    {len(instances)}  (train {len(train_dataset)} / val {len(val_dataset)})")
    log(f"  distinct repos (images): {len(distinct_repos)}")
    log("  runner:       k3s Job + OpenCode")

    config = build_config(
        model=model,
        agl_base_url=agl_base_url,
        agl_key=agl_key,
        run_name=run_name,
        config_overrides=config_overrides,
        critic_warmup=critic_warmup,
    )
    log("\n=== VERL config ===")
    display_config = OmegaConf.to_container(config, resolve=True)
    display_config["agentlightning"]["agl_key"] = "<redacted>"
    pprint(display_config)
    log("\n=== Start VERL training ===")
    run_ppo(config=config, train_dataset=train_dataset, val_dataset=val_dataset)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train SWE-smith OpenCode with VERL/GRPO via k3s")
    parser.add_argument("--train-dataset-path", default=str(EXAMPLE_DIR / "train_dataset_mixed.jsonl"))
    parser.add_argument("--val-dataset-path", default=str(EXAMPLE_DIR / "val_dataset_filtered.jsonl"))
    parser.add_argument("--max-val-instances", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--agl-base-url", default="http://localhost:8080")
    parser.add_argument("--agl-key", default="")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--critic-warmup", action="store_true",
                        help="Use the bounded fixed-actor attention-frozen PPO critic smoke profile")
    args, config_overrides = parser.parse_known_args()
    return args, config_overrides


def main() -> None:
    args, config_overrides = parse_args()
    train(
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        max_val_instances=args.max_val_instances,
        model=args.model,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
        config_overrides=config_overrides,
        critic_warmup=args.critic_warmup,
    )


if __name__ == "__main__":
    main()
