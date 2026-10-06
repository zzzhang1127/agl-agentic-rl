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
        str(Path("/workspace/agl-checkpoints/swe_smith_opencode")),
    )
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "enable_rollout_level_advantage": True,
        },
        "data": {
            "train_batch_size": 4,
            "max_prompt_length": 18432,
            "max_response_length": 6144,
            "truncation": "left",
        },
        "actor_rollout_ref": {
            "hybrid_engine": True,
            # NCCL process-group watchdog timeout. MUST live here: the FSDP worker
            # reads it from actor_rollout_ref.config (fsdp_workers.py init_process_group),
            # NOT from trainer.nccl_timeout. Long step-2 updates (5 resolved = long
            # multi-turn seqs + triple CPU offload) exceed the 600s default -> watchdog
            # abort -> ActorDiedError. 3600s lets slow-but-progressing updates finish.
            "nccl_timeout": 3600,
            "rollout": {
                "mode": "async",
                "tensor_model_parallel_size": 2,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 24576,
                "log_prob_use_dynamic_bsz": False,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                # TP=2 → 2 vLLM replicas (2 GPUs each); util/seqs tuned from smoke.
                "gpu_memory_utilization": 0.85,
                "max_model_len": 24576,
                "max_num_seqs": 8,
                "max_num_batched_tokens": 8192,
                "free_cache_engine": True,
                "enforce_eager": True,
                "engine_kwargs": {
                    "vllm": {
                        # Absolute KV budget: 8 seqs x 24576 = 196,608 tok = 12288 blocks
                        # (~14.5GB/rank). util 0.85 is only the profiling allowance ceiling.
                        "num_gpu_blocks_override": 12288,
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                        "chat_template": str(EXAMPLE_DIR / "swe_smith_chat_template.jinja"),
                    }
                },
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "val_kwargs": {"temperature": 0.7, "do_sample": True},
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "logprobs_mode": None,
                "checkpoint_engine": {"update_weights_bucket_megabytes": 4096},
            },
            "actor": {
                "ppo_mini_batch_size": 4,
                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_max_token_len_per_gpu": 24576,
                "use_dynamic_bsz": False,
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
                "policy_loss": {"loss_mode": "per_rollout_mean"},
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 24576,
                "log_prob_use_dynamic_bsz": False,
                "fsdp_config": {"param_offload": True, "fsdp_size": 4},
            },
            "model": {
                "path": DEFAULT_MODEL,
                "use_remove_padding": True,
                "use_fused_kernels": True,
                "fused_kernel_options": {"impl_backend": "torch"},
                "enable_gradient_checkpointing": True,
                "enable_activation_offload": True,
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
            "rollout_timeout_seconds": 2400,
            "reward_fillna_value": 0.0,
            "max_ppo_update_times": 2,
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 18432,
                "trajectory_max_response_length": 6144,
            },
            "async_rollout": {
                "enabled": False,
                "async_train_batch_size": 50,
            },
            # DAPO dynamic sampling: drop GRPO groups whose rollouts all got the
            # same reward (zero advantage) and roll out extra batches until
            # min_valid_groups groups survive, at most max_gen_batches rounds.
            # min_valid_groups=None means data.train_batch_size.
            "dynamic_sampling": {
                "enabled": False,
                "max_gen_batches": 3,
                "min_valid_groups": None,
            },
            # Permute rows before mini-batch splitting so importance-ratio drift over
            # the update sequence is spread across rollouts instead of always landing
            # on the last ones (legacy dp_actor splits in order, no shuffle).
            "shuffle_train_rows": True,
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
    config = OmegaConf.merge(base_cfg, override_conf, cli_override_conf)
    OmegaConf.set_struct(config, False)
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
) -> None:
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
    )
    log("\n=== VERL config ===")
    pprint(OmegaConf.to_container(config, resolve=True))
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
    )


if __name__ == "__main__":
    main()
