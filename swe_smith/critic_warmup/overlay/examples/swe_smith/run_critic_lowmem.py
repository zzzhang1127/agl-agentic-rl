# Copyright (c) Microsoft. All rights reserved.
"""Bounded two-GPU critic-only launcher; explicit data storage and runtime inputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from critic_storage import runtime_environment

# Validated MiniCPM5 low-memory settings; not the legacy four-GPU smoke defaults.
OVERRIDES = ['trainer.logger=[console,swanlab]',
 'trainer.project_name=minicpm5_sao_critic',
 'trainer.del_local_ckpt_after_load=False',
 'trainer.n_gpus_per_node=2',
 'trainer.total_training_steps=200',
 'trainer.total_epochs=999',
 'trainer.save_freq=20',
 'data.train_batch_size=2',
 'data.max_prompt_length=13312',
 'data.max_response_length=13312',
 'data.truncation=error',
 'agentlightning.rollout_timeout_seconds=600',
 'agentlightning.max_ppo_update_times=3',
 'agentlightning.trace_aggregator.trajectory_max_prompt_length=13312',
 'agentlightning.trace_aggregator.trajectory_max_response_length=13312',
 'actor_rollout_ref.rollout.tensor_model_parallel_size=1',
 'actor_rollout_ref.rollout.n_gpus_per_node=2',
 'critic.optim.lr=5e-6',
 'actor_rollout_ref.rollout.gpu_memory_utilization=0.35',
 'actor_rollout_ref.rollout.max_model_len=13312',
 'actor_rollout_ref.rollout.max_num_batched_tokens=2048',
 'actor_rollout_ref.rollout.max_num_seqs=1',
 'actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=1024',
 'actor_rollout_ref.rollout.free_cache_engine=True',
 'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=26624',
 'actor_rollout_ref.actor.ppo_mini_batch_size=2',
 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1',
 'actor_rollout_ref.actor.ppo_max_token_len_per_gpu=26624',
 'actor_rollout_ref.actor.fsdp_config.fsdp_size=2',
 'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=26624',
 'actor_rollout_ref.ref.fsdp_config.fsdp_size=2',
 'critic.ppo_mini_batch_size=2',
 'critic.ppo_micro_batch_size_per_gpu=1',
 'critic.forward_micro_batch_size_per_gpu=1',
 'critic.ppo_max_token_len_per_gpu=26624',
 'critic.forward_max_token_len_per_gpu=26624',
 'critic.model.fsdp_config.fsdp_size=2',
 'actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=minicpm5',
 'ray_kwargs.ray_init.address=local',
 'ray_kwargs.ray_init.num_cpus=16',
 'ray_kwargs.ray_init.num_gpus=2',
 'ray_kwargs.ray_init.include_dashboard=false',
 'ray_kwargs.ray_init.object_store_memory=2147483648',
 'ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE="0"',
 'ray_kwargs.ray_init.runtime_env.env_vars.CUMEM_ENABLE="0"',
 'ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"',
 'ray_kwargs.ray_init.runtime_env.env_vars.IB_DISABLE="1"',
 'ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NVLS_ENABLE="0"',
 'ray_kwargs.ray_init.runtime_env.env_vars.NCCL_MNNVL_ENABLE="0"',
 'ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG="INFO"',
 'ray_kwargs.ray_init.runtime_env.env_vars.TORCH_NCCL_ASYNC_ERROR_HANDLING="1"',
 'ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="1"']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-dir", "run-name", "model", "train-data", "val-data", "job-template", "parser-plugin", "chat-template"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--code-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--gpus", default="2,7")
    parser.add_argument("--agl-url", default="http://127.0.0.1:18131")
    parser.add_argument("--agl-key-file", required=True)
    parser.add_argument("--resume", help="Explicit complete global_step_N directory; omit for initialization")
    args = parser.parse_args()
    gpu_ids = args.gpus.split(",")
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(x.isdigit() for x in gpu_ids) or "5" in gpu_ids:
        parser.error("select exactly two distinct GPUs, excluding GPU5")
    environment = runtime_environment(args.project_dir, args.run_name)
    os.environ.update(environment)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["SWANLAB_MODE"] = "cloud"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["AGL_GAE_KEEP_TERMINAL_SEGMENT_ON_PREFIX_MISMATCH"] = "1"
    code = Path(args.code_dir).resolve()
    os.environ["PYTHONPATH"] = str(code) + ":" + str(code / "examples/swe_smith") + ":" + os.environ.get("PYTHONPATH", "")
    sys.path[:0] = [str(code), str(code / "examples/swe_smith")]
    for name in ("model", "train_data", "val_data", "job_template", "parser_plugin", "chat_template", "agl_key_file"):
        if not Path(getattr(args, name)).exists():
            parser.error("missing runtime input: " + name)
    if shutil.disk_usage("/data").free < 64 * 1024**3 or shutil.disk_usage("/").free < 12 * 1024**3:
        parser.error("insufficient disk headroom; no root-filesystem fallback")
    if not os.environ.get("SWANLAB_API_KEY"):
        parser.error("SWANLAB_API_KEY must be supplied through the environment")
    checkpoint = Path(args.project_dir).resolve() / "checkpoints" / args.run_name
    if not checkpoint.resolve().is_relative_to(Path("/data").resolve()):
        parser.error("checkpoint directory escapes /data")
    if checkpoint.exists() and not args.resume:
        parser.error("existing checkpoint output requires explicit resume")
    overrides = list(OVERRIDES) + [
        "trainer.default_local_dir=" + str(checkpoint),
        "trainer.resume_mode=" + ("resume_path" if args.resume else "disable"),
        "agentlightning.k8s.job_template_path=" + str(Path(args.job_template).resolve()),
        "actor_rollout_ref.rollout.engine_kwargs.vllm.tool_parser_plugin=" + str(Path(args.parser_plugin).resolve()),
        "actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template=" + str(Path(args.chat_template).resolve()),
    ]
    if args.resume:
        from agentlightning.verl.critic_checkpoint import validate_critic_checkpoint
        resume = Path(args.resume).resolve()
        if not resume.is_relative_to(Path("/data").resolve()):
            parser.error("resume checkpoint must be on /data")
        validate_critic_checkpoint(str(resume), world_size=2, model_path=args.model)
        if resume.parent != checkpoint:
            parser.error("resume and output must share the same checkpoint root to retain one copy")
        overrides.append("trainer.resume_from_path=" + str(resume))
    environment["AGL_GAE_KEEP_TERMINAL_SEGMENT_ON_PREFIX_MISMATCH"] = "1"
    environment.update({key: os.environ[key] for key in ("PYTHONPATH", "SWANLAB_MODE", "PYTHONDONTWRITEBYTECODE")})
    # Forward storage settings explicitly into Ray; credentials remain in the runtime environment.
    overrides += ["ray_kwargs.ray_init.runtime_env.env_vars." + key + "=" + json.dumps(value)
                  for key, value in environment.items()]
    from train_opencode_agent import train
    train(train_dataset_path=args.train_data, val_dataset_path=args.val_data, model=args.model,
          agl_base_url=args.agl_url, agl_key=Path(args.agl_key_file).read_text().strip(),
          run_name=args.run_name, critic_warmup=True, config_overrides=overrides)


if __name__ == "__main__":
    main()
