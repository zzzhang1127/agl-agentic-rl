#!/bin/bash
# SFT of the base MiniCPM5-2B on teacher (deepseek-v4-flash) trajectories, dataset v3 clean rows
# (resolved && no flags; main/compact/post segments; synthetic summary messages masked).
# 2026-09-20 user: "对原始minicpm基座SFT，batch_size可以开大一点，让GPU利用率拉满".
# 4 cards (1,2,3,5) = FSDP world 4, Ulysses SP 4 (dp 1): one 44k-token row is split over 4 cards, so a
# micro-batch holds up to 49152 tokens (max_token_len_per_gpu 12288 x sp 4) and dynamic bsz packs short rows.
# Global batch 32 rows (~660k tokens/step), 858 rows -> 26 steps/epoch, 3 epochs = 78 steps.
set -u
D=/workspace/agl-checkpoints/swe_smith_smoke/sft
export CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,5}
export CUDA_MPS_PIPE_DIRECTORY=/workspace/mps_bypass_empty   # §29 global MPS bypass
export NCCL_NVLS_ENABLE=0                                        # §34 NVLS broken since Xid31
export NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1 VERL_LOGGING_LEVEL=WARN
export SWANLAB_LOG_DIR=/workspace/agl-checkpoints/swanlog SWANLAB_MODE=cloud   # same account/logdir as the GRPO runs
CKPT=${CKPT:-/workspace/agl-checkpoints/swe_smith_sft_v3}
cd /workspace/projects/agent-lightning
exec .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=${MASTER_PORT:-29517} \
  -m verl.trainer.sft_trainer \
  data.train_files=$D/data/train.parquet data.val_files=$D/data/val.parquet \
  data.custom_cls.path=$D/swe_sft_dataset.py data.custom_cls.name=SWESmithSFTDataset \
  data.train_batch_size=${BATCH:-32} data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=True data.max_token_len_per_gpu=12288 data.max_length=49152 \
  data.pad_mode=no_padding data.truncation=error data.num_workers=2 \
  model.path=/workspace/models/MiniCPM5-2B model.use_remove_padding=True \
  model.enable_gradient_checkpointing=True \
  engine.ulysses_sequence_parallel_size=4 engine.strategy=fsdp \
  optim.lr=${LR:-1e-5} optim.lr_scheduler_type=cosine optim.lr_warmup_steps=3 optim.min_lr_ratio=0.1 \
  optim.weight_decay=0.01 optim.clip_grad=1.0 optim.betas=[0.9,0.95] \
  trainer.total_epochs=${EPOCHS:-3} trainer.save_freq=after_each_epoch trainer.test_freq=after_each_epoch \
  trainer.max_ckpt_to_keep=3 trainer.resume_mode=disable \
  'checkpoint.save_contents=[hf_model]' \
  trainer.project_name=swe_smith_sft trainer.experiment_name=sft_v3_base \
  trainer.default_local_dir=$CKPT trainer.logger=[console,swanlab] trainer.n_gpus_per_node=4 trainer.seed=1
