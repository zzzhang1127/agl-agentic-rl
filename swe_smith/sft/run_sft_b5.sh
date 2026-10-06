#!/bin/bash
# SFT round "b5" (2026-10-06): continue from the previous round's weights (SFT ep3) on the batch-5 teacher
# trajectories collected with the OFFICIAL smith harness (teacher_smith_b5.sh -> assemble_smith_teacher.py ->
# prep_sft_data.py => sft/data_b5: 677 train rows + 30 held-out problems, all main-segment, teacher-RESOLVED only).
# User directives: 09-21 「每轮新采 1000 题(互斥)在上轮权重上续训 1 epoch」; 10-06 「蒸馏就用 deepseekv4flash …
# 做错的题就没必要参与训练了」. Gains from this run are DISTILLATION gains, never to be reported as RL gains.
# Same 4-card FSDP + Ulysses SP4 layout as run_sft_v3.sh (one 49k-token row split over 4 cards, dynamic packing).
# Global batch 32 rows -> 21 steps/epoch; 1 epoch. LR defaults to 5e-6 (half of the from-base round: this is a
# continuation, cosine to 0.1x, 3 warmup steps). Needs 4 free cards: s5 (1/2/3/5) must be paused first.
# usage: GPUS=1,2,3,5 bash run_sft_b5.sh    (env: LR, EPOCHS, BATCH, CKPT, INIT, MASTER_PORT)
set -u
D=/workspace/agl-checkpoints/swe_smith_smoke/sft
export CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,5}
export CUDA_MPS_PIPE_DIRECTORY=/workspace/mps_bypass_empty   # §29 global MPS bypass
export NCCL_NVLS_ENABLE=0                                        # §34 NVLS broken since Xid31
export NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1 VERL_LOGGING_LEVEL=WARN
export SWANLAB_LOG_DIR=/workspace/agl-checkpoints/swanlog SWANLAB_MODE=cloud
CKPT=${CKPT:-/workspace/agl-checkpoints/swe_smith_sft_b5}
INIT=${INIT:-/workspace/models/MiniCPM5-2B-sft-v3-ep3}
for f in $D/data_b5/train.parquet $D/data_b5/val.parquet $INIT/config.json; do [ -f $f ] || { echo "missing $f"; exit 1; }; done
# refuse to start on busy cards (never fight the trainer / neighbours for memory)
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$used" -lt 8000 ] || { echo "GPU $g has ${used}MiB in use; refusing to start"; exit 1; }
done
cd /workspace/projects/agent-lightning
exec .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=${MASTER_PORT:-29519} \
  -m verl.trainer.sft_trainer \
  data.train_files=$D/data_b5/train.parquet data.val_files=$D/data_b5/val.parquet \
  data.custom_cls.path=$D/swe_sft_dataset.py data.custom_cls.name=SWESmithSFTDataset \
  data.train_batch_size=${BATCH:-32} data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=True data.max_token_len_per_gpu=12288 data.max_length=49152 \
  data.pad_mode=no_padding data.truncation=error data.num_workers=2 \
  model.path=$INIT model.use_remove_padding=True \
  model.enable_gradient_checkpointing=True \
  engine.ulysses_sequence_parallel_size=4 engine.strategy=fsdp \
  optim.lr=${LR:-5e-6} optim.lr_scheduler_type=cosine optim.lr_warmup_steps=3 optim.min_lr_ratio=0.1 \
  optim.weight_decay=0.01 optim.clip_grad=1.0 optim.betas=[0.9,0.95] \
  trainer.total_epochs=${EPOCHS:-1} trainer.save_freq=after_each_epoch trainer.test_freq=after_each_epoch \
  trainer.max_ckpt_to_keep=1 trainer.resume_mode=disable \
  trainer.project_name=swe_smith_sft trainer.experiment_name=sft_b5_from_ep3 \
  trainer.default_local_dir=$CKPT trainer.logger=[console,swanlab] trainer.n_gpus_per_node=4 trainer.seed=1
