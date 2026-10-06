#!/bin/bash
# v4 (2026-09-17 人类拍板):从 step48-v3 合并权重起训(step48 ckpt 已被 keep=2 清掉,优化器状态重置),
# lr 2e-6 / PPO_MINI 8 / KL 0.001 / 动态采样 / 行打乱 / 刹车(configMap 已刷)。
# 每 20 步由 swe_smith_smoke/v4_cycle.sh 停训做 100 题探针,再 resume_mode=auto 续。
# 09-17 21:34 首次启动挂在 vLLM "No available memory for the cache blocks":vLLM 的可用 KV =
# total×util − 卡上全部已占显存(含邻居;kv_cache_utils.py:527 在 num_gpu_blocks_override 生效之前检查),
# 邻居突发 +39G 就算成负数。实际 KV 大小由 num_gpu_blocks_override=25600(≈8.2G/rank)固定,
# util 只影响这道检查 → 提到 0.95。真正的显存余量由 v4_cycle.sh 启动前等 free≥40G 保证。
# v3 训练脚本(2026-09-15)。相对 v2 的改动见文件末尾对照表。
# 与 v2 的区别只在最后一段 override;前面的环境变量逐字沿用 v2(§29/§33/§34/§35)。
#
# !!! 启动前先读 CHANGES 那一段:PPO_MINI 的取值依赖一次 smoke 实测,
# !!! 现在写的是按"采集率还是 8.1%"算出来的数,修好采集之后它是错的。

export CUDA_MPS_PIPE_DIRECTORY=/workspace/mps_bypass_empty
# 09-17 23:00 人类:4 卡等不到就先用 2 卡(卡 3+5)训,batch 减半,显卡充足再回 4 卡。
# GPU_MODE=4(默认)= 卡 1,2,3,5 / fsdp 4 / 2 个 TP2 vLLM 副本 / train_batch 8 / PPO_MINI 8 / 动态采样 6 组。
# GPU_MODE=2 = 卡 3,5 / fsdp 2 / 1 个 TP2 副本 / train_batch 4 / PPO_MINI 4 / 3 组 / MAX_UPDATES 翻倍(行数上限不变 512)。
# 注意 FSDP ckpt 按 world_size 分片(fsdp_checkpoint_manager.py:139),两种模式的 ckpt 不能互相 resume,
# 所以各用自己的 ckpt 目录,切换模式要经过 model_merger 合并出的 HF 权重(优化器重置)—— v4_cycle.sh 负责。
GPU_MODE=${GPU_MODE:-4}
if [ "$GPU_MODE" = 2 ]; then
  export CUDA_VISIBLE_DEVICES=${GPUS:-3,5}
  NGPU=2; FSDP_SIZE=2; TRAIN_BATCH=4; PPO_MINI_DEFAULT=4; DYN_MIN_DEFAULT=3; MAX_UPDATES_DEFAULT=32
  CKPT_DEFAULT=/workspace/agl-checkpoints/swe_smith_opencode_minicpm50k_v4_2gpu
else
  export CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,5}
  NGPU=4; FSDP_SIZE=4; TRAIN_BATCH=8; PPO_MINI_DEFAULT=8; DYN_MIN_DEFAULT=6; MAX_UPDATES_DEFAULT=16
  CKPT_DEFAULT=/workspace/agl-checkpoints/swe_smith_opencode_minicpm50k_v4
fi
export NCCL_NVLS_ENABLE=0  # 09-12: Xid31 后 NVLS 组播损坏,禁用回退 NVLink 环(§34)
export VLLM_USE_V1=1  # 09-12: verl 要求 V1 AsyncLLMEngine(§33)
export AGL_CKPT_DIR=${AGL_CKPT_DIR:-$CKPT_DEFAULT}
export RAY_TMPDIR=/workspace/ray_tmp
export RAY_local_fs_capacity_threshold=0.99

# 采集率闸门(§41.5 第5条):低于此值直接拒绝训练,不静默地在残缺数据上跑。
# 首跑建议开着;确认稳定在 0.9 以上后可以摘掉。
export AGL_MIN_CAPTURE_RATE=${AGL_MIN_CAPTURE_RATE:-0.9}

# --- v3 可调项,集中在这里方便改 ---
LR=${LR:-2e-6}       # 09-17 人类拍板:5e-6 → 2e-6(step76 塌缩后,配合 KL 0.001 + 动态采样稳住)。
                     # 原注:v2: 1e-6。你给的区间是 5e-6~1e-5,当时取下界 5e-6。
                     # 注意:配合下面 4 次更新,单步位移约是 v2 的 20 倍。
                     # 想更可归因就先用 3e-6。官方没公布可用的全参 SFT lr
                     # (cookbook 那个 2e-4 是 LoRA 的,不能拿来锚),verl 自己的
                     # actor 默认就是 1e-6,也就是 v2 在用的值。
PPO_MINI=${PPO_MINI:-$PPO_MINI_DEFAULT}  # v2: 8。全局 mini-batch = PPO_MINI × rollout.n = 32 行。
                     # !!! verl 硬约束 train_batch_size >= ppo_mini_batch_size
                     # (workers/config/actor.py:216),所以 PPO_MINI 最大只能是 8。
                     # verl 假定"一行=一个(prompt,sample)",多轮下一行只是一"轮",
                     # 于是全局 mini-batch 最多只有 train_batch×n = 32 行,
                     # 而一整批有 ~800 行 —— 想吃完得更新 25 次。见 CHANGES 末尾。
MAX_UPDATES=${MAX_UPDATES:-$MAX_UPDATES_DEFAULT}  # 09-16: 7 → 16。这是唯一还会真丢行的地方(§44.12):
                     # 余数已改成补齐、预算超限已改成切行,只剩这个封顶。
                     # 16×32 = 512 行/步;实测每步 66~110 行,离封顶很远,等于不封。
                     # v2: 2。7×32 = 224 行/步。
                     # smoke step1 实测 n_transition=227(多轮 rollout 碎成 227 行),
                     # MAX_UPDATES=4 只吃 128 行,剩下 99 行被丢(48 同奖励 + 51 随机),
                     # 随机丢弃那 51 行是真丢数据,还把 8 个 uid 组砍到 4 组。
                     # 7 覆盖 224/227;verl 本来就会按 n_transition//mini_bs 再截断,
                     # 所以这个值只是上限,批次小的时候不会硬凑。
                     # smoke 先用这个跑通并量数,真实取值等 n_transition 出来再定。
ROLLOUT_N=4          # v2: 4,保持不动 —— 见文件末尾"关于 rollout.n"。
# 09-17 动态采样(DAPO):同奖励组(优势全 0)整组丢掉,不算 old_log_prob/ref/actor;
# 不够 DYN_MIN_VALID_GROUPS 个有方差的组就再采一批,最多 DYN_MAX_GEN_BATCHES 轮
# (trainer.py _collect_train_batch)。每多一轮 = 多一次 rollout 时间(~20-30 min)。
DYN_SAMPLING=${DYN_SAMPLING:-true}
DYN_MAX_GEN_BATCHES=${DYN_MAX_GEN_BATCHES:-2}
DYN_MIN_VALID_GROUPS=${DYN_MIN_VALID_GROUPS:-$DYN_MIN_DEFAULT}   # train_batch 8 组里要 6 组有方差;null = 8

cd /workspace/projects/agent-lightning
exec .venv/bin/python examples/swe_smith/train_opencode_agent.py \
  --agl-base-url http://127.0.0.1:18082 --agl-key dummy \
  --train-dataset-path examples/swe_smith/train_dataset_mixed.jsonl \
  --val-dataset-path examples/swe_smith/val_dataset_filtered.jsonl \
  --model ${MODEL:-/workspace/models/MiniCPM5-2B-step48-v3} --run-name v4 \
  actor_rollout_ref.rollout.max_model_len=51200 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.95} \
  actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=25600 \
  actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=minicpm5 \
  actor_rollout_ref.rollout.engine_kwargs.vllm.tool_parser_plugin=/workspace/agl-checkpoints/swe_smith_smoke/minicpm5_parser_plugin_085.py \
  actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template=/workspace/models/MiniCPM5-2B/chat_template.jinja \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=51200 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51200 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=51200 \
  data.max_prompt_length=45056 data.max_response_length=6144 \
  agentlightning.trace_aggregator.trajectory_max_prompt_length=45056 \
  agentlightning.trace_aggregator.trajectory_max_response_length=6144 \
  trainer.max_actor_ckpt_to_keep=2 data.shuffle=True \
  actor_rollout_ref.rollout.temperature=1.0 \
  data.train_batch_size=${TRAIN_BATCH} \
  trainer.n_gpus_per_node=${NGPU} \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
  actor_rollout_ref.ref.fsdp_config.fsdp_size=${FSDP_SIZE} \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI} \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  agentlightning.max_ppo_update_times=${MAX_UPDATES} \
  agentlightning.dynamic_sampling.enabled=${DYN_SAMPLING} \
  agentlightning.dynamic_sampling.max_gen_batches=${DYN_MAX_GEN_BATCHES} \
  agentlightning.dynamic_sampling.min_valid_groups=${DYN_MIN_VALID_GROUPS} \
  algorithm.norm_adv_by_std_in_grpo=False \
  trainer.resume_mode=${RESUME_MODE:-auto} \
  "$@" \
  2>&1 | tee -a /workspace/agl-checkpoints/swe_smith_smoke/trainer_v4.log

# ============================== CHANGES vs v2 ==============================
# 1. lr 1e-6 → ${LR}
# 2. max_ppo_update_times 2 → 8     一个 batch 更新八次(§42.2:光改 mini_batch 不够)
# 3. ppo_mini_batch_size 8 → 25     ←← 这个数依赖 n_transition,见下
# 4. rollout.n 保持 4               涨到 8 的性价比是负的,见下
# 5. norm_adv_by_std_in_grpo True → False   取消优势标准差(Dr.GRPO)
# 6. loss_mode 保持 per_rollout_mean        GSPO 已撤(会关掉 rollout 级归一化,§42.1)
# 7. gamma 不动                              adv_estimator=grpo 根本不读它(§42.0)
# 8. KL 保持开启 coef=0.001                  按人类 09-07 明令
# 9. ckpt 目录换成 _v3 + resume_mode=disable 从基座重训:v2 那 284 步是在残缺数据上
#    跑的(评测与基座无统计差异),留着只会污染归因。想续训就把这两行改回 v2 的值。
#
# 关于第 3 条(更新次数怎么算):
#   verl 的 batch 单位是"行"(一次 LLM 调用 = 一个 (prompt,response) 序列),不是轨迹。
#   全局 mini-batch = ppo_mini_batch_size × rollout.n            (fsdp_workers.py:249)
#   更新次数 = n_transition ÷ 全局 mini-batch,再被 MAX_UPDATES 截断 (trainer.py:594-598)
#   v2 实测:mini_bs = 8×4 = 32 行,cap=2 → n_sample_trained ∈ {32,64},与日志完全吻合。
#   采集修好后:一条 rollout ≈25 轮(32,880 调用 ÷ 1,280 条),
#   8 题 × 4 采样 = 32 条 rollout ≈ 800 行 → 800/100 = 8 次更新。
#
# 关于 rollout.n 为什么不涨到 8:
#   n 只决定 GRPO 组内样本数。实测零优势组 46.8%(n=4),反解单条成功率 p≈0.175。
#   n=8 时零组率 45% → 21.5%,可用组 4.4 → 6.3(每步 8 组)。
#   但生成开销翻倍:n=4 每步 ~800 次调用,n=8 ~1,600 次。
#   折算成"每千次调用拿到几个可用组":n=4 → 5.5,n=8 → 3.9。n=8 性价比反而低 30%。
#   零组问题的对症解法是动态采样(只重采死掉的那些组),不是对所有组加倍采样。
# ==========================================================================
