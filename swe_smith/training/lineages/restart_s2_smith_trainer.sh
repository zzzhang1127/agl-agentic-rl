#!/bin/bash
# s2 血统:官方 smith harness(mini-swe-agent 形态)上的 agentic RL,2026-10-04 起。
# 由 s1 的失败结论改配方 —— s1 从基座 + lr 2e-6 跑了 90 步:
#   · 训练 reward 平的(斜率 -0.00027/step,置换检验 p=0.43),优化目标自己就没动
#   · 474 全量 109 vs 基座 123(配对 McNemar p=0.065),方向是回退
#   · 48 题探针同向(step10/20 的 9、10 → step50-90 均值 7.2)
# 人类 10-04 决定:改从 **SFT v3 ep3**(474 全量 136,当前最好)起训、lr 2e-6 → **5e-6**、
# 探针一律跑**全量 474**(48 题噪声不够判断力)、明显下降就停。
# 其余(KL 开、长度惩罚、和 OpenCode 血统 v2/v3/v4 分开、无人值守)沿用 s1。
#
# 和 v2–v4 的根本区别不在超参,在 harness:smith 没有上下文压缩,所以单条 rollout 的
# **累计** prompt token 高达 1.50M 均值(474 题实测 p50 1.23M / max 7.09M),是 OpenCode
# 带压缩时的一个数量级以上。这决定了三件事:
#   1. 批量必须小。一步 n 条 rollout 要跑 n×1.5M token 的 old_log_prob + ref + actor,
#      官方 train_batch_size=4 不是随手写的,是被这个量级逼出来的。
#   2. rollout 阶段靠 prefix caching 救命(逐轮共享前缀,只 prefill 新增部分);
#      但 log_prob/update 三遍是全量重算,省不掉,所以步时由批量线性决定。
#   3. prompt:completion = 45:1,是彻底的 prefill-bound 负载 → max_num_batched_tokens
#      从 8192 提到 16384,把卡的算力吃满(人类 10-02:「有充足显存的每张卡的算力都要
#      得到充分利用」)。
#
# 卡的选择:实测只有 1/2/3/5 四张在评测退场后有 ≥50G 空闲(卡 0 被邻居 1619028 占 67.8G、
# 卡 4 是人类自己的 llama-server 几乎满、卡 6/7 邻居占到只剩 11–17G)。FSDP 是同步的,
# 把一张只剩 16G 的卡拉进来会由它定全队节奏,所以**只用 1/2/3/5**,"充分利用"靠
# max_num_seqs×副本数 和 prefill 批量做到,不靠凑卡数。
#
#   bash restart_s1_smith_trainer.sh            # 正式跑
#   SMOKE=1 bash restart_s1_smith_trainer.sh    # 一步 smoke,量步时和采集率
set -u

ROOT=/workspace/projects/agent-lightning
EX=$ROOT/examples/swe_smith
CKPT_DEFAULT=/workspace/agl-checkpoints/smith_rl_s2

export CUDA_MPS_PIPE_DIRECTORY=/workspace/mps_bypass_empty  # 全局 MPS 坏了会 807
export NCCL_NVLS_ENABLE=0        # 09-12 Xid31 后 NVLS 组播损坏,禁用回退 NVLink 环
export VLLM_USE_V1=1             # verl 要求 V1 AsyncLLMEngine
export RAY_TMPDIR=/workspace/ray_tmp
export RAY_local_fs_capacity_threshold=0.99
export AGL_CKPT_DIR=${AGL_CKPT_DIR:-$CKPT_DEFAULT}
export AGL_MIN_CAPTURE_RATE=${AGL_MIN_CAPTURE_RATE:-0.9}
export CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,5}
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"

NGPU=${NGPU:-4}
FSDP_SIZE=${FSDP_SIZE:-4}
LR=${LR:-5e-6}                   # 人类 10-04 指定:s1 的 2e-6 在 90 步内没推动目标
ROLLOUT_N=${ROLLOUT_N:-4}        # GRPO 组内样本数

# ---- 批量 ----
# verl 硬约束 train_batch_size >= ppo_mini_batch_size(workers/config/actor.py:216),
# 全局 mini-batch = ppo_mini × rollout.n 行(fsdp_workers.py:249),
# 更新次数 = n_transition ÷ 全局 mini-batch,再被 max_ppo_update_times 截断。
# 实测一条 smith rollout ≈30.2 轮 = 30 行,所以 8×4=32 条 → ≈960 行 → 960/(8×4)=30 次
# = 刚好一个 epoch。MAX_UPDATES 给 32 只当上限用(verl 自己会按行数截断),
# 这样既不丢数据也不超过一个 epoch。
TRAIN_BATCH=${TRAIN_BATCH:-8}
PPO_MINI=${PPO_MINI:-8}
MAX_UPDATES=${MAX_UPDATES:-32}

# ---- 长度几何:和 123/474 基线逐项对齐 ----
CTX=51200          # = data.max_prompt_length + max_response_length,vLLM 窗口
MAXPROMPT=47104    # 51200 − 4096,实测最大 45448,留 1.6k 余量
MAXRESP=4096       # 和 job 模板里的 AGL_MAX_TOKENS 一致

if [ "${SMOKE:-0}" = 1 ]; then
  TRAIN_BATCH=2; PPO_MINI=2; MAX_UPDATES=4
  SMOKE_ARGS="trainer.total_training_steps=1 trainer.test_freq=0 trainer.save_freq=1000
              agentlightning.dynamic_sampling.enabled=false trainer.resume_mode=disable"
  RUN=s2smoke
  LOGF=$AGL_CKPT_DIR/trainer_s2_smoke.log
else
  SMOKE_ARGS=""
  RUN=s2
  LOGF=$AGL_CKPT_DIR/trainer_s2.log
fi

mkdir -p $AGL_CKPT_DIR/agl-logs/new $RAY_TMPDIR
cd $ROOT

exec .venv/bin/python examples/swe_smith/train_smith_agent.py \
  --agl-base-url http://127.0.0.1:18082 --agl-key dummy \
  --train-dataset-path examples/swe_smith/train_dataset_mixed.jsonl \
  --val-dataset-path examples/swe_smith/val_dataset_filtered.jsonl \
  --max-val-instances ${VAL_N:-48} \
  --model ${MODEL:-/workspace/models/MiniCPM5-2B-sft-v3-ep3} --run-name $RUN \
  \
  `# ---- 新血统的 rollout Job:纯文本 smith agent + stdlib 客户端 ----` \
  agentlightning.k8s.job_template_path=$EX/job-template-smith.yaml \
  \
  `# ---- vLLM:TP=1 → 4 个独立副本,每卡一个,四张卡同时出 rollout ----` \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len=$CTX \
  actor_rollout_ref.rollout.max_num_seqs=${VLLM_SEQS:-12} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${VLLM_PREFILL:-16384} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.95} \
  actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=25600 \
  actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template=/workspace/models/MiniCPM5-2B/chat_template.jinja \
  actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=null \
  actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=null \
  actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=null \
  `# enforce_eager:10-02 smoke 实测 gen 占掉整步 956.8s 里的 914.3s(95.6%),` \
  `# 而 ref+old_log_prob+update_actor+update_weights 合计只有 41.7s(4.4%)。` \
  `# 瓶颈全在解码延迟上,所以关掉 eager 换 CUDA graph(官方 smith 配方也是 False;` \
  `# v2–v4 设 True 是当年显存紧张留下的,现在 1/2/3/5 卡各有 52–94G 空闲,` \
  `# graph 那 1–2G 捕获开销完全吃得下)。启动失败就 EAGER=True 退回已验证路径。` \
  actor_rollout_ref.rollout.enforce_eager=${EAGER:-False} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.logprobs_mode=null \
  \
  `# ---- 长度几何 + 动态打包(短轮打包、长轮单独走) ----` \
  data.max_prompt_length=$MAXPROMPT data.max_response_length=$MAXRESP \
  data.truncation=left \
  agentlightning.trace_aggregator.trajectory_max_prompt_length=$MAXPROMPT \
  agentlightning.trace_aggregator.trajectory_max_response_length=$MAXRESP \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$CTX \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$CTX \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$CTX \
  \
  `# ---- 批量 / 优化 ----` \
  data.train_batch_size=$TRAIN_BATCH data.shuffle=True \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI \
  agentlightning.max_ppo_update_times=$MAX_UPDATES \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.enable_rollout_level_advantage=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=per_rollout_mean \
  \
  `# ---- DAPO 动态采样:全同奖励的组(优势恒 0)丢掉再采 ----` \
  agentlightning.dynamic_sampling.enabled=${DYN:-true} \
  agentlightning.dynamic_sampling.max_gen_batches=2 \
  agentlightning.dynamic_sampling.min_valid_groups=${DYN_MIN:-6} \
  agentlightning.shuffle_train_rows=True \
  \
  `# ---- 训推偏差校正(09-21 欠的账:v2–v4 全程没开) ----` \
  `#    rollout_log_probs 不是 verl 的 calculate_log_probs 给的,是 AGL 代理在每个上游` \
  `#    请求里强塞 logprobs=True(server/proxy.py:83)再由 rollout_adapter.py:999 汇总的。` \
  `#    所以这里不碰 calculate_log_probs。要是某一行没有 logprobs,整批就不带这个字段,` \
  `#    trainer.py:851 的 'in batch.batch' 会让校正静默跳过 —— smoke 要看 rollout_corr/ 有没有出。` \
  algorithm.rollout_correction.rollout_is=${ROLLOUT_IS:-token} \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  \
  `# ---- FSDP / 稳定性(全部是 OpenCode 血统上流血换来的,官方 smith 配置没有) ----` \
  trainer.n_gpus_per_node=$NGPU \
  actor_rollout_ref.nccl_timeout=3600 \
  trainer.nccl_timeout=3600 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=$FSDP_SIZE \
  actor_rollout_ref.ref.fsdp_config.fsdp_size=$FSDP_SIZE \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  \
  `# ---- 探针与存盘 ----` \
  trainer.test_freq=${TEST_FREQ:-20} \
  trainer.save_freq=${SAVE_FREQ:-5} \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.val_before_train=False \
  trainer.resume_mode=${RESUME_MODE:-auto} \
  agentlightning.rollout_timeout_seconds=${ROLLOUT_TO:-2400} \
  $SMOKE_ARGS \
  "$@" \
  2>&1 | tee -a $LOGF

# =========================== 和官方 smith 配方的差异清单 ===========================
#  1. lr 1e-6 → 2e-6                     人类 10-02 指定
#  2. use_kl_loss False → True (0.001)   人类明令「KL 开,为了保持不偏」;注意这会
#                                        触发 need_reference_policy(ppo/utils.py:76)
#                                        → 多一个 ref worker 和一遍全量 log_prob
#  3. TP 2 → 1                           4 张卡 = 4 个独立 vLLM 副本,而不是 2 个 TP2 副本。
#                                        smith 是 prefill-bound,TP 的通信开销换不回吞吐,
#                                        副本多 = 并发高 = 每张卡都在算
#  4. max_model_len 32768 → 51200        和 123/474 基线同窗口;实测 prompt max 45448,
#                                        32768 会让 ~15% 的 episode 死于溢出
#  5. num_gpu_blocks_override=25600      KV 绝对预算 409,600 token = 16.4G/卡
#                                        (42.0 KiB/token × 42 层 × 2 KV head)。
#                                        util 0.95 只是 profiling 允许上限,真实大小由它定
#  6. chat_template 换成模型自带的        官方那个 swe_smith_chat_template.jinja 是
#                                        Qwen 系的;换掉才能和基线同一套 serving
#  7. 关掉 auto_tool_choice / tool_parser  smith 不走 tool_calls,整条 parser bug 线
#                                        (§44 CDATA 缩进、finish_reason 丢块)在这条路上
#                                        不存在。评测舰队也是这么起的,保持一致
#  8. moe_backend=null                   dense 2B 用不上,且 vLLM 0.8.5 不认这个参数
#  9. train_batch 4 → 8,max_ppo_update 2 → 32
#                                        官方 2 次更新只吃掉 ~7% 的采集行;改成按
#                                        "一个 epoch"走,把 rollout 的钱花完
# 10. nccl_timeout=3600 放在 actor_rollout_ref 下
#                                        必须在这个节点:FSDP worker 从
#                                        actor_rollout_ref.config 读(fsdp_workers.py
#                                        init_process_group),不读 trainer.nccl_timeout。
#                                        长 update 超过 600s 默认值 → watchdog abort →
#                                        ActorDiedError。这是 v3 血统最贵的一个教训
# 11. 加 enable_rollout_level_advantage / per_rollout_mean / 动态采样 / shuffle_train_rows
# 12. 加 rollout_correction.rollout_is=token  v2–v4 全程没做训推 IS 校正(09-21 人类指出)。
#                                        v4 实测 log_ppl_diff 3.5e-4,阈值 2.0 基本不会
#                                        触发,是安全网不是主动干预
# 13. truncation error → left            无人值守下,一行超长不该让整步崩掉
# 14. test_freq 32 → 20 + --max-val-instances 48
#                                        48 题探针 ≈ 一个 rollout wave,便宜;
#                                        全量 474 的里程碑评测在外面单独跑(temp 0.6)
# 15. save_freq 8 → 5,keep=1            一份 FSDP resume 包 ~29G,磁盘闸门 50G,
#                                        keep=2 在存盘瞬间会压到闸门以下
# ================================================================================
