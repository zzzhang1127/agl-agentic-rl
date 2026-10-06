#!/bin/bash
# s4 血统:2026-10-05。配方 = s3,改动四处,**全部由同一个病根逼出来的**。
#
# ── 0. 病根:每条 episode 被打碎成「一轮一行」,多轮 credit assignment 从未生效 ──
#   AGL 的 rollout adapter 要把一条 episode 的各轮拼成一条训练序列,判据是
#   「第 k+1 轮的 prompt_ids 是否以(第 k 轮 prompt+response)为前缀」
#   (rollout_adapter.py:816-864)。不成立就把当前组当成一行独立训练样本 flush 掉,
#   **并给它打上整条 episode 的最终奖励**。
#   实测:s1 和 s3 **每一步都是 unmerged_rollouts: 64 / 64**,
#   n_sample_collected = 2044/1613/1416/1169/1049(步 1-5)—— 64 条 episode 碎成一两千行,
#   一条 39 轮的 episode 因此拿到 39 倍于 1 轮 episode 的梯度权重。
#   于是 enable_rollout_level_advantage=True 和 policy_loss.loss_mode=per_rollout_mean
#   这两项从来没拿到它们要的那条合并序列。s1/s2/s3 全中招。
#
#   机理(逐字符定位到模板):MiniCPM5 的 chat_template.jinja 对**历史里的** assistant 轮
#   **无条件**插入 `<think>\n\n</think>\n\n`(:121-122),而**生成位置**只在显式传了
#   `enable_thinking is false` 时才插(:170-172)。训练侧渲染没拿到那个 kwarg,于是
#     第 k 轮 prompt 结尾 = `<|im_start|>assistant\n`
#     第 k+1 轮重渲染同一轮 = `<|im_start|>assistant\n<think>\n\n</think>\n\n`
#   差 4 个 token,前缀判据必败。两条血统的 mismatch 记录逐字相同,是同一个病因。
#
#   修法(已落地,见 models/*/chat_template.jinja.bak-pre-thinkfix):把生成位置改成
#   **默认非思考**,和 :121-122 对齐。token 级验收:不传 kwarg 时前缀判据 False → True;
#   且 enable_thinking=False / True 两条路补丁前后**渲染逐字相同**,所以
#   基座 123/474、SFT 136/474 这些历史基线仍然可比。
#
# ── 1. 长度几何重算:MAXRESP 4096 → 43008(**不改这项等于没修**) ──────────────
#   合并后一条训练行的 response = 所有轮的输出 + 所有观测。实测 635 条 episode:
#     第 1 轮 prompt  中位 1212  p95 1381  max 5277
#     合并 response   中位 11889 p95 22953 max 44532(34 轮上限口径)
#   现行 MAXRESP=4096 会截断 **88%** 的合并行。改成 MAXPROMPT=8192(截断 0.0%)+
#   MAXRESP=43008(截断 ~0),和 CTX=51200 对齐。adapter 自己的
#   trajectory_max_response_length 也跟着这个值走(:673-674 按它截断)。
#
# ── 2. PPO_MINI 16 → 4(**不改这项会空转 25%**) ───────────────────────────────
#   全局 mini-batch = PPO_MINI × rollout.n 行。合并前一步采到 ~1262 行 → 约 20 次
#   优化更新;合并后一步只有 ~64 行(= TRAIN_BATCH 16 × ROLLOUT_N 4,动态采样后更少),
#   配 mini_bs=64 就只剩 1 次更新,且凑不满时 _pad_with_neutral_duplicates 会补中性复制
#   (trainer.py:757-789)。PPO_MINI=4 → mini_bs=16 行 → 每步 3 次更新、0 填充。
#
# ── 3. SMITH_MAX_TURNS 维持 40(**一度想砍 34,被当晚数据否了,记在这里别再犯**) ──
#   训练 rollout 里 reward=1 的 episode 轮数 p90=30 / max=32,看着砍 34 是免费的;
#   但 10-05 val-474 实测:解出来的 127 条里 21 条(16.5%)用了 >34 轮,18 条正好顶到 40。
#   两个分布不是一回事(训练题 vs val 题、RL 策略 vs SFT 策略),不能互相代入。
#   砍 34 省 ~11% gen,赌掉一成多成功样本 —— 不划算。训练=评测=40,口径天然对齐。
#
# 预期步时(按实测外推,待 SMOKE 验证):前向 token 量降 18.3×,
# update_actor+old_log_prob+ref 这 71% 的部分 ÷≈15,gen 约 440s → 一步约 10 分钟
# (s3 实测 54 分钟)。gen 之后会变成主要成本。
#
# ── 以下为 s3 的原始论证,逐条仍然成立 ──────────────────────────────────────
#
# 和 s1/s2 的区别只有一处是真正的新东西:**奖励从「F2P 全过才给分」改成「F2P 通过比例」**。
# 其余超参回到 s1 已验证的那一组。为什么这么配,逐条都有 s1 的 5735 条 rollout 实测支撑:
#
# ── 1. 奖励:F2P 通过比例(稠密) ──────────────────────────────────────────────
#   s1 失败的根因不是奖励稀疏、也不是每组 4 条太少(gsm8k 涨 +pp 用的也是 n=4),
#   是 **credit assignment 的梯度相互抵消**:一个标量摊给 ~5400 个 token / ~30 轮,
#   而 30 轮里大部分是 ls/cat/grep 导航,成功和失败里长得一样 —— 成功给 +A、失败给 −A,
#   在 batch 里对消。**抵消 Adam 救不了**(一阶矩≈0 而二阶矩大 → 更新≈0),
#   这就是 entropy 90 步只从 0.24842 走到 0.25363 的原因。
#   改成通过比例的实测收益(examples/swe_smith/f2p_offline_check.py):
#     · 零方差组 40.4% → **30.3%**,有梯度的组 **×1.17**
#     · 四条全错的 416 组救回 **145 个(34.9%)**
#     · 被救活组的组内 std 中位 **0.2500**(原本有方差组的 0.59×)—— 够用,不需要靠
#       std 归一去放大,所以 norm_adv_by_std_in_grpo 保持 False(见下)
#   注意这项由 job-template-smith-s4.yaml 的 SMITH_F2P_PARTIAL=1 控制(在 rollout pod 里读),
#   **只对训练生效**;验证集永远二值,否则基座 123/474、SFT 136/474 失去可比性。
#
# ── 2. lr 回 2e-6(不是 s2 的 5e-6) ─────────────────────────────────────────
#   人类 10-04 给 s2 指定 5e-6,结果 4 步坍塌(response_length 274→112→60)。机理:
#   这批数据里唯一不被抵消的信号是**长度**(失败比成功长 1208 token、55.1% 的失败撞上限),
#   5e-6 四步就把它学成了「少写 token」。部分分**没有**消除长度捷径
#   (高分−低分 token 差 −1597 → 仅 −1544),因为「短的更好」在这批数据里是真事实
#   (0 分失败中位 40 轮撞上限、有分失败中位 31 轮)。部分分给的是另一重保护:
#   **近失手(0.33)严格优于什么都不写(0)** —— 这正是 s2 坍塌时缺的反向压力。
#   所以奖励改了、lr 必须收回来,两件事不能一起赌。
#
# ── 3. 训练集 700 题(train_dataset_s3.jsonl) ───────────────────────────────
#   = 按 s1 实测通过率筛出的 741 题,再剔掉 41 道 **P2P 回归护栏为空**的题:
#   SMITH_F2P_ONLY=1 把 P2P 过滤到只剩含 F2P 的文件,保留率 6.4%(中位 269→32),
#   有一批题过滤完 P2P 为 0 —— 它们压根不构成回归信号,留着只是白占采样预算。
#   筛题本身的收益比预想小(有效算力只 ×1.24),但它是免费的。
#
# ── 4. P2P 保持二值硬闸(破任何一条 → 奖励 0) ───────────────────────────────
#   实测过四种口径,软化闸门只能把有梯度组从 69.7% 买到 71.1%(净 +20 组 = 1.4%),
#   代价是:乘性软闸会在 **119 个组**里给「破 3 条」比「破 8 条」更高的分(教「破少点就行」);
#   加性软罚(可为负)会让「什么都没干」严格优于「试了但改坏」,在 **233 个组(16.2%)**
#   里给白忙的 rollout 正优势 —— 那正是 s2 坍塌的方向。
#   而且破 P2P 的 1276 条里 786 条(61.6%)是**破光**(P2P 分母中位 34,一补丁打掉 34 个
#   原本通过的测试),乘性闸给它们 fr×0=0,和硬闸一字不差。所以硬闸不动。
#
# ── 5. norm_adv_by_std_in_grpo 保持 False ──────────────────────────────────
#   开归一能把总 |优势| 质量从 ×1.017 提到 ×1.162、把救活组的人均优势提到 1.44×,
#   但会把 11 个 std<0.05 的组放大 20 倍以上(4 个放大 50 倍)。那是部分分**新增**的
#   噪声面,而且 std 归一正是 DAPO/Dr.GRPO 明确去掉的难度偏置。
#   被救活组 std 中位 0.25 已是可用量级,不靠归一。一次只改一件事,否则归因全乱。
#
# ── 6. 探针(人类要求「维护探针防止训练崩塌」) ───────────────────────────────
#   两层,因为 s2 证明了 **奖励不会报警**(坍塌 5 步里 score 全程 0.40~0.56):
#     Tier 1  s3_watch.py   每步读 trainer 日志 + rollout 日志,盯长度/零方差/采集率,
#                           连续两步越线就 SIGINT 暂停(存了盘,resume_mode=auto 可续)
#     Tier 2  s3_cycle.sh   每 PROBE_EVERY 步停训跑**全量 474**,配对 McNemar 判显著
#   阈值按 s1(健康 91 步)和 s2(坍塌 5 步)实测标定,见 s3_watch.py 里的注释。
#
# 卡:只用 1/2/3/5(卡 0 被邻居占 67.8G、卡 4 是人类自己的 llama-server、卡 6/7 只剩 11-17G)。
#
#   bash restart_s3_smith_trainer.sh            # 正式跑
#   SMOKE=1 bash restart_s3_smith_trainer.sh    # 一步 smoke,量步时和采集率
set -u

ROOT=/workspace/projects/agent-lightning
EX=$ROOT/examples/swe_smith
CKPT_DEFAULT=/workspace/agl-checkpoints/smith_rl_s4

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
LR=${LR:-2e-6}                   # s1 已验证不坍塌;s2 的 5e-6 四步崩,见上文 §2
ROLLOUT_N=${ROLLOUT_N:-4}        # GRPO 组内样本数(n=8 只 ×1.25,不值双倍 gen 时间)

# ---- 批量 ----
# verl 硬约束 train_batch_size >= ppo_mini_batch_size(workers/config/actor.py:216),
# 全局 mini-batch = ppo_mini × rollout.n 行(fsdp_workers.py:249)。
TRAIN_BATCH=${TRAIN_BATCH:-16}
PPO_MINI=${PPO_MINI:-4}         # 合并修好后一步只有 ~64 行,16 会只剩 1 次更新+补中性复制,见上文 §2
MAX_UPDATES=${MAX_UPDATES:-32}

# ---- 长度几何:和 123/474 基线逐项对齐 ----
CTX=51200          # = data.max_prompt_length + max_response_length,vLLM 窗口
MAXPROMPT=8192     # 合并后 prompt 只剩第 1 轮,实测 max 5277 → 8192 截断 0.0%,见上文 §1
MAXRESP=43008      # 合并 response 实测 p95 22953 / max 44532;4096 会截断 88%,见上文 §1

if [ "${SMOKE:-0}" = 1 ]; then
  TRAIN_BATCH=2; PPO_MINI=2; MAX_UPDATES=4
  SMOKE_ARGS="trainer.total_training_steps=1 trainer.test_freq=0 trainer.save_freq=1000
              agentlightning.dynamic_sampling.enabled=false trainer.resume_mode=disable"
  RUN=s4smoke
  LOGF=$AGL_CKPT_DIR/trainer_s4_smoke.log
else
  SMOKE_ARGS=""
  RUN=s4
  LOGF=$AGL_CKPT_DIR/trainer_s4.log
fi

mkdir -p $AGL_CKPT_DIR/agl-logs/new $RAY_TMPDIR
cd $ROOT

exec .venv/bin/python examples/swe_smith/train_smith_agent.py \
  --agl-base-url http://127.0.0.1:18082 --agl-key dummy \
  `# 700 题 = 741 筛后题 − 41 道 P2P 护栏为空的题,见上文 §3` \
  --train-dataset-path examples/swe_smith/train_dataset_s3.jsonl \
  --val-dataset-path examples/swe_smith/val_dataset_filtered.jsonl \
  --max-val-instances ${VAL_N:-48} \
  --model ${MODEL:-/workspace/models/MiniCPM5-2B-sft-v3-ep3} --run-name $RUN \
  \
  `# ---- 记录:swanlab(train_smith_agent.py:139 已是默认,这里显式写出来留痕) ----` \
  `#    云端 project agentlightning / run swe_smith_async_<model>_fsdp_s3` \
  trainer.logger='["console","swanlab"]' \
  \
  `# ---- 新血统的 rollout Job:纯文本 smith agent + stdlib 客户端 ----` \
  agentlightning.k8s.job_template_path=$EX/job-template-smith-s4.yaml \
  \
  `# ---- vLLM:TP=1 → 4 个独立副本,每卡一个,四张卡同时出 rollout ----` \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len=$CTX \
  actor_rollout_ref.rollout.max_num_seqs=${VLLM_SEQS:-16} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${VLLM_PREFILL:-16384} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.95} \
  actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=25600 \
  actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template=/workspace/models/MiniCPM5-2B/chat_template.jinja \
  actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=null \
  actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=null \
  actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=null \
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
  `# False = Dr.GRPO/DAPO 的去偏口径。实测开 True 会把 11 个 std<0.05 的组放大 20×,见 §5` \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.enable_rollout_level_advantage=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=per_rollout_mean \
  \
  `# ---- DAPO 动态采样:全同奖励的组(优势恒 0)丢掉再采 ----` \
  `#    预测:部分分把零方差组从 s1 的 41.0% 压到 ~30%(离线算 40.4%→30.3%)。` \
  `#    **10-05 六步实测 0/3/2/4/3 /16,均值 15.0%** —— 比预测还好一倍,部分分在它` \
  `#    瞄准的那个指标上是成功的(工单「强化组内差异」这条达成了)。` \
  `#    在线读数 = training/dynamic_sampling/n_groups_zero_adv ÷ n_groups_seen。` \
  `#    **原注释「重采变少 → 步时变快」写反了,已删**:重采只在留存组不足时才花钱` \
  `#    (s1 九十一步里只重采 3 次,s3 六步 0 次,全在 round 1/2 就够 min_valid_groups),` \
  `#    所以它从来不是步时的分母。步时完全由训练 token 数决定(update_actor 0.0793 ms/token,` \
  `#    乘出来无残差),而 token 数由轮数决定、prompt 随轮数平方增长 —— 见 pitfalls §51/§54。` \
  agentlightning.dynamic_sampling.enabled=${DYN:-true} \
  agentlightning.dynamic_sampling.max_gen_batches=2 \
  agentlightning.dynamic_sampling.min_valid_groups=${DYN_MIN:-6} \
  agentlightning.shuffle_train_rows=True \
  \
  `# ---- 训推偏差校正(09-21 欠的账:v2–v4 全程没开;s1 实测 is_mean 0.9946,是安全网) ----` \
  algorithm.rollout_correction.rollout_is=${ROLLOUT_IS:-token} \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  \
  `# ---- FSDP / 稳定性 ----` \
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
  `#    TEST_FREQ=0:48 题内置探针噪声不够判断力(人类 10-04),由 s3_cycle.sh 的全量 474 取代` \
  trainer.test_freq=${TEST_FREQ:-0} \
  trainer.save_freq=${SAVE_FREQ:-5} \
  `#    keep=1 会把交付物删掉:Tier-2 探针一次 45 分钟,等它判完「step N 好」,` \
  `#    训练已经往前走了 2-3 步,那份权重早被回收了。3 份 × 29G = 87G,` \
  `#    /data 还剩 230G,买得起。` \
  trainer.max_actor_ckpt_to_keep=${KEEP_CKPT:-3} \
  trainer.val_before_train=False \
  trainer.resume_mode=${RESUME_MODE:-auto} \
  agentlightning.rollout_timeout_seconds=${ROLLOUT_TO:-2400} \
  $SMOKE_ARGS \
  "$@" \
  2>&1 | tee -a $LOGF
