#!/usr/bin/env bash
# GSM8K RLVR(AGL 官方示例)—— 本机改写版。人类 10-04 后手授权:
#   「如果你实在训不出来,就把 AGL 官方的其他几个示例训一遍,比如 gsm8k、MCP 计算器等等」
# 触发条件已成立:s2(SFT ep3 起训 / lr 5e-6)step5 全量探针 50/474 vs 基线 136/474,
# 配对 McNemar p≈0 → VERDICT=STOP。
#
# 为什么不用官方 run_local.sh —— 它有三处会在本机造成实际损害:
#   1. cleanup() 里 `pkill -f agl-server` / `pkill -f agl-controller`,启动时和退出时各跑一次。
#      本机有必须保活的 agl-server(18082)和 agl-controller,会被一起打死。
#      本脚本只按**自己记下的 PID** 收自己起的进程。
#   2. `ray stop --force` —— 本机明令禁用。本脚本退出时用不带 --force 的 ray stop。
#   3. 日志写 /tmp —— 根分区 100% 满、0 可用。本脚本全部写 /data。
#
# 另外从命令行覆盖了官方默认里三个对"拿到可汇报结果"不利的设置
# (train_gsm8k_agent.py 用 parse_known_args(),未知参数进 OmegaConf dotlist,所以不必改官方脚本):
#   trainer.logger=[console]      官方默认带 wandb。那会把运行数据发往外部服务且需要鉴权,未经授权。
#   trainer.val_before_train=True 官方默认 False。没有训前基线就没有对照,等于白训。
#   trainer.save_freq=50          官方默认 -1 = 一个权重都不存。
set -uo pipefail

ROOT=/workspace/projects/agent-lightning
EX=$ROOT/examples/gsm8k
CK=/workspace/agl-checkpoints/gsm8k_r1
DATA=/workspace/dataset/gsm8k/main
MODEL=/workspace/models/Qwen2.5-1.5B-Instruct
PORT=${PORT:-8181}
RAY_PORT=${RAY_PORT:-6399}   # 不用 ray 默认的 6379,避开和任何别人的集群撞车
CARD=${CARD:-2}
KEY=dummy

mkdir -p $CK
SERVER_LOG=$CK/agl_server.log
CTRL_LOG=$CK/agl_controller.log
TRAIN_LOG=$CK/trainer_gsm8k.log

# --- 前置闸门 ---
avail=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ "$avail" -ge 50 ] || { echo "ABORT: /data 只剩 ${avail}G,低于 50G 闸门"; exit 3; }
[ -f $DATA/train-00000-of-00001.parquet ] || { echo "ABORT: 训练集缺失 $DATA"; exit 3; }
# 注意:aria2c 会**预分配整个文件**,所以单看 -f / 看大小都会在下载中途假通过。
# 判完整的依据是控制文件 model.safetensors.aria2 已被 aria2 删除。
[ -f $MODEL/model.safetensors ] || { echo "ABORT: 模型权重缺失 $MODEL"; exit 3; }
[ -f $MODEL/model.safetensors.aria2 ] && { echo "ABORT: 权重还在下载(.aria2 控制文件仍在)"; exit 3; }
ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "ABORT: 端口 $PORT 已被占用"; exit 3; }
pgrep -f "[t]rain_gsm8k_agent.py" >/dev/null && { echo "ABORT: 已有 gsm8k trainer 在跑"; exit 2; }

MY_SERVER=""; MY_CTRL=""; MY_RAY=0
cleanup() {
  # 只收自己起的。绝不 pkill 模式匹配,绝不 ray stop --force。
  [ -n "$MY_CTRL" ]   && kill $MY_CTRL   2>/dev/null
  [ -n "$MY_SERVER" ] && kill $MY_SERVER 2>/dev/null
  # 注意 `ray stop` 是**整节点**的:它会停掉本机所有 ray 进程,不只我这一个集群。
  # 所以只在确实是我起的时候才调,而且用 venv 的 ray(和起的时候同一个)。
  [ "$MY_RAY" = 1 ]   && $ROOT/.venv/bin/ray stop >/dev/null 2>&1
  return 0
}
trap cleanup EXIT INT TERM

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=$CARD
export HF_HOME=/workspace/hf_cache
export TMPDIR=/workspace/tmp

# --- 以下四条全部沿用 restart_s2_smith_trainer.sh:30-40 的本机必备环境变量。
#     10-04 第二次启动就是因为漏了 VLLM_USE_V1 直接挂在 vLLM 引擎初始化上。
export VLLM_USE_V1=1                                            # verl 要 V1 AsyncLLMEngine,不设会 ValueError
export CUDA_MPS_PIPE_DIRECTORY=/workspace/mps_bypass_empty   # 全局 MPS 坏了,不绕会 CUDA error 807
export RAY_local_fs_capacity_threshold=0.99                      # /data 的 df 口径天然 >95%(root 保留块),不抬阈值 ray 会拒绝 spill
export NCCL_NVLS_ENABLE=0                                       # 09-12 Xid31 后 NVLS 组播损坏
# 本机有全局 http_proxy,有**两类**本地调用必须绕开它,只写 127.0.0.1 只挡住了第一类:
#   (a) agent → 127.0.0.1:$PORT 的 rollout / proxy 调用;
#   (b) agl-server → verl 起的 vLLM 上游。**这一跳的地址是 eth0 的 <LAN_IP>:<随机端口>,
#       不是 127.0.0.1** —— verl 用 ray node IP 注册 vLLM server。于是它被发给公司代理
#       <CORP_PROXY>,代理够不到内网地址,全部回 502。10-04 第三次启动就是这么挂的:
#       3359 次 502,200 条 val rollout 零成功,卡 2 吃了 53G 显存但 util 一直 0%。
#       长期跑着的那个 agl-server(18082)躲过这一劫,只是因为它的环境里根本没有 proxy 变量。
# 所以按网卡实际地址生成,别手写 —— 容器/网卡地址换了也不会再踩。
LOCAL_IPS=$(ip -4 -o addr show | awk '{split($4,a,"/"); print a[1]}' | paste -sd,)
export no_proxy="127.0.0.1,localhost,::1,$LOCAL_IPS"
export NO_PROXY="$no_proxy"
# /tmp 所在的根分区 100% 满,ray 的 session 目录必须落 /data。
export RAY_TMPDIR=/workspace/tmp/ray
# 显式指定地址,别让 ray.init() 去自动发现 —— 它只会去默认 temp-dir 找地址文件,
# 而我们的 temp-dir 是自定义的,自动发现要么找不到、要么连到别人的集群上。
export RAY_ADDRESS=127.0.0.1:$RAY_PORT

# 自己起一个专属端口的 ray,不碰任何已有集群。
# **必须用 venv 里的 ray**:裸 `ray` 在 PATH 上解析到 /root/miniconda3 的 ray 2.48/py3.12.2,
# 而 trainer 跑在 .venv 的 ray 2.58/py3.12.13 上 —— 10-04 实测就是这个组合导致
# `RuntimeError: Version mismatch`,集群起来了但 ray.init() 连上去直接拒绝。
if ! ss -ltn 2>/dev/null | grep -q ":$RAY_PORT "; then
  $ROOT/.venv/bin/ray start --head --port=$RAY_PORT \
    --dashboard-host=127.0.0.1 --temp-dir=$RAY_TMPDIR \
    --disable-usage-stats >>$CK/ray.log 2>&1 && MY_RAY=1
fi

$ROOT/.venv/bin/agl-server port=$PORT key=$KEY \
  default_proxy.model_name=$MODEL >$SERVER_LOG 2>&1 &
MY_SERVER=$!
for _ in $(seq 1 90); do
  curl -sf --noproxy '*' "http://127.0.0.1:$PORT/healthz" >/dev/null && break
  sleep 1
done
curl -sf --noproxy '*' "http://127.0.0.1:$PORT/healthz" >/dev/null \
  || { echo "ABORT: agl-server 没起来,看 $SERVER_LOG"; exit 4; }

$ROOT/.venv/bin/agl-controller runner_type=local \
  agl_server.url="http://127.0.0.1:$PORT" agl_server.key=$KEY >$CTRL_LOG 2>&1 &
MY_CTRL=$!

cd $EX
$ROOT/.venv/bin/python train_gsm8k_agent.py \
  --agl-base-url "http://127.0.0.1:$PORT" --agl-key $KEY --run-name zz_r1 \
  --model $MODEL \
  --train-file $DATA/train-00000-of-00001.parquet \
  --val-file   $DATA/test-00000-of-00001.parquet \
  --val-size 200 --seed 42 \
  trainer.logger=[console] \
  trainer.val_before_train=True \
  trainer.test_freq=10 \
  trainer.save_freq=50 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.default_local_dir=$CK \
  trainer.total_epochs=2 \
  >>$TRAIN_LOG 2>&1
echo "gsm8k trainer 退出码 $? ,日志 $TRAIN_LOG"
