#!/usr/bin/env bash
# GSM8K 收官评测：**全量 1319 题**，基座和训练后权重各跑一次，同一条代码路径。
#
# 为什么必须这么做（[[eval-baseline-noise-ceiling]]）：训练中的 val 只有 200 题，
# 基线自身 1σ = sqrt(0.66*0.34/200) = 3.35pp，把 z 封在 ~3.0 —— 同样的 +10.8pp
# 在 1319 题上 1σ 只有 1.30pp。**小题量只看趋势，收官必须基线 + 终点同上大题量重测。**
#
# 用法:
#   bash eval_gsm8k_full.sh base                      # 基座
#   bash eval_gsm8k_full.sh step460 /path/to/hf_dir   # 训练后（先 merge 成 HF 目录）
#
# merge 步骤（FSDP 分片 → HF 目录）:
#   .venv/bin/python -m verl.model_merger merge --backend fsdp \
#     --local_dir $CK/global_step_<N>/actor \
#     --target_dir /workspace/models/gsm8k_r1_step<N>_hf
#
# 和 run_gsm8k_zz.sh 的差别只有三处：`trainer.val_only=True`（verl
# ray_trainer.py:1263 支持，跑完 val_before_train 就 return）、`--val-size 0`
# （train_gsm8k_agent.py:174 的 `val_size > 0` 判据 ⇒ 0 表示不抽样、用整个 1319）、
# 以及端口/ray 端口错开，免得和还在跑的 trainer 撞。其余环境变量一字不动地沿用，
# 那些坑（MPS 绕过 / no_proxy 按网卡生成 / ray 版本必须用 venv 的）全都还在。
set -uo pipefail

TAG=${1:?用法: eval_gsm8k_full.sh <tag> [模型目录]}
ROOT=/workspace/projects/agent-lightning
EX=$ROOT/examples/gsm8k
CK=/workspace/agl-checkpoints/gsm8k_r1
DATA=/workspace/dataset/gsm8k/main
BASE=/workspace/models/Qwen2.5-1.5B-Instruct
MODEL=${2:-$BASE}
PORT=${PORT:-8183}
RAY_PORT=${RAY_PORT:-6397}
CARD=${CARD:-2}
KEY=dummy

OUT=$CK/eval_full1319_$TAG
mkdir -p "$OUT"
SERVER_LOG=$OUT/agl_server.log
CTRL_LOG=$OUT/agl_controller.log
EVAL_LOG=$OUT/eval.log

# --- 前置闸门（和训练脚本同一套） ---
avail=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ "$avail" -ge 50 ] || { echo "ABORT: /data 只剩 ${avail}G，低于 50G 闸门"; exit 3; }
[ -f $DATA/test-00000-of-00001.parquet ] || { echo "ABORT: 测试集缺失"; exit 3; }
[ -f "$MODEL/config.json" ] || { echo "ABORT: 模型目录不像 HF 权重: $MODEL"; exit 3; }
ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "ABORT: 端口 $PORT 已被占用"; exit 3; }

MY_SERVER=""; MY_CTRL=""; MY_RAY=0
cleanup() {
  # 只收自己起的。绝不 pkill 模式匹配（会打死必须保活的 18082 agl-server），
  # 绝不 ray stop --force。
  [ -n "$MY_CTRL" ]   && kill $MY_CTRL   2>/dev/null
  [ -n "$MY_SERVER" ] && kill $MY_SERVER 2>/dev/null
  [ "$MY_RAY" = 1 ]   && $ROOT/.venv/bin/ray stop >/dev/null 2>&1
  return 0
}
trap cleanup EXIT INT TERM

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=$CARD
export HF_HOME=/workspace/hf_cache
export TMPDIR=/workspace/tmp
export VLLM_USE_V1=1
export CUDA_MPS_PIPE_DIRECTORY=/workspace/mps_bypass_empty
export RAY_local_fs_capacity_threshold=0.99
export NCCL_NVLS_ENABLE=0
LOCAL_IPS=$(ip -4 -o addr show | awk '{split($4,a,"/"); print a[1]}' | paste -sd,)
export no_proxy="127.0.0.1,localhost,::1,$LOCAL_IPS"
export NO_PROXY="$no_proxy"
export RAY_TMPDIR=/workspace/tmp/ray
export RAY_ADDRESS=127.0.0.1:$RAY_PORT

if ! ss -ltn 2>/dev/null | grep -q ":$RAY_PORT "; then
  $ROOT/.venv/bin/ray start --head --port=$RAY_PORT \
    --dashboard-host=127.0.0.1 --temp-dir=$RAY_TMPDIR \
    --disable-usage-stats >>$OUT/ray.log 2>&1 && MY_RAY=1
fi

$ROOT/.venv/bin/agl-server port=$PORT key=$KEY \
  default_proxy.model_name=$MODEL >$SERVER_LOG 2>&1 &
MY_SERVER=$!
for _ in $(seq 1 90); do
  curl -sf --noproxy '*' "http://127.0.0.1:$PORT/healthz" >/dev/null && break
  sleep 1
done
curl -sf --noproxy '*' "http://127.0.0.1:$PORT/healthz" >/dev/null \
  || { echo "ABORT: agl-server 没起来，看 $SERVER_LOG"; exit 4; }

$ROOT/.venv/bin/agl-controller runner_type=local \
  agl_server.url="http://127.0.0.1:$PORT" agl_server.key=$KEY >$CTRL_LOG 2>&1 &
MY_CTRL=$!

cd $EX
$ROOT/.venv/bin/python train_gsm8k_agent.py \
  --agl-base-url "http://127.0.0.1:$PORT" --agl-key $KEY --run-name zz_eval_$TAG \
  --model "$MODEL" \
  --train-file $DATA/train-00000-of-00001.parquet \
  --val-file   $DATA/test-00000-of-00001.parquet \
  --val-size 0 --seed 42 \
  trainer.logger=[console] \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.save_freq=-1 \
  trainer.default_local_dir=$OUT \
  >>$EVAL_LOG 2>&1
rc=$?
echo "[$TAG] 退出码 $rc"
grep -oE "val/reward:[0-9.]+" $EVAL_LOG | tail -3
echo "日志 $EVAL_LOG"
