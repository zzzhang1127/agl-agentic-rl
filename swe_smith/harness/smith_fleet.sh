#!/bin/bash
# 2026-10-02: val-474 评测舰队,跑**官方 smith harness**(mini-swe-agent 形态:
# 纯文本一轮一个 ```bash 块,没有 tool_calls JSON,没有上下文压缩,没有代理)。
#
# 和 v5_fleet.sh(OpenCode 口径)的区别,都是故意的:
#   1. vLLM 不挂 tool parser —— smith 不走 tool_calls,整条 parser bug 线(§44 CDATA 缩进、
#      finish_reason 丢块)在这条路上不存在;
#   2. 不插 tool_trunc_proxy —— 步数上限是 smith_agent 自带的 SMITH_MAX_TURNS,不用代理伪造;
#   3. 端口必须落在 18000–18009 —— 容器里 prepare_smith_net() 的 iptables 只放通这一段
#      (run_smith_sweep.py:137-138),用别的端口 agent 连不上 vLLM 且表现为静默超时。
#
# 口径对齐(为了能和 OpenCode 的 474 直接比):temp 0.6 / top_p 0.95 / top_k 20、
# 单题 600s、评测 600s、步数上限 40。注意 upstream smith_rollout.py 把 temperature
# 硬编码成 1.0,会盖掉 --override-generation-config,所以这里显式传 SMITH_TEMPERATURE。
#
#   smith_fleet.sh start <MODEL_DIR> <OUT_DIR_NAME> <TAG>
#   smith_fleet.sh wait  <TAG> <OUT_DIR_NAME>
#   smith_fleet.sh stop  <TAG>
set -u
ROOT=/workspace/agl-checkpoints/swe_smith_smoke
PY=/workspace/projects/agent-lightning/.venv/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/workspace/mps_bypass_empty   # 全局 MPS 坏了会 807,见 memory
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"
cd $ROOT

# 10-05:这个目录下的 smith_agent.py 曾经是 examples/ 那份的分叉拷贝(949 行 vs 1220 行),
# 于是在 examples/ 改好的原生格式解析根本进不了评测路径(训练走 configMap,评测走这里)。
# 现在两份必须逐字相同,不同就拒跑 —— 45 分钟的评测不该跑完才发现用的是旧 harness。
CANON=/workspace/projects/agent-lightning/examples/swe_smith/agents/smith_agent.py
if ! cmp -s $ROOT/smith_agent.py $CANON; then
  echo "FATAL: smith_agent.py 与 $CANON 不一致。先 cp -p $CANON $ROOT/smith_agent.py" >&2
  exit 3
fi
# smith_rollout.py:161 猴补了 sa._query,所以改在 _query 里的东西要在那边重复一遍。
for k in skip_special_tokens strip_turn_delims _STOP_STRINGS; do
  grep -q "$k" $ROOT/smith_rollout.py || { echo "FATAL: smith_rollout.py 缺 $k(见 §55)" >&2; exit 3; }
done

CANDIDATES=(${FLEET_CARDS:-5 3 1 2}); PORTS=(18005 18006 18007 18008)
BUDGET=${FLEET_BUDGET:-30000}; SEQS=${FLEET_SEQS:-8}; WORKERS=${FLEET_WORKERS:-6}
TO=${FLEET_TIMEOUT:-600}            # 人类 09-21:单题 600s 不提,超时要从训练侧解决
CTX=${FLEET_CTX:-51200}             # MiniCPM5-2B 的 max_position_embeddings 是 131072,51200 够 40 轮
OUTTOK=${FLEET_MAX_TOKENS:-4096}    # prompt 上限 = CTX - OUTTOK = 47104
TURNS=${SMITH_MAX_TURNS:-40}        # 09-05 那次用了 10000,69% 的 episode 死于上下文溢出
TEMP=${SMITH_TEMPERATURE:-0.6}
LIMIT=${AGL_SWEEP_LIMIT:-}
# 10-06 换模型评测(人类令:同框架评 Qwen3-8B)。默认值与此前逐字相同,MiniCPM 血统的读数不受影响。
#   FLEET_SERVED_NAME   vLLM 对外模型名 = agent 请求里的 model 字段
#   FLEET_CHAT_TEMPLATE 模板文件;"none" = 用模型 tokenizer_config 自带的模板(Qwen3 走这条)
#   FLEET_ROPE_SCALING  传给 --rope-scaling 的 JSON(Qwen3-8B 原生 40960 < CTX 51200,要 YaRN)
NAME=${FLEET_SERVED_NAME:-MiniCPM5-2B}
TEMPLATE=${FLEET_CHAT_TEMPLATE:-/workspace/models/MiniCPM5-2B/chat_template.jinja}
ROPE=${FLEET_ROPE_SCALING:-}
TPL_ARGS=(); [ "$TEMPLATE" != none ] && TPL_ARGS=(--chat-template "$TEMPLATE")
ROPE_ARGS=(); [ -n "$ROPE" ] && ROPE_ARGS=(--rope-scaling "$ROPE")

cmd=$1; shift
if [ "$cmd" = start ]; then
  MODEL=$1; OUT=$2; TAG=$3
  mkdir -p $ROOT/$OUT
  CARDS=()
  for g in "${CANDIDATES[@]}"; do
    read used total <<< $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $g | tr -d ',')
    free=$((total - used))
    if [ $free -ge $((BUDGET + 2000)) ]; then CARDS+=($g); else echo "skip gpu$g free=${free}MiB < $((BUDGET+2000))"; fi
  done
  N=${#CARDS[@]}
  [ $N -ge 1 ] || { echo "FLEET_FAIL 没有一张卡有 ${BUDGET}MiB 空闲"; exit 2; }
  echo "${CARDS[*]}" > smith_${TAG}.cards

  launch_vllm() {  # $1=k
    local k=$1 g=${CARDS[$1]} port=${PORTS[$1]} used total util
    read used total <<< $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $g | tr -d ',')
    util=$(python3 -c "print(round(($used + $BUDGET)/$total, 3))")
    echo "gpu$g port=$port used=${used}MiB budget=${BUDGET}MiB util=$util ctx=$CTX name=$NAME template=$TEMPLATE rope=${ROPE:-none}"
    CUDA_VISIBLE_DEVICES=$g VLLM_USE_V1=1 nohup $PY -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 --port $port --model $MODEL --served-model-name $NAME --dtype bfloat16 \
      --max-model-len $CTX --max-num-seqs $SEQS --gpu-memory-utilization $util \
      "${TPL_ARGS[@]}" "${ROPE_ARGS[@]}" \
      --override-generation-config '{"temperature": 0.6, "top_p": 0.95, "top_k": 20}' \
      > vllm_smith_${TAG}_gpu$g.log 2>&1 &
    echo $! > vllm_smith_${TAG}_k$k.pid
  }
  wait_vllm() {  # $1=k
    local k=$1 port=${PORTS[$1]} i
    for i in $(seq 1 120); do
      curl -s -m 3 --noproxy '*' http://127.0.0.1:$port/v1/models >/dev/null 2>&1 && return 0
      kill -0 $(cat vllm_smith_${TAG}_k$k.pid) 2>/dev/null || return 1
      sleep 5
    done
    return 1
  }
  for k in $(seq 0 $((N-1))); do launch_vllm $k; done
  OK=()
  for k in $(seq 0 $((N-1))); do
    up=0
    for try in 1 2 3 4; do
      if wait_vllm $k; then up=1; break; fi
      echo "retry gpu${CARDS[$k]} (try $try failed)"; kill $(cat vllm_smith_${TAG}_k$k.pid) 2>/dev/null; sleep 20
      [ $try -lt 4 ] && launch_vllm $k
    done
    if [ $up = 1 ]; then OK+=($k); else echo "drop gpu${CARDS[$k]}"; fi
  done
  M=${#OK[@]}
  [ $M -ge 1 ] || { echo "FLEET_FAIL 没有一个 vllm 起来"; exit 2; }

  s=0
  for k in "${OK[@]}"; do
    AGL_SWEEP_SPLIT=val AGL_SWEEP_REPO=ALL AGL_SWEEP_ROOT=$ROOT/$OUT \
    AGL_SWEEP_SHARD_COUNT=$M AGL_SWEEP_SHARD_INDEX=$s AGL_SWEEP_WORKERS=$WORKERS AGL_SWEEP_RESUME=1 \
    AGL_SWEEP_LIMIT=$LIMIT \
    AGL_SAMPLE_TIMEOUT=$TO SMITH_EVAL_TIMEOUT=$TO \
    SMITH_MAX_TURNS=$TURNS SMITH_TEMPERATURE=$TEMP SMITH_OBS_CHAR_CAP=${SMITH_OBS_CHAR_CAP:-6000} \
    AGL_MAX_TOKENS=$OUTTOK AGL_MAX_MODEL_LEN=$CTX AGL_MODEL=$NAME \
    AGL_VLLM_URL=http://127.0.0.1:${PORTS[$k]} \
    nohup python3 run_smith_sweep.py > smith_${TAG}_shard$s.log 2>&1 &
    echo $! > smith_${TAG}_k$k.pid
    s=$((s+1))
  done
  echo "FLEET_UP $TAG harness=smith cards=${CARDS[*]} up_idx=${OK[*]} shards=$M workers=$WORKERS turns=$TURNS temp=$TEMP ctx=$CTX"

elif [ "$cmd" = wait ]; then
  TAG=$1; OUT=$2
  while true; do
    alive=0; for f in smith_${TAG}_k*.pid; do kill -0 $(cat $f 2>/dev/null) 2>/dev/null && alive=$((alive+1)); done
    done_=$(ls $ROOT/$OUT/*/result.json 2>/dev/null | wc -l)
    echo "[$(date +%T)] smith $TAG done=$done_/474 shards_alive=$alive"
    [ $alive -eq 0 ] && break
    sleep 120
  done

elif [ "$cmd" = stop ]; then
  TAG=$1
  for f in smith_${TAG}_k*.pid vllm_smith_${TAG}_k*.pid; do
    p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill $p 2>/dev/null
  done
  sleep 15
  for f in vllm_smith_${TAG}_k*.pid; do p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && kill -9 $p; done
  # 清掉本 TAG 残留的 rollout 容器(只动 agl-sm-* 前缀,别人的容器一律不碰)
  docker ps -a --format '{{.Names}}' 2>/dev/null | grep '^agl-sm-' | xargs -r docker rm -f >/dev/null 2>&1
  echo "FLEET_DOWN $TAG"
fi
