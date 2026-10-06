#!/bin/bash
# 100-problem probe fleet for v4 (2026-09-17): vLLM on whichever of cards 1/2/3/5 currently has
# room (≥ budget+2G free; at least 2 cards), one truncating proxy + one sweep shard per card,
# over the first 100 val problems (val_first100_ids.json).
#   v4_probe_fleet.sh start <MODEL_DIR> <OUT_DIR_NAME> <TAG>
#   v4_probe_fleet.sh wait  <TAG> <OUT_DIR_NAME>
#   v4_probe_fleet.sh stop  <TAG>
set -u
ROOT=/workspace/agl-checkpoints/swe_smith_smoke
PY=/workspace/projects/agent-lightning/.venv/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/workspace/mps_bypass_empty
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"
cd $ROOT
CANDIDATES=(${FLEET_CARDS:-1 2 3 5}); PORTS=(18013 18014 18015 18016); LP=(18022 18023 18024 18025)   # FLEET_CARDS="5 2" limits the fleet (2026-09-21 morning: ≤2 cards)
BUDGET=30000; SEQS=8; WORKERS=${FLEET_WORKERS:-7}; TO=${FLEET_TIMEOUT:-600}   # 2026-09-22: FLEET_WORKERS=3 + step cap (FLEET_TIMEOUT only a hang guard)
cmd=$1; shift
if [ "$cmd" = start ]; then
  MODEL=$1; OUT=$2; TAG=$3
  mkdir -p $ROOT/$OUT
  # pick cards with room
  CARDS=()
  for g in "${CANDIDATES[@]}"; do
    read used total <<< $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $g | tr -d ',')
    free=$((total - used))
    if [ $free -ge $((BUDGET + 2000)) ]; then CARDS+=($g); else echo "skip gpu$g free=${free}MiB"; fi
  done
  N=${#CARDS[@]}
  [ $N -ge 2 ] || { echo "FLEET_FAIL only $N cards with ${BUDGET}MiB free"; exit 2; }
  echo "${CARDS[*]}" > fleet_${TAG}.cards
  # 2026-09-21: per-card retry (vLLM 0.8.5 asserts when a neighbour frees memory during profiling) and
  # drop cards that still fail; shards are numbered over the cards that came up.
  launch_vllm() {  # $1=k
    local k=$1 g=${CARDS[$1]} port=${PORTS[$1]} used total util
    read used total <<< $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $g | tr -d ',')
    util=$(python3 -c "print(round(($used + $BUDGET)/$total, 3))")
    echo "gpu$g port=$port used=$used budget=$BUDGET util=$util"
    CUDA_VISIBLE_DEVICES=$g VLLM_USE_V1=1 nohup $PY -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 --port $port --model $MODEL --served-model-name MiniCPM5-2B --dtype bfloat16 \
      --max-model-len 51200 --max-num-seqs $SEQS --gpu-memory-utilization $util \
      --chat-template /workspace/models/MiniCPM5-2B/chat_template.jinja \
      --override-generation-config '{"temperature": 0.6, "top_p": 0.95, "top_k": 20}' \
      --enable-auto-tool-choice --tool-parser-plugin $ROOT/minicpm5_parser_plugin_085.py --tool-call-parser minicpm5 \
      > vllm_${TAG}_gpu$g.log 2>&1 &
    echo $! > vllm_${TAG}_k$k.pid
  }
  wait_vllm() {  # $1=k
    local k=$1 port=${PORTS[$1]} i
    for i in $(seq 1 120); do
      curl -s -m 3 http://127.0.0.1:$port/v1/models >/dev/null 2>&1 && return 0
      kill -0 $(cat vllm_${TAG}_k$k.pid) 2>/dev/null || return 1
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
      echo "retry gpu${CARDS[$k]} (try $try failed)"; kill $(cat vllm_${TAG}_k$k.pid) 2>/dev/null; sleep 20
      [ $try -lt 4 ] && launch_vllm $k
    done
    if [ $up = 1 ]; then OK+=($k); else echo "drop gpu${CARDS[$k]}"; fi
  done
  M=${#OK[@]}
  [ $M -ge 2 ] || { echo "FLEET_FAIL only $M vllm up"; exit 2; }
  for k in "${OK[@]}"; do
    nohup python3 tool_trunc_proxy.py ${LP[$k]} ${PORTS[$k]} > trunc_proxy_${LP[$k]}.log 2>&1 &
    echo $! > trunc_proxy_${TAG}_k$k.pid
  done
  sleep 2
  s=0
  for k in "${OK[@]}"; do
    AGL_SWEEP_SPLIT=val AGL_SWEEP_REPO=ALL AGL_SWEEP_ROOT=$ROOT/$OUT \
    AGL_SWEEP_SHARD_COUNT=$M AGL_SWEEP_SHARD_INDEX=$s AGL_SWEEP_WORKERS=$WORKERS AGL_SWEEP_RESUME=1 \
    AGL_SAMPLE_TIMEOUT=$TO SMITH_EVAL_TIMEOUT=$TO \
    AGL_VLLM_URL=http://127.0.0.1:${LP[$k]} AGL_OPENCODE_JSON=$ROOT/opencode.minicpm40k.${LP[$k]}.json \
    AGL_SWEEP_MODEL=agl/MiniCPM5-2B \
    nohup python3 run_repo_sweep.py > val_${TAG}_shard$s.log 2>&1 &
    echo $! > val_${TAG}_k$k.pid
    s=$((s+1))
  done
  echo "FLEET_UP $TAG cards=${CARDS[*]} up_idx=${OK[*]}"
elif [ "$cmd" = wait ]; then
  TAG=$1; OUT=$2
  while true; do
    alive=0; for f in val_${TAG}_k*.pid; do kill -0 $(cat $f 2>/dev/null) 2>/dev/null && alive=$((alive+1)); done
    done_=$(ls $ROOT/$OUT/*/result.json 2>/dev/null | wc -l)
    echo "[$(date +%T)] probe $TAG done=$done_/474 shards_alive=$alive"
    [ $alive -eq 0 ] && break
    sleep 120
  done
elif [ "$cmd" = stop ]; then
  TAG=$1
  for f in val_${TAG}_k*.pid trunc_proxy_${TAG}_k*.pid vllm_${TAG}_k*.pid; do
    p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill $p 2>/dev/null
  done
  sleep 15
  for f in vllm_${TAG}_k*.pid; do p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && kill -9 $p; done
  echo "FLEET_DOWN $TAG"
fi
