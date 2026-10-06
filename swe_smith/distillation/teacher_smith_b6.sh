#!/bin/bash
# Teacher batch 6 (2026-10-06 17:2x, user: 「趁这个时间跑完所有训练集的教师轨迹」): every train problem that has no
# official-harness teacher trajectory yet = train_dataset_mixed (6338) minus batch 5 (1000). Batches 1-4 were
# collected with the opencode tool-calling harness (different prompt/observation format) so they are NOT usable
# for the smith-harness SFT and are re-run here. Same proxy (:18009, with normalize_teacher_message), same
# harness/env as teacher_smith_b5.sh; only ROOT/IDS/PFX/SHARDS differ (PFX is now a parameter so the b5 logs/pids
# stay untouched). Problems whose docker image is not local are listed in teacher_batch6_ids_missing_image.txt
# and skipped (pulling 3.2GB images would breach the 50G /data gate); run them later with IDS=<that file>.
# usage: bash teacher_smith_b6.sh proxy|start|stop|status        (env: SHARDS WORKERS IDS ROOT PFX)
set -u
D=/workspace/agl-checkpoints/swe_smith_smoke
PORT=${PORT:-18009}
ROOT=${ROOT:-$D/teacher_train_smith_b6}
IDS=${IDS:-$D/teacher_batch6_ids_rest.txt}
SHARDS=${SHARDS:-15}
WORKERS=${WORKERS:-10}
PREFIX=agl-st
TMODEL=${TMODEL:-deepseek-v4-flash}
PLOG=$D/teacher_traj_smith/proxy_$PORT
PFX=${PFX:-$D/teacher_smith_b6}
cmd=${1:-status}

proxy_up() { curl -s --noproxy '*' -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; }

case "$cmd" in
  proxy)
    if proxy_up; then echo "proxy :$PORT already up"; exit 0; fi
    [ -f $D/teacher_api.env ] || { echo "missing teacher_api.env"; exit 1; }
    mkdir -p $PLOG
    TOKDIR=""; python3 -c 'import transformers' 2>/dev/null && TOKDIR=/workspace/models/MiniCPM5-2B-sft-v3-ep3
    ( cd $D; set -a; . $D/teacher_api.env; set +a; \
      exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      MINICPM_TOKENIZER=$TOKDIR TEACHER_MODEL=$TMODEL \
      TEACHER_MAX_TURNS=0 TEACHER_UPSTREAM_TIMEOUT=900 TEACHER_RETRY_MAX=6 \
      nohup python3 $D/teacher_proxy_smith.py $PORT $PLOG >> $PLOG/proxy.log 2>&1 ) &
    echo $! > $PFX.proxy.pid
    for i in 1 2 3 4 5 6 7 8 9 10; do sleep 1; proxy_up && { echo "proxy :$PORT up (pid $(cat $PFX.proxy.pid))"; exit 0; }; done
    echo "proxy :$PORT did not come up; see $PLOG/proxy.log"; exit 1
    ;;
  start)
    bash "$0" proxy || exit 1
    [ -f "$IDS" ] || { echo "missing ids file $IDS"; exit 1; }
    mkdir -p $ROOT
    [ -f $PFX.launch_ts ] || date +%s > $PFX.launch_ts
    for k in $(seq 0 $((SHARDS-1))); do
      if [ -f ${PFX}_k$k.pid ] && kill -0 $(cat ${PFX}_k$k.pid) 2>/dev/null; then
        echo "shard $k already running"; continue
      fi
      ( cd $D; exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        AGL_SWEEP_CONTAINER_PREFIX=$PREFIX AGL_SWEEP_SPLIT=train AGL_SWEEP_REPO=ALL AGL_SWEEP_ROOT=$ROOT \
        AGL_SWEEP_ONLY=$IDS AGL_SWEEP_SHARD_COUNT=$SHARDS AGL_SWEEP_SHARD_INDEX=$k \
        AGL_SWEEP_WORKERS=$WORKERS AGL_SWEEP_RESUME=1 \
        AGL_SAMPLE_TIMEOUT=2400 SMITH_EVAL_TIMEOUT=600 \
        SMITH_MAX_TURNS=40 AGL_MAX_TOKENS=16384 AGL_MAX_MODEL_LEN=51200 AGL_MODEL=deepseek-v4-flash \
        SMITH_TEMPERATURE=0.6 SMITH_OBS_CHAR_CAP=6000 SMITH_CMD_TIMEOUT=120 SMITH_MAX_FORMAT_ERRORS=3 \
        AGL_VLLM_URL=http://127.0.0.1:$PORT \
        nohup python3 run_smith_sweep.py >> ${PFX}_k$k.log 2>&1 ) &
      echo $! > ${PFX}_k$k.pid
      echo "shard $k started (pid $! log ${PFX}_k$k.log)"
    done
    ;;
  stop)
    for p in $(pgrep -f run_smith_sweep.py); do
      if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | /usr/bin/grep -q "^AGL_SWEEP_ROOT=$ROOT$"; then kill -9 $p && echo "killed sweep pid $p"; fi
    done
    sleep 2
    for c in $(docker ps -a --format '{{.Names}}' | /usr/bin/grep -E "^$PREFIX-[0-9]+-"); do
      case "$(docker inspect -f '{{range .Mounts}}{{.Source}} {{end}}' $c 2>/dev/null)" in
        *"$ROOT"*) docker rm -f $c >/dev/null 2>&1 && echo "removed $c";;
      esac
    done
    ;;
  status)
    alive=0
    for p in $(pgrep -f run_smith_sweep.py); do
      tr '\0' '\n' < /proc/$p/environ 2>/dev/null | /usr/bin/grep -q "^AGL_SWEEP_ROOT=$ROOT$" && alive=$((alive+1))
    done
    total=$(/usr/bin/grep -c . $IDS 2>/dev/null)
    n=$(ls $ROOT/*/result.json 2>/dev/null | wc -l)
    res=$(cat $ROOT/*/result.json 2>/dev/null | /usr/bin/grep -o '"resolved": *true' | wc -l)
    cost=$(EV=$PLOG/events.jsonl T0=$(cat $PFX.launch_ts 2>/dev/null || echo 0) python3 - <<'EOF'
import json, os
t0=float(os.environ['T0']); first=last=None; err=0; n=0; retry=0; lat=[]
try:
    for l in open(os.environ['EV']):
        try: r=json.loads(l)
        except Exception: continue
        if (r.get('ts') or 0) < t0: continue
        if r.get('event')=='retry': retry+=1; continue
        n+=1
        if r.get('key_spend_usd'):
            first = r['key_spend_usd'] if first is None else first; last=r['key_spend_usd']
        if r.get('event')=='upstream_error' or (r.get('status') and r['status']!=200): err+=1
        if r.get('latency_s'): lat.append(r['latency_s'])
except FileNotFoundError:
    pass
lat.sort(); p50 = lat[len(lat)//2] if lat else None
print(f"requests={n} retries={retry} non200_or_err={err} lat_p50={p50}s key_spend_usd={last} spent_this_batch={None if first is None else round(last-first,2)}")
EOF
)
    echo "$(date +%H:%M) proxy=$(proxy_up && echo up || echo DOWN) sweeps_alive=$alive/$SHARDS done=$n/$total resolved=$res containers=$(docker ps --format '{{.Names}}' | /usr/bin/grep -c "^$PREFIX-") load=$(cut -d' ' -f1 /proc/loadavg) $cost disk_free=$(df -h /data | awk 'NR==2{print $4}')"
    ;;
  *) echo "usage: $0 proxy|start|stop|status"; exit 1;;
esac
