#!/bin/bash
# Teacher batch 5 (2026-10-06, user: 「你先继续训练,蒸馏同步进行,增大并发度,使用AGL的简易harness」+
# 「deepseekv4flash做错的题就没必要参与训练了」).
#   * teacher = deepseek-v4-flash via the gateway (teacher_api.env), driven by the OFFICIAL smith harness
#     (smith_agent.py / smith_rollout.py: plain text, one bash block per turn, no tools, no compaction) so the
#     trajectories are byte-for-byte in the student's prompt format (same SYSTEM_PROMPT / INSTANCE_PROMPT /
#     observation template as the s5 training + the 474 val probes).
#   * 1000 fresh train problems: teacher_batch5_ids_1000.{txt,json} (seed 20261006; disjoint from batches 1-4
#     and from val_dataset_filtered). All 98 candidate images are already in docker, no pulls.
#   * proxy = teacher_proxy_smith.py on :18009 (containers' iptables only allow 172.17.0.1:18000-18009);
#     it strips vLLM-only params, retries 429/5xx itself (the stdlib client has no retry and would burn the
#     turn with an empty assistant message), logs every request to teacher_traj_smith/proxy_18009/
#     (events.jsonl + <md5(first user msg)>.json = last full request of each rollout).
#   * container prefix agl-st: smith_fleet.sh (the s5 probe) does `docker rm -f` on every ^agl-sm- container.
#   * runs in parallel with the s5 trainer (cards 1/2/3/5) and its probes; CPU-only on the host.
#   * SFT data = teacher-RESOLVED rollouts only (rejection sampling), assembled by assemble_smith_teacher.py.
# usage: bash teacher_smith_b5.sh proxy|start|stop|status
set -u
D=/workspace/agl-checkpoints/swe_smith_smoke
PORT=${PORT:-18009}
ROOT=${ROOT:-$D/teacher_train_smith_b5}
IDS=${IDS:-$D/teacher_batch5_ids_1000.txt}
SHARDS=${SHARDS:-10}
WORKERS=${WORKERS:-10}
PREFIX=agl-st
# teacher_api.env exports TEACHER_MODEL="gpt-5.5" (another job's default) -> must be forced AFTER sourcing it.
# batches 1-4 ran deepseek-v4-flash (proxies 18034/18035 env); same id here.
TMODEL=${TMODEL:-deepseek-v4-flash}
PLOG=$D/teacher_traj_smith/proxy_$PORT
PFX=$D/teacher_smith_b5
cmd=${1:-status}

proxy_up() { curl -s --noproxy '*' -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; }

case "$cmd" in
  proxy)
    if proxy_up; then echo "proxy :$PORT already up"; exit 0; fi
    [ -f $D/teacher_api.env ] || { echo "missing teacher_api.env"; exit 1; }
    mkdir -p $PLOG
    # usage.prompt_tokens rewritten to MiniCPM counts (status.json max_prompt_tokens then means student tokens);
    # only if transformers is importable in this python.
    TOKDIR=""; python3 -c 'import transformers' 2>/dev/null && TOKDIR=/workspace/models/MiniCPM5-2B-sft-v3-ep3
    # NOTE: `a && b && cmd &` backgrounds the whole AND-list as one subshell, so `$!` would be the wrapper,
    # not python (first launch 16:4x: killing the pid file left the real server alive). Hence `exec`.
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
    n=$(ls $ROOT/*/result.json 2>/dev/null | wc -l)
    res=$(cat $ROOT/*/result.json 2>/dev/null | /usr/bin/grep -o '"resolved": *true' | wc -l)
    cost=$(EV=$PLOG/events.jsonl python3 - <<'EOF'
import json, os
first=last=None; err=0; n=0; retry=0; lat=[]
try:
    for l in open(os.environ['EV']):
        try: r=json.loads(l)
        except Exception: continue
        if r.get('event')=='retry': retry+=1; continue
        n+=1
        if r.get('key_spend_usd'):
            first = r['key_spend_usd'] if first is None else first; last=r['key_spend_usd']
        if r.get('event')=='upstream_error' or (r.get('status') and r['status']!=200): err+=1
        if r.get('latency_s'): lat.append(r['latency_s'])
except FileNotFoundError:
    pass
lat.sort(); p50 = lat[len(lat)//2] if lat else None
print(f"requests={n} retries={retry} non200_or_err={err} lat_p50={p50}s key_spend_usd={last} spent_since_proxy_start={None if first is None else round(last-first,2)}")
EOF
)
    echo "$(date +%H:%M) proxy=$(proxy_up && echo up || echo DOWN) sweeps_alive=$alive/$SHARDS done=$n/1000 resolved=$res containers=$(docker ps --format '{{.Names}}' | /usr/bin/grep -c "^$PREFIX-") load=$(cut -d' ' -f1 /proc/loadavg) $cost disk_free=$(df -h /data | awk 'NR==2{print $4}')"
    ;;
  *) echo "usage: $0 proxy|start|stop|status"; exit 1;;
esac
