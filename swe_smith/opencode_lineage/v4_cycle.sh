#!/bin/bash
# v4 train/probe cycle (human order 2026-09-17): train 20 steps, stop, merge, evaluate the
# first 100 val problems, compare with the base model on the same 100, resume unless collapsed.
#   Base on these 100: 26 resolved (val_minicpm5_2b_50k); step48-v3: 20; step76-v3 (collapsed): 11.
#   COLLAPSE rule: resolved < PROBE_MIN (default 18 = base-8) -> do NOT resume, write HALT file.
# 09-17 23:00 human: if 4 cards (1/2/3/5) are not free within 1h, train on 2 cards (3/5) with the
# batch halved (GPU_MODE=2 in restart_v4_trainer.sh); go back to 4 cards when memory is available.
# FSDP ckpts are world-size-specific, so every mode switch goes through a merged HF checkpoint
# (fresh optimizer). Step numbers reported here are TOTAL steps = BASE (steps done in earlier
# stints) + latest ckpt of the current stint; state kept in v4_cycle.state.
# Runs under nohup; log in v4_cycle.log. Trainer runs in tmux pane minicpm50k:2.0.
set -u
ROOT=/workspace/agl-checkpoints/swe_smith_smoke
CK4=/workspace/agl-checkpoints/swe_smith_opencode_minicpm50k_v4
CK2=/workspace/agl-checkpoints/swe_smith_opencode_minicpm50k_v4_2gpu
AGL=/workspace/projects/agent-lightning
PY=$AGL/.venv/bin/python
STATE=$ROOT/v4_cycle.state
PROBE_EVERY=${PROBE_EVERY:-20}
PROBE_MIN=${PROBE_MIN:-18}
MIN_FREE_MB=${MIN_FREE_MB:-40000}     # per-card need at peak ~25-35G (4-card) / ~40G (2-card)
MIN_FREE_MB_2=${MIN_FREE_MB_2:-45000}
WAIT4_SEC=${WAIT4_SEC:-3600}          # how long to wait for 4 cards before falling back to 2
MAX_RETRIES=${MAX_RETRIES:-6}
PANE=minicpm50k:2.0
INIT_MODEL=/workspace/models/MiniCPM5-2B-step48-v3
cd $ROOT
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---- state: MODE (4|2), BASE (total steps before current stint), MODEL (HF weights current stint started from)
MODE=4; BASE=0; MODEL=$INIT_MODEL; LAST_PROBE=0
[ -f $STATE ] && source $STATE
save_state() { printf 'MODE=%s\nBASE=%s\nMODEL=%s\nLAST_PROBE=%s\n' "$MODE" "$BASE" "$MODEL" "$LAST_PROBE" > $STATE; }
# NEXT_TARGET=<total step> (env) forces the next probe at that step (human 09-18 14:55: probe at 18, not 20);
# afterwards probes continue every PROBE_EVERY steps counted from the last probe.
ck_dir() { [ "$1" = 2 ] && echo $CK2 || echo $CK4; }
stint_step() { cat $(ck_dir $MODE)/latest_checkpointed_iteration.txt 2>/dev/null || echo 0; }
total_step() { echo $(( BASE + $(stint_step) )); }
trainer_pid() { pgrep -n -f "train_opencode_agent.py --agl-base-url" ; }

min_free() {  # min free MiB over the given cards
  local worst=999999
  for g in "$@"; do
    read used total <<< $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $g | tr -d ',')
    local free=$((total - used)); [ $free -lt $worst ] && worst=$free
  done
  echo $worst
}
four_free() { [ $(min_free 1 2 3 5) -ge $MIN_FREE_MB ]; }
two_free()  { [ $(min_free 3 5) -ge $MIN_FREE_MB_2 ]; }

# Decide the mode for the next stint: prefer 4 cards; after WAIT4_SEC fall back to 2 cards.
choose_mode() {
  local waited=0
  while true; do
    if four_free; then MODE=4; return 0; fi
    if [ $waited -ge $WAIT4_SEC ] && two_free; then MODE=2; log "4 cards not free after ${waited}s -> 2-card mode (cards 3,5)"; return 0; fi
    [ $((waited % 600)) -eq 0 ] && log "waiting for GPU memory: 4-card min free $(min_free 1 2 3 5)MiB, 2-card min free $(min_free 3 5)MiB"
    sleep 30; waited=$((waited + 30))
  done
}

start_trainer() {   # uses MODE, MODEL; resume auto if the stint dir has a ckpt, else fresh from MODEL
  local ck=$(ck_dir $MODE) mode
  mkdir -p $ck/trajectories
  if trainer_pid >/dev/null; then log "trainer already running (pid $(trainer_pid)), attaching"; return 0; fi
  if [ "$(stint_step)" = 0 ]; then mode=disable; else mode=auto; fi
  log "starting trainer GPU_MODE=$MODE resume_mode=$mode model=$MODEL (total step $(total_step)) in pane $PANE"
  tmux send-keys -t $PANE "cd $AGL && GPU_MODE=$MODE MODEL=$MODEL RESUME_MODE=$mode bash examples/swe_smith/restart_v4_trainer.sh" C-m
  for i in $(seq 1 60); do sleep 5; trainer_pid >/dev/null && return 0; done
  log "TRAINER_START_FAIL"; return 1
}

stop_trainer() {
  local pid=$(trainer_pid); [ -z "$pid" ] && return 0
  log "SIGINT trainer $pid"
  kill -INT $pid
  for i in $(seq 1 90); do kill -0 $pid 2>/dev/null || break; sleep 10; done
  if kill -0 $pid 2>/dev/null; then log "trainer still alive after 15 min, SIGTERM"; kill $pid; sleep 30; fi
  for i in $(seq 1 30); do pgrep -f "_AglTaskRunner|WorkerDict|vLLMHttpServer" >/dev/null || break; sleep 10; done
  sleep 20
}

# Wait until total step reaches $1. Returns 0 reached, 1 trainer died, 2 = switch to 4 cards requested
# (only in 2-card mode, right after a new ckpt landed, with 4 cards free on 3 consecutive polls).
wait_for_step() {
  local target=$1 last=$(stint_step) stable=0
  while true; do
    local s=$(stint_step) t=$(total_step)
    [ "$t" -ge "$target" ] && return 0
    trainer_pid >/dev/null || { log "TRAINER_DIED at total step $t (target $target)"; return 1; }
    if [ "$MODE" = 2 ]; then
      if four_free; then stable=$((stable + 1)); else stable=0; fi
      if [ $stable -ge 3 ] && [ "$s" != "$last" ]; then log "4 cards free again (3 polls) and ckpt just landed at total $t -> switch back"; return 2; fi
    fi
    last=$s
    sleep 120
  done
}

merge_current() {   # merge latest ckpt of the current stint -> HF dir for total step; echo path
  local t=$(total_step) s=$(stint_step) ck=$(ck_dir $MODE)
  local out=/workspace/models/MiniCPM5-2B-v4-step$t
  if [ ! -f $out/config.json ]; then
    log "merging $ck/global_step_$s -> $out"
    (cd $AGL && CUDA_VISIBLE_DEVICES="" $PY -m verl.model_merger merge --backend fsdp \
      --local_dir $ck/global_step_$s/actor --target_dir $out 2>&1 | tail -2)
    [ -f $out/config.json ] || { log "MERGE_FAIL total $t"; return 2; }
  fi
  echo $out
}

# Close the current stint: merge -> new BASE/MODEL; retire the stint's ckpt dir (weights live in HF now).
close_stint() {
  local hf; hf=$(merge_current) || return 2
  local t=$(total_step) ck=$(ck_dir $MODE)
  BASE=$t; MODEL=$(echo "$hf" | tail -1); save_state
  rm -rf $ck.prev; mv $ck $ck.prev
  log "stint closed at total $t; next stint starts from $MODEL"
}

probe() {
  local step=$1 model=$2
  local out=val_v4_step${step}_p100 tag=v4s$step
  bash $ROOT/v4_probe_fleet.sh start $model $out $tag || { bash $ROOT/v4_probe_fleet.sh stop $tag; log "PROBE_FLEET_FAIL step $step"; return 2; }
  bash $ROOT/v4_probe_fleet.sh wait $tag $out
  bash $ROOT/v4_probe_fleet.sh stop $tag
  local res=$(python3 - <<EOF
import json,glob
n=r=0
for p in glob.glob("$ROOT/$out/*/result.json"):
    d=json.load(open(p)); ev=d.get("eval") or d; n+=1; r+=bool(ev.get("resolved"))
print(f"{r} {n}")
EOF
)
  set -- $res
  log "PROBE_RESULT step=$step resolved=$1/$2 (base 26/100, step48 20/100, halt if < $PROBE_MIN)"
  (cd $ROOT && python3 compare_runs.py val_minicpm5_2b_50k $out 2>&1 | tail -3)
  echo "$step $1 $2 $(date '+%m-%d %H:%M')" >> $ROOT/v4_probe_results.txt
  if [ "$2" -lt 90 ]; then log "PROBE_INCOMPLETE ($2/100) -> halt for human"; return 3; fi
  [ "$1" -lt "$PROBE_MIN" ] && return 1
  return 0
}

halt() { log "HALT: $1"; echo "$1" > $ROOT/v4_HALT; exit 1; }

# ---------------- main loop ----------------
retries=0
save_state
while true; do
  # a stint dir with a ckpt means we resume it in its own mode; otherwise pick a mode by free memory
  if ! trainer_pid >/dev/null; then   # a running trainer occupies the cards; only pick a mode when we must start one
    if [ "$(stint_step)" = 0 ]; then choose_mode; save_state; fi
    if [ "$MODE" = 4 ] && ! four_free; then choose_mode; save_state; fi
    if [ "$MODE" = 2 ] && [ "$(stint_step)" != 0 ] && ! two_free; then log "waiting for cards 3,5"; sleep 60; continue; fi
  fi
  t=$(total_step)
  if [ -n "${NEXT_TARGET:-}" ] && [ "$NEXT_TARGET" -gt "$t" ]; then
    target=$NEXT_TARGET
  else
    target=$(( LAST_PROBE + PROBE_EVERY )); while [ $target -le $t ]; do target=$((target + PROBE_EVERY)); done
  fi
  start_trainer || halt "trainer start failed at total $(total_step)"
  log "training to total step $target (now $t, mode $MODE)"
  wait_for_step $target; rc=$?
  if [ $rc = 1 ]; then
    retries=$((retries + 1))
    log "trainer died (retry $retries/$MAX_RETRIES); last errors:"; grep -a "Error\|error:" $ROOT/trainer_v4.log | tail -3 | cut -c1-200
    [ $retries -gt $MAX_RETRIES ] && halt "trainer died $retries times, last at total $(total_step)"
    stop_trainer; sleep 60; continue
  fi
  retries=0
  stop_trainer
  if [ $rc = 2 ]; then close_stint || halt "merge failed on mode switch"; MODE=4; save_state; continue; fi
  # reached a probe boundary
  hf=$(merge_current) || halt "merge failed at total $(total_step)"
  hf=$(echo "$hf" | tail -1)
  probe $target $hf; prc=$?
  case $prc in
    0) log "probe ok"; LAST_PROBE=$target; NEXT_TARGET=""; save_state;;
    1) halt "COLLAPSE at total $target: $(tail -1 $ROOT/v4_probe_results.txt)";;
    *) halt "probe error rc=$prc at total $target";;
  esac
  # after a probe, re-pick the mode: 2-card stint switches to 4 cards if they are free now
  if [ "$MODE" = 2 ] && four_free; then close_stint || halt "merge failed on mode switch"; MODE=4; save_state; fi
done
