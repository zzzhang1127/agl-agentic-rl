#!/bin/bash
# s2 训练 / 全量探针循环驱动(人类 10-04 配方)。
#
#   起点   MiniCPM5-2B-sft-v3-ep3(474 全量 136,当前最好)
#   lr     5e-6
#   探针    每 PROBE_EVERY 步停训,跑**全量 474**,和 SFT ep3 做配对 McNemar
#   停训    探针显著低于基线(p<0.05)就停 —— 见 s2_verdict.py
#
# 为什么探针要停训才能跑:474 评测要四张卡各 40G 起 vLLM,和训练抢同样的卡 1/2/3/5。
# 一次探针约 45 分钟,PROBE_EVERY=10 时开销 ≈9%。
#
#   bash s2_cycle.sh start           # 首次起训
#   bash s2_cycle.sh wait   <STEP>   # 阻塞到 global_step_<STEP>/data.pt 落盘
#   bash s2_cycle.sh pause
#   bash s2_cycle.sh probe  <STEP>   # merge + 474 + 判决(退出码 1 = STOP)
#   bash s2_cycle.sh resume
#
# 纪律:不用 `ray stop --force`;只动 agl-rollout-*;不碰邻居进程;/data 红线 50G。
set -u
ROOT=/workspace/projects/agent-lightning
EX=$ROOT/examples/swe_smith
CK=/workspace/agl-checkpoints/smith_rl_s2
EVALROOT=/workspace/agl-checkpoints/swe_smith_smoke
PROBE_EVERY=${PROBE_EVERY:-10}

# 训练 env:沿用 s1 已验证的那一组,只改 lr(脚本默认已是 5e-6)和起点(默认已是 SFT ep3)。
# TEST_FREQ=0 关掉 48 题内置探针 —— 它被全量 474 取代了,再跑是白烧 gen 时间。
TRAIN_ENV=(TRAIN_BATCH=16 PPO_MINI=16 ROLLOUT_N=4 VLLM_SEQS=16 EAGER=False
           TEST_FREQ=0 SAVE_FREQ=5)

disk_guard() {
  local avail
  avail=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
  [ "$avail" -ge 50 ] || { echo "ABORT: /data 只剩 ${avail}G,低于 50G 闸门"; return 1; }
}

case "${1:-}" in

start|resume)
  disk_guard || exit 3
  pgrep -f "[t]rain_smith_agent.py" >/dev/null && { echo "ABORT: 已有 trainer 在跑"; exit 2; }
  mkdir -p $CK
  cd $EX
  env "${TRAIN_ENV[@]}" nohup bash restart_s2_smith_trainer.sh >> $CK/s2_launch.log 2>&1 &
  echo "s2 已拉起($1),日志 $CK/trainer_s2.log"
  ;;

wait)
  S=${2:?用法: $0 wait <STEP>}
  echo "等 global_step_$S/data.pt ..."
  while [ ! -f $CK/global_step_$S/data.pt ]; do
    pgrep -f "[t]rain_smith_agent.py" >/dev/null || { echo "TRAINER_DEAD 训练进程没了,看 $CK/trainer_s2.log"; exit 4; }
    sleep 30
  done
  echo "CKPT_READY step=$S $(date +%T)"
  ;;

pause)
  P=$(pgrep -f "[t]rain_smith_agent.py" | head -1)
  [ -n "$P" ] || { echo "训练进程已不在,跳过"; exit 0; }
  echo "SIGINT -> trainer pid $P"; kill -INT $P
  for _ in $(seq 1 60); do kill -0 $P 2>/dev/null || break; sleep 5; done
  kill -0 $P 2>/dev/null && { echo "5min 未退,SIGTERM"; kill -TERM $P; sleep 30; }
  kill -0 $P 2>/dev/null && { echo "仍在,SIGKILL"; kill -9 $P; sleep 20; }
  kubectl get jobs -n default --no-headers 2>/dev/null | awk '$1 ~ /^agl-rollout-/ {print $1}' \
    | xargs -r kubectl delete job -n default --wait=false >/dev/null 2>&1
  sleep 20
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i 1,2,3,5 \
    | awk -F'[ ,]+' '{printf "卡%s 空闲%dMiB %s\n",$1,$4-$2,($4-$2>=42000?"OK":"不足")}'
  ;;

probe)
  S=${2:?用法: $0 probe <STEP>}
  [ -f $CK/global_step_$S/data.pt ] || { echo "ABORT: global_step_$S/data.pt 不存在"; exit 2; }
  HF=/workspace/models/s2_step${S}_hf
  OUT=val_s2_step$S
  if [ ! -d $HF ]; then
    cd $ROOT
    .venv/bin/python -m verl.model_merger merge --backend fsdp \
      --local_dir $CK/global_step_$S/actor --target_dir $HF 2>&1 | tail -2
    cp -n /workspace/models/MiniCPM5-2B/chat_template.jinja $HF/ 2>/dev/null
  fi
  cd $EVALROOT
  FLEET_CARDS="5 3 1 2" bash smith_fleet.sh start $HF $OUT s2s$S || exit 5
  while [ "$(ls $EVALROOT/$OUT/*/result.json 2>/dev/null | wc -l)" -lt 474 ]; do sleep 30; done
  bash smith_fleet.sh stop s2s$S >/dev/null 2>&1
  python3 $EX/s2_verdict.py $EVALROOT/$OUT
  ;;

*) echo "用法: $0 {start|wait <STEP>|pause|probe <STEP>|resume}"; exit 1;;
esac
