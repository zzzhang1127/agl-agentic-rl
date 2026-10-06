#!/bin/bash
# s3 训练 / 探针循环驱动(10-05)。
#
#   起点   MiniCPM5-2B-sft-v3-ep3(474 全量 136,当前最好)
#   lr     2e-6(s1 已验证不坍塌;s2 的 5e-6 四步就崩)
#   奖励    **训练集改成 F2P 通过比例**(SMITH_F2P_PARTIAL=1),P2P 仍是二值硬闸;
#          验证集永远二值 —— 基座 123/474、SFT 136/474 都是二值口径
#   数据    train_dataset_s3.jsonl(700 = 筛过的 741 去掉 41 道「P2P 过滤后无护栏」)
#
# 两层探针。立项时写的理由是「s2 坍塌时奖励完全没报警」,**10-05 复查发现那是我盯
# 错了指标**:`critic/score/mean` 确实全程 0.40~0.56,但 `training/reward` 在同一步
# 掉了 4.2 倍(0.580→0.138)。score/mean 不能当奖励看(bf16 量化 + 行加权 + 过滤后
# 选择偏差,s2 上整体偏高 +0.2045),详见 s3_watch.py 模块 docstring 的 a/b/c。
# 结论没变、理由变了。两条判据各有胜负(连续2步口径):健康侧长度 1.66× > 奖励 1.50×,
# 坍塌侧奖励 1.54× > 长度 1.18×。长度仍是主判据的理由是**可测的**:s2 第 5 步奖励已
# 回升过门槛、判据 7 沉默,长度还在门槛下 —— 探针漏一轮轮询,长度兜得住。而健康的
# s1 奖励会掉到 0.2137,和 s2 坍塌值 0.1375 区间重叠,所以奖励只能用相对门槛。
#   Tier 1  s3_watch.py   每 2 分钟读指标,8 条判据,越线自动 SIGINT 并等人
#           已在 s1(90 步)/ s2(5 步)上回放验收:s2 step4 开火(判据 1/2/7 同步),
#           s1 全程不开火;验收**逐条**核每个标定过的判据各自开火(见 §53)
#   Tier 2  probe <STEP>  停训跑**全量 474**,和 SFT ep3 配对 McNemar(s2_verdict.py)
#
# 为什么 Tier 2 要停训:474 评测要四张卡各起 40G vLLM,和训练抢同样的卡 1/2/3/5。
# 一次约 45 分钟,PROBE_EVERY=10 时开销 ≈9%。
#
#   bash s3_cycle.sh start           # 起训 + 拉起 Tier 1 探针
#   bash s3_cycle.sh wait   <STEP>   # 阻塞到 global_step_<STEP>/data.pt 落盘
#   bash s3_cycle.sh pause           # 停训(SIGINT 优先,存盘后退)
#   bash s3_cycle.sh probe  <STEP>   # merge + 474 + 判决(退出码 1 = STOP)
#   bash s3_cycle.sh resume
#   bash s3_cycle.sh status          # 一行现状(巡检用)
#
# 纪律:不用 `ray stop --force`;只动 agl-rollout-*;不碰邻居进程;/data 红线 50G。
set -u
ROOT=/workspace/projects/agent-lightning
EX=$ROOT/examples/swe_smith
CK=/workspace/agl-checkpoints/smith_rl_s3
EVALROOT=/workspace/agl-checkpoints/swe_smith_smoke
PROBE_EVERY=${PROBE_EVERY:-10}

# 训练 env:沿用 s1 已验证的那一组。lr 和起点在 restart_s3_smith_trainer.sh 里已是默认。
# TEST_FREQ=0 关掉 48 题内置探针 —— 它被全量 474 取代了,再跑是白烧 gen 时间;而且
# 48 题的判别力实测不够(s1 八个点均值 7.63 全落在 ±1σ 内,见 smith-rl-s1-run)。
TRAIN_ENV=(TRAIN_BATCH=16 PPO_MINI=16 ROLLOUT_N=4 VLLM_SEQS=16 EAGER=False
           TEST_FREQ=0 SAVE_FREQ=5)

disk_guard() {
  local avail
  avail=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
  [ "$avail" -ge 50 ] || { echo "ABORT: /data 只剩 ${avail}G,低于 50G 闸门"; return 1; }
}

# 只认「真的是 python 在跑 s3_watch.py」的进程:argv[0] 是 python、argv[1] 以
# s3_watch.py 结尾。为什么不直接 pgrep -f:实测它会匹配到**我自己的 shell 命令行**
# (命令里出现过这个串就算)。误报的后果是「以为探针在跑、其实没起」—— 和 §50
# 同一类静默失效。而且 pause 里原来用 pkill -f 模式匹配,对 trainer 是明令禁止的,
# 对探针也不该例外。
watch_pid() {
  local p a
  for p in $(pgrep -f s3_watch.py 2>/dev/null); do
    [ -r /proc/$p/cmdline ] || continue
    mapfile -d '' -t a < /proc/$p/cmdline 2>/dev/null || continue
    case "${a[0]##*/}" in python|python3|python3.*) ;; *) continue ;; esac
    case "${a[1]:-}" in *s3_watch.py) echo "$p"; return 0 ;; esac
  done
  return 1
}

# 探针在被信任之前先自证。§50:回放验收只验证「用到的」判据,指标解析失败是静默的
# (缺值和「这版 verl 没这项」在 judge() 里长得一样),所以必须有解析率断言。
selftest_gate() {
  python3 $EX/s3_watch.py --selftest || {
    echo "ABORT: s3_watch.py 自检失败 —— 探针不可信,不许在没有探针的情况下起训"; return 1; }
}

watch_up() {   # Tier 1 探针。重复调用是安全的(已在跑就跳过)。
  local p
  p=$(watch_pid) && { echo "Tier1 探针已在跑 pid $p"; return 0; }
  selftest_gate || return 1
  nohup python3 $EX/s3_watch.py >> $CK/s3_watch.log 2>&1 &
  echo "Tier1 探针已拉起 pid $!,日志 $CK/s3_watch.log"
}

case "${1:-}" in

start|resume)
  disk_guard || exit 3
  # 自检放在起训**之前**:探针坏了就根本不要起训,而不是起完训才发现没探针。
  selftest_gate || exit 8
  pgrep -f "[t]rain_smith_agent.py" >/dev/null && { echo "ABORT: 已有 trainer 在跑"; exit 2; }
  # 上一轮探针开过火就不许无脑续训 —— 必须人看过并挪走告警文件。
  [ -f $CK/WATCH_ALARM.json ] && {
    echo "ABORT: $CK/WATCH_ALARM.json 还在(上次探针开过火)。看过之后 mv 走再续训:"
    sed -n '1,25p' $CK/WATCH_ALARM.json; exit 6; }
  mkdir -p $CK
  cd $EX
  env "${TRAIN_ENV[@]}" nohup bash restart_s3_smith_trainer.sh >> $CK/s3_launch.log 2>&1 &
  echo "s3 已拉起($1),日志 $CK/trainer_s3.log"
  sleep 5; watch_up
  ;;

watch) watch_up ;;

wait)
  S=${2:?用法: $0 wait <STEP>}
  echo "等 global_step_$S/data.pt ..."
  while [ ! -f $CK/global_step_$S/data.pt ]; do
    pgrep -f "[t]rain_smith_agent.py" >/dev/null || { echo "TRAINER_DEAD 训练进程没了,看 $CK/trainer_s3.log"; exit 4; }
    [ -f $CK/WATCH_ALARM.json ] && { echo "WATCH_ALARM Tier1 探针开火,停在这里等人:"; cat $CK/WATCH_ALARM.json; exit 7; }
    sleep 30
  done
  echo "CKPT_READY step=$S $(date +%T)"
  ;;

pause)
  # 探针先撤,免得把人为停训当坍塌。按 pid 杀,不用 pkill 模式匹配。
  W=$(watch_pid) && { echo "撤 Tier1 探针 pid $W"; kill -TERM $W 2>/dev/null; }
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
  HF=/workspace/models/s3_step${S}_hf
  OUT=val_s3_step$S
  if [ ! -d $HF ]; then
    cd $ROOT
    .venv/bin/python -m verl.model_merger merge --backend fsdp \
      --local_dir $CK/global_step_$S/actor --target_dir $HF 2>&1 | tail -2
    cp -n /workspace/models/MiniCPM5-2B/chat_template.jinja $HF/ 2>/dev/null
  fi
  cd $EVALROOT
  FLEET_CARDS="5 3 1 2" bash smith_fleet.sh start $HF $OUT s3s$S || exit 5
  while [ "$(ls $EVALROOT/$OUT/*/result.json 2>/dev/null | wc -l)" -lt 474 ]; do sleep 30; done
  bash smith_fleet.sh stop s3s$S >/dev/null 2>&1
  python3 $EX/s2_verdict.py $EVALROOT/$OUT     # 判决脚本与血统无关,基线默认 SFT ep3
  ;;

status)
  P=$(pgrep -f "[t]rain_smith_agent.py" | head -1)
  W=$(watch_pid || true)
  ST=$(ls -d $CK/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
  echo "trainer=${P:-无} tier1=${W:-无} 最新ckpt=step${ST:-无} 告警=$([ -f $CK/WATCH_ALARM.json ] && echo 有 || echo 无)"
  tail -1 $CK/s3_health.jsonl 2>/dev/null
  ;;

*) echo "用法: $0 {start|resume|watch|wait <STEP>|pause|probe <STEP>|status}"; exit 1;;
esac
