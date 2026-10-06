#!/bin/bash
# s5 训练 / 探针循环驱动(10-06 13:00)。= s4_cycle.sh 逐字血统替换(第 51 行起 s4→s5),
# 驱动的是 restart_s5_smith_trainer.sh + job-template-smith-s5.yaml。
#
# s5 与 s4 的**唯一**配方差别:训练奖励不整形(SMITH_LEN_PEN_LAMBDA=0、SMITH_PROMPT_PEN_MAX=0,
# reward==raw)。其余 lr 2e-6 / KL 0.001 / SFT ep3 起点 / F2P 比例部分分 / P2P 硬闸 / 700 题 /
# TRAIN_BATCH=16 ROLLOUT_N=4 PPO_MINI=4 / 动态采样 / 40 轮 全部相同。
#
# 为什么是这一刀:s4 step20 做对 134→117(p=0.030),配对画像是"早交卷"(提交率 51→71%、
# 生成 token 36983→19829、轮数 27.8→22.4)。训练侧证据:组内"同样做对"的 rollout 之间整形后
# reward 差 -0.0032/轮(t=-35.18)而 raw 方差为 0;31/391 组(7.9%)的优势 100% 来自轮数罚,
# 并被它从零方差过滤里捞回。KL 0.001 只占梯度 ~2%,拦不住。这是头号嫌疑、尚未消融 ⇒ s5 就是消融。
#
# 可证伪预测(s5 若机理成立):训练侧 response_length/mean 不再单调下行(s4 为 -343 tok/步,
# t=-4.64);纯整形组消失 ⇒ 零方差组比例升约 8pp,有梯度组由 ~300/391 降到 ~269/391。
#
# 对照:val_sft3_tmpl2 = 134/474(10-06 重测,当前 harness)。探针 PROBE_EVERY=10 全量 474 配对
# McNemar;**明显下降(p<0.05 且低于 134)就停**(人类 10-05 指令)。
#
# 用法同 s4:
#   bash s5_cycle.sh start | wait <STEP> | pause | probe <STEP> | resume | status
#
# 纪律:不用 `ray stop --force`;只动 agl-rollout-*;不碰邻居进程;/data 红线 50G(size-used)。
set -u
ROOT=/workspace/projects/agent-lightning
EX=$ROOT/examples/swe_smith
PY=$ROOT/.venv/bin/python
CK=/workspace/agl-checkpoints/smith_rl_s5
EVALROOT=/workspace/agl-checkpoints/swe_smith_smoke
PROBE_EVERY=${PROBE_EVERY:-10}
export S3_CKPT=$CK          # 探针据此推出 trainer_s5.log / s5_health.jsonl

# 训练 env:沿用 s1 已验证的那一组。lr / 起点 / MAXRESP / PPO_MINI 在
# restart_s5_smith_trainer.sh 里已经是默认值(PPO_MINI=4、MAXRESP=43008),这里不重复。
# TEST_FREQ=0 关掉 48 题内置探针:判别力实测不够(s1 八个点均值 7.63 全落在 ±1σ 内)。
TRAIN_ENV=(TRAIN_BATCH=16 ROLLOUT_N=4 VLLM_SEQS=16 EAGER=False TEST_FREQ=0 SAVE_FREQ=5)

# /data 的 root 可用量 = size - used。df 的 avail 扣掉了 276G 的 root 保留块,
# 是非 root 口径(10-06:avail 13G / root 可用 310G)。拿 avail 当闸门会误杀。
disk_guard() {
  local free
  free=$(df -B1 /data | tail -1 | awk '{printf "%d", ($2-$3)/1073741824}')
  [ "${free:-0}" -ge 50 ] || { echo "ABORT: /data root 可用只剩 ${free}G,低于 50G 闸门"; return 1; }
  echo "磁盘 OK:root 可用 ${free}G"
}

# 只认「真的是 python 在跑 s3_watch.py」的进程。不用 pgrep -f 直接判:实测它会匹配到
# 我自己的 shell 命令行(命令里出现过这个串就算),误报成「探针在跑、其实没起」。
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

# 探针在被信任之前先自证。解析失败在 judge() 里是静默的(缺值和「这版 verl 没这项」
# 长得一样),所以必须有解析率断言;判据 9 还要求在两份已知坏例上都开火。
selftest_gate() {
  $PY $EX/s3_watch.py --selftest || {
    echo "ABORT: s3_watch.py 自检失败 —— 探针不可信,不许在没有探针的情况下起训"; return 1; }
}

# 模板一致性闸门。§56/§57/§58 连栽三次,每次都是同一个形状:
# **「我以为在用的那份模板」≠「真正被加载的那份」**。
#   §56 改了模板,但训练侧的 prompt 根本不是模板渲染的;
#   §57 修对了代码,但 agl-server 是长命进程、`_TOKENIZERS` 是进程内缓存;
#   §58 机理定对了,补丁却打在 base 目录,而 proxy 按**请求里的 model 名**加载的是
#       **actor 目录**那份(agent 发 model="auto" → 上游真实模型名 = ACTOR_DIR)。
# 三次的代价各是一轮 smoke。人记不住,所以写成断言:退出码非 0 就不许起训。
# 顺带检查 agl-server 的启动时间是否晚于任一模板 mtime —— 改磁盘文件不算改。
ACTOR_DIR=${MODEL:-/workspace/models/MiniCPM5-2B-sft-v3-ep3}
BASE_DIR=/workspace/models/MiniCPM5-2B
template_gate() {
  $PY $EX/preflight_templates.py \
      --dirs "$ACTOR_DIR" "$BASE_DIR" \
      --explicit-template "$BASE_DIR/chat_template.jinja" \
      --agl-port 18082 || {
    echo "ABORT: 模板 preflight 不通过 —— 往返不变式一破,多轮轨迹合并必败"
    echo "       (每一轮被单独 flush 成一行、各自带整条 episode 的最终奖励,"
    echo "        也就是 s1/s2/s3 三条血统都没增益的那个上游病因)。先修模板。"
    return 1; }
}

watch_up() {   # Tier 1 探针。重复调用是安全的(已在跑就跳过)。
  local p
  p=$(watch_pid) && { echo "Tier1 探针已在跑 pid $p"; return 0; }
  selftest_gate || return 1
  nohup $PY $EX/s3_watch.py >> $CK/s5_watch.log 2>&1 &
  echo "Tier1 探针已拉起 pid $!,日志 $CK/s5_watch.log"
}

case "${1:-}" in

start|resume)
  disk_guard || exit 3
  # 自检放在起训**之前**:探针坏了就根本不要起训,而不是起完训才发现没探针。
  selftest_gate || exit 8
  # 同理,模板坏了也不要起训 —— 它坏的时候训练照样跑完、指标照样有值,
  # 只是学的东西不是多轮轨迹。这种「不报错的坏」必须在起训前拦。
  template_gate || exit 10
  pgrep -f "[t]rain_smith_agent.py" >/dev/null && { echo "ABORT: 已有 trainer 在跑"; exit 2; }
  # 上一轮探针开过火就不许无脑续训 —— 必须人看过并挪走告警文件。
  [ -f $CK/WATCH_ALARM.json ] && {
    echo "ABORT: $CK/WATCH_ALARM.json 还在(上次探针开过火)。看过之后 mv 走再续训:"
    sed -n '1,25p' $CK/WATCH_ALARM.json; exit 6; }
  # start 必须是**干净起训**。restart_s5 用 trainer.resume_mode=auto,只要目录里
  # 留着任何 global_step_*(典型来源:SMOKE 跑完存的那一个 —— 它是 TRAIN_BATCH=2 /
  # PPO_MINI=2、而且 smoke#1 那次还带着断链 bug),它就会**悄悄从那里续训**,
  # 而日志里只有一行 resume 提示。这正是「从 SFT ep3 重新起训、不继承污染权重」
  # 这条要求最容易被破掉的地方,所以宁可在这里硬停。
  if [ "$1" = start ] && ls -d $CK/global_step_* >/dev/null 2>&1; then
    echo "ABORT: $CK 下已有 ckpt,start 会被 resume_mode=auto 变成续训:"
    ls -d $CK/global_step_* | sed 's/^/  /'
    echo "  要干净起训就先挪走(例:mv $CK/global_step_1 $CK/smoke_global_step_1.bak"
    echo "  并删掉 $CK/latest_checkpointed_iteration.txt);确实想续训就用 resume。"
    exit 9
  fi
  mkdir -p $CK
  cd $EX
  env "${TRAIN_ENV[@]}" nohup bash restart_s5_smith_trainer.sh >> $CK/s5_launch.log 2>&1 &
  echo "s5 已拉起($1),日志 $CK/trainer_s5.log"
  sleep 5; watch_up
  ;;

watch) watch_up ;;

wait)
  S=${2:?用法: $0 wait <STEP>}
  echo "等 global_step_$S/data.pt ..."
  while [ ! -f $CK/global_step_$S/data.pt ]; do
    pgrep -f "[t]rain_smith_agent.py" >/dev/null || { echo "TRAINER_DEAD 训练进程没了,看 $CK/trainer_s5.log"; exit 4; }
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
  # SIGINT 先礼后兵:万一哪天 verl 肯在 SIGINT 上 flush 一次 ckpt,这条路还在。
  # 但实测它**从不**响应:10-06 一天之内 4 次(step10 探针、step20 探针、
  # step20 停训,外加 09-xx 一次)全都是等满了再靠 SIGTERM 收掉。
  # 原来等 60×5s=300s —— 一轮探针白扔 5 分钟,12 轮就是 1 小时训练时间。
  # 改成 12×5s=60s:足够给任何真会响应的版本机会,又不至于把时间烧在空等上。
  echo "SIGINT -> trainer pid $P"; kill -INT $P
  for _ in $(seq 1 12); do kill -0 $P 2>/dev/null || break; sleep 5; done
  kill -0 $P 2>/dev/null && { echo "1min 未退,SIGTERM(实测必走这条)"; kill -TERM $P; sleep 30; }
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
  # 探针和训练抢的是同一组卡:FLEET_CARDS="5 3 1 2" 就是 GPUS=1,2,3,5
  # (restart_s5_smith_trainer.sh:123)。训练侧卡上已经有 FSDP actor + vLLM
  # rollout engine,再并发起 4 个评测 vLLM,必然挤爆一边 —— 一次探针废掉整条跑。
  # 第 34 行注释本来就写着「停训跑全量 474」,但代码从没落实,所以写成硬拒。
  # 不做自动 pause:那会让 probe 变成破坏性操作,万一评测起不来就白停了训练。
  if pgrep -f "[t]rain_smith_agent.py" >/dev/null; then
    echo "ABORT: 训练进程还在跑,探针会和它抢卡 1/2/3/5 → OOM"
    echo "       正确顺序:bash $0 pause  →  bash $0 probe $S  →  bash $0 resume"
    exit 11
  fi
  HF=/workspace/models/s5_step${S}_hf
  OUT=val_s5_step$S
  if [ ! -d $HF ]; then
    cd $ROOT
    $PY -m verl.model_merger merge --backend fsdp \
      --local_dir $CK/global_step_$S/actor --target_dir $HF 2>&1 | tail -2
    # 模板必须跟着权重走:补丁后的生成位默认发 `<think>\n\n</think>\n\n`,
    # 评测侧少这一份就和训练侧的 prompt 口径分叉。两条路径渲染逐字相同已验过,
    # 所以这不改变历史 474 基线的可比性。
    cp -n /workspace/models/MiniCPM5-2B/chat_template.jinja $HF/ 2>/dev/null
  fi
  cd $EVALROOT
  FLEET_CARDS="5 3 1 2" bash smith_fleet.sh start $HF $OUT s5s$S || exit 5
  while [ "$(ls $EVALROOT/$OUT/*/result.json 2>/dev/null | wc -l)" -lt 474 ]; do sleep 30; done
  bash smith_fleet.sh stop s5s$S >/dev/null 2>&1
  # 基线必须是**同一套模板**下测的那一份。s2_verdict.py:23 的默认值 val_sft3_fixed
  # (127/474)是旧模板下的读数,而补丁改了 13.3% 的 assistant 轮的历史渲染 ——
  # 拿新模板的 step-N 去比旧模板的基线,涨跌分不清是 RL 还是模板,判据直接作废。
  # 所以显式传 val_sft3_tmpl2(SFT ep3 在当前模板下重测的那份),并先确认它测满了。
  BASE=$EVALROOT/val_sft3_tmpl2
  NB=$(ls $BASE/*/result.json 2>/dev/null | wc -l)
  [ "$NB" -eq 474 ] || { echo "ABORT: 基线 $BASE 只有 $NB/474,不能配对比较"; exit 12; }
  $PY $EX/s2_verdict.py $EVALROOT/$OUT $BASE
  ;;

status)
  P=$(pgrep -f "[t]rain_smith_agent.py" | head -1)
  W=$(watch_pid || true)
  ST=$(ls -d $CK/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
  echo "trainer=${P:-无} tier1=${W:-无} 最新ckpt=step${ST:-无} 告警=$([ -f $CK/WATCH_ALARM.json ] && echo 有 || echo 无)"
  tail -1 $CK/s5_health.jsonl 2>/dev/null
  ;;

*) echo "用法: $0 {start|resume|watch|wait <STEP>|pause|probe <STEP>|status}"; exit 1;;
esac
