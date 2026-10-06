#!/bin/bash
# 10-06 OPSD go/no-go 诊断:同一权重(SFT ep3)在固定 100 题 val 子集上跑四种特权信息条件,
# 只看 resolved / 交卷率的差距。只诊断,不进训练。
#   none  = 原 harness(对照;与 val_sft3_tmpl2 同配置,可核对抽样噪声)
#   tests = 题面附 F2P 测试 node id(数据集自带,训练时零成本可得)
#   files = 题面附金标补丁触及的源文件路径
#   patch = 题面附金标补丁本身(上限条件)
# 每个条件起一次 fleet(vLLM ×4 卡 + 4 分片),跑完 stop,再起下一个;并发与 474 评测逐字相同,
# 所以单题 600s 超时的口径不变。结果目录 $E/val_hint_<mode>,日志 hint_diag/run_hint_diag.log。
set -u
E=/workspace/agl-checkpoints/swe_smith_smoke
MODEL=${HINT_MODEL:-/workspace/models/MiniCPM5-2B-sft-v3-ep3}
MODES=${HINT_MODES:-"none tests files patch"}
IDS=$E/hint_diag/val100_ids.txt
LOG=$E/hint_diag/run_hint_diag.log
ts() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a $LOG; }
cd $E
[ -s $IDS ] || { say "FATAL no ids file"; exit 2; }
say "start model=$MODEL modes=[$MODES] n_ids=$(wc -l < $IDS)"
for m in $MODES; do
  OUT=val_hint_$m; TAG=hint$m
  if [ -f STOP_HINT_DIAG ]; then say "STOP_HINT_DIAG present; abort before $m"; exit 3; fi
  say "== $m: fleet start -> $OUT"
  AGL_SWEEP_ONLY=$IDS SMITH_HINT_MODE=$m ./smith_fleet.sh start $MODEL $OUT $TAG 2>&1 | tail -3 | tee -a $LOG
  /usr/bin/grep -q "FLEET_UP $TAG" $LOG || { say "FLEET_FAIL for $m"; ./smith_fleet.sh stop $TAG >/dev/null 2>&1; exit 4; }
  ./smith_fleet.sh wait $TAG $OUT 2>&1 | tail -2 | tee -a $LOG
  ./smith_fleet.sh stop $TAG 2>&1 | tail -1 | tee -a $LOG
  n=$(ls $E/$OUT/*/result.json 2>/dev/null | wc -l)
  say "== $m done: results=$n"
done
python3 $E/hint_diag/analyze.py 2>&1 | tee -a $LOG
say "HINT_DIAG_FINISHED"
