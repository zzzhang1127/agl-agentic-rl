#!/usr/bin/env bash
# 收官评测驱动:基座和 step1550 顺序各跑一次全量 1319 题。
# 顺序跑(不并发)是因为 eval_gsm8k_full.sh 固定 PORT/RAY_PORT,并发会撞端口。
set -uo pipefail
EX=/workspace/projects/agent-lightning/examples/gsm8k
CK=/workspace/agl-checkpoints/gsm8k_r1
export CARD=${CARD:-2}
for spec in "base:" "step1550:/workspace/models/gsm8k_r1_step1550_hf"; do
  tag=${spec%%:*}; model=${spec#*:}
  echo "=== [$(date +%H:%M:%S)] 开始 $tag ==="
  bash $EX/eval_gsm8k_full.sh "$tag" ${model:+"$model"}
  echo "=== [$(date +%H:%M:%S)] $tag 结束 ==="
done
echo "=== 汇总 ==="
for tag in base step1550; do
  v=$(grep -oE "val/reward[^:]*:[0-9.]+" $CK/eval_full1319_$tag/eval.log 2>/dev/null | tail -1)
  echo "$tag  $v"
done
