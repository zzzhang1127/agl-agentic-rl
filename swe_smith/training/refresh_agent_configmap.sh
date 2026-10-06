#!/bin/bash
# Push examples/swe_smith/agents/{smith_agent,opencode_agent}.py into the k3s configMap
# that rollout pods mount at /agl/agents (job-template-opencode.yaml:75-79).
# Run this BEFORE resuming training whenever the agent scripts change; pods
# started after the apply pick up the new files, running pods keep the old ones.
set -euo pipefail
EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
AGL_NAMESPACE="${AGL_NAMESPACE:-default}"
cd "$EXAMPLE_DIR/agents" && python3 -m pytest -q test_episode_guard.py test_truncation_stability.py
kubectl -n "$AGL_NAMESPACE" create configmap swe-smith-opencode-scripts \
  --from-file=smith_agent.py="$EXAMPLE_DIR/agents/smith_agent.py" \
  --from-file=opencode_agent.py="$EXAMPLE_DIR/agents/opencode_agent.py" \
  --dry-run=client -o yaml | kubectl -n "$AGL_NAMESPACE" apply -f -
# 反查特征字符串 —— §46.1 的教训:本地改了不等于 pod 里改了,而且**校验必须覆盖
# 这次真正改的东西**。10-05 实测:configMap 里躺的是 1040 行的旧 smith_agent.py
# (evaluate 返回 4 元组、没有 f2p_ratio),只 grep EpisodeGuard 完全照不出来 ——
# 要是就那么起了 s3,整轮会用二值奖励跑完,而我会以为自己测的是部分分。
# 所以每加一个 pod 侧生效的特征,就在这里加一条反查。
for pair in "opencode_agent.py:class EpisodeGuard" \
            "smith_agent.py:f2p_ratio" \
            "smith_agent.py:SMITH_F2P_PARTIAL"; do
  key=${pair%%:*}; pat=${pair#*:}
  n=$(kubectl -n "$AGL_NAMESPACE" get configmap swe-smith-opencode-scripts \
        -o jsonpath="{.data.${key//./\\.}}" | grep -c -- "$pat" || true)
  [ "$n" -gt 0 ] || { echo "FAIL: configMap 的 $key 里找不到 '$pat'"; exit 1; }
  echo "ok: $key 含 '$pat' ($n 处)"
done
echo "configmap refreshed"
