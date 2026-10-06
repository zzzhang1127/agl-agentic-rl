#!/usr/bin/env bash
# k3s DiskPressure 闩锁解除器 —— 2026-10-02 为 s1 血统写的,根因见下。
#
# 症状:rollout pod 全部 Pending,kubectl describe 说
#   FailedScheduling: 0/1 nodes are available: 1 node(s) had untolerated taint(s)
# 节点带着 node.kubernetes.io/disk-pressure=:NoSchedule,DiskPressure=True。
#
# 根因(2026-10-02 实测确认,整整闩死 8 天):
#   1. 2026-09-24 10:03:41,/data 真的掉到 5GiB 以下,踩中 kubelet 硬阈值
#      nodefs.available<5Gi,DiskPressure 置 True —— 这一步是对的。
#   2. k3s 默认带 evictionMinimumReclaim={nodefs.available:10%, imagefs.available:10%}。
#      kubelet 的 synchronize() 里,对**已经 met 过**的阈值会再用
#      enforceMinReclaim=true 复核一遍(eviction/helpers.go thresholdsMet):
#      解除条件变成 available > 5Gi + 10%×capacity。本机 capacity 6046G,
#      → 需要 ~610G 空闲才肯解锁。盘上常驻 5.6T 数据,这个条件永远满足不了。
#   3. 于是驱逐循环每 10 秒跑一次(实测 30 分钟 1056 次),每次先 DeleteUnusedImages
#      (想释放 MaxInt64 字节,全部因为"容器正在引用"失败),再去排队驱逐 pod。
#      kube-system 的 pod 活下来的唯一原因是 eviction_manager.go:616
#      "cannot evict a critical pod"。
#   4. 这个闩只在 kubelet 进程内存里(m.thresholdsMet),重启即清。
#      10-02 17:37 实测:systemctl restart k3s 之后 **5 秒** DiskPressure 翻 False、污点消失。
#
# 为什么不是"磁盘真满了":判危要看硬阈值 5GiB,不是看百分比。
# imagefs/nodefs 都是 /data(docker root = /data/docker),98.14% 用掉 = 还剩 113G,
# 离 5GiB 硬阈值差着 20 倍。`/` 那个 100% 满的 2T 盘根本不参与 kubelet 的判断。
#
# 安全阀:只有在可用空间**远高于**硬阈值(默认 20GiB)时才认定是误闩并重启。
# 真的快满了就不要重启 k3s —— 那时候 DiskPressure 是实话,该去清磁盘。
#
# 重启 k3s 的副作用评估(10-02 实测过一次):
#   - 钉住 132 个 swesmith 镜像的是 96 个 pin-* + 22 个 agl-keep-val-* **纯 docker**
#     容器,只有 6 个容器带 io.kubernetes.pod.uid 标签。kubelet 的 container GC
#     只回收自己管的,所以重启不会解钉镜像,镜像 GC 也就删不掉它们。
#   - 集群上除了 kube-system 3 个 pod 就只有我自己的 rollout Job,没有邻居的负载。
#   - 已经 Pending 的 Job 不会丢,污点一消就自己调度(实测 13 秒内 8/8 Running)。
#
# 用法:
#   bash k3s_unlatch.sh          # 检查,必要时重启 k3s,等污点消失
#   DRY_RUN=1 bash k3s_unlatch.sh # 只报告不动手
# 退出码:0 = 节点可调度(本来就好 / 已修好);1 = 需要人介入(真的磁盘不够等)。
set -uo pipefail

MIN_FREE_GIB="${MIN_FREE_GIB:-20}"   # 低于这个就不认定误闩(硬阈值 5GiB 的 4 倍余量)
WAIT_S="${WAIT_S:-90}"               # 重启后等污点消失的上限
DRY_RUN="${DRY_RUN:-0}"
TAINT="node.kubernetes.io/disk-pressure"

log() { echo "[k3s_unlatch $(date +%H:%M:%S)] $*"; }

command -v kubectl >/dev/null 2>&1 || { log "没有 kubectl,跳过检查"; exit 0; }

NODE="${NODE:-$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)}"
[ -n "$NODE" ] || { log "拿不到节点名(API 不通?),跳过"; exit 1; }

node_state() {
  DP=$(kubectl get node "$NODE" -o jsonpath='{range .status.conditions[?(@.type=="DiskPressure")]}{.status}{end}' 2>/dev/null)
  TAINTS=$(kubectl get node "$NODE" -o jsonpath='{.spec.taints[*].key}' 2>/dev/null)
}

node_state
log "节点 $NODE: DiskPressure=${DP:-?} taints=[${TAINTS:-none}]"

case "$TAINTS" in
  *"$TAINT"*) ;;
  *) log "没有 disk-pressure 污点,不用动。"; exit 0 ;;
esac

# docker 的 root dir 决定 kubelet 的 imagefs/nodefs 在哪个文件系统
DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
FS_TARGET=$([ -d "$DOCKER_ROOT" ] && echo "$DOCKER_ROOT" || echo /)
AVAIL_B=$(df -B1 --output=avail "$FS_TARGET" 2>/dev/null | tail -1 | tr -d ' ')
AVAIL_GIB=$(( ${AVAIL_B:-0} / 1024 / 1024 / 1024 ))
log "imagefs/nodefs = $FS_TARGET,可用 ${AVAIL_GIB}GiB(硬阈值 5GiB,误闩判定线 ${MIN_FREE_GIB}GiB)"

if [ "$AVAIL_GIB" -lt "$MIN_FREE_GIB" ]; then
  log "可用空间太少,DiskPressure 可能是实话 —— 不重启 k3s,请先清磁盘。"
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN:判定为误闩,本该执行 systemctl restart k3s。"
  exit 0
fi

log "判定为 evictionMinimumReclaim 误闩 → 重启 k3s 清掉 kubelet 内存里的 m.thresholdsMet"
systemctl restart k3s || { log "重启 k3s 失败"; exit 1; }

deadline=$(( $(date +%s) + WAIT_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  node_state
  if [ "${DP:-}" = "False" ] && [ -z "${TAINTS:-}" ]; then
    log "污点已清除(DiskPressure=False),节点恢复可调度。"
    exit 0
  fi
  sleep 3
done
log "等了 ${WAIT_S}s 污点还在:DiskPressure=${DP:-?} taints=[${TAINTS:-none}] —— 需要人看一眼。"
exit 1
