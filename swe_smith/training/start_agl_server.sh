#!/bin/bash
# 起 agl-server(训练用的 AGL proxy),**带正确的 no_proxy**。
#
# 为什么要专门一个脚本:10-05 深夜我手工重启 agl-server,从一个 no_proxy 为空的
# shell 里起,结果 proxy 转发给上游 vLLM 的每一个请求都被本机 http_proxy
# (<CORP_PROXY>)截走 → 全部 502 → rollout 侧 "LLM call failed / burning turn",
# 8 条 rollout 烧了十五六轮一条都没成,一次 smoke 全废。
#
# 关键事实(反直觉,所以写死在脚本里):
#  * 上游 vLLM 的地址是 **eth0 的 <LAN_IP>**,不是 127.0.0.1 —— verl 的
#    vllm_async_server 按节点 IP 注册,所以只放 127.0.0.1/localhost 没用。
#  * httpx 的 no_proxy **不支持 CIDR**:它把每一项映射成 `all://<host>` 的 mount,
#    只做精确/后缀匹配。所以这里逐个列本机 IP,不写 172.17.0.0/16。
#  * 验证必须用 httpx 自己选路(python),**不能用 `curl --noproxy`** —— curl 的
#    参数只证明 curl 绕过了,证明不了 python 侧的 env 生效。
#
#   bash start_agl_server.sh            # 默认 18082
#   PORT=18083 bash start_agl_server.sh
set -u
ROOT=/workspace/projects/agent-lightning
PORT=${PORT:-18082}
MODEL=${MODEL:-/workspace/models/MiniCPM5-2B}
LOG=${LOG:-/workspace/agl-checkpoints/agl_server_${PORT}.log}

# 按网卡生成 no_proxy:本机所有 IPv4 + 回环 + 常用别名。
IPS=$(ip -4 -o addr show 2>/dev/null | awk '{split($4,a,"/"); print a[1]}' | sort -u | paste -sd,)
export no_proxy="127.0.0.1,localhost,::1,$IPS"
export NO_PROXY="$no_proxy"
echo "no_proxy=$no_proxy"

# 只杀「确实是 agl-server 且端口是本端口」的进程。绝不 pkill 模式匹配 ——
# 这台机器上有邻居的服务,误杀的代价远大于多等一会儿。
for p in $(pgrep -f "agl-server" 2>/dev/null); do
  [ -r /proc/$p/cmdline ] || continue
  c=$(tr '\0' ' ' < /proc/$p/cmdline)
  case "$c" in *agl-server*"port=$PORT"*) echo "停掉旧 agl-server pid $p"; kill -TERM $p; sleep 3;; esac
done

cd $ROOT
nohup .venv/bin/python3 .venv/bin/agl-server \
  port=$PORT host=0.0.0.0 key=dummy "default_proxy.model_name=$MODEL" >> "$LOG" 2>&1 &
NEW=$!
echo "agl-server 已拉起 pid $NEW,端口 $PORT,日志 $LOG"

for i in $(seq 1 30); do
  sleep 1
  curl -s -o /dev/null --noproxy '*' "http://127.0.0.1:$PORT/api/rollouts?limit=1" && { echo "端口 $PORT 就绪($i s)"; break; }
done

# 选路自检:用 httpx 打一个本机不存在的端口。直连 → ConnectError;走代理 → 代理会
# 回一个 HTTP 响应(502/503)。后者说明 no_proxy 没生效,这时候**不要起训**。
.venv/bin/python - "$IPS" <<'PY'
import sys, httpx
ip = sys.argv[1].split(",")[0]
for host in {ip, "<LAN_IP>"}:
    try:
        r = httpx.get(f"http://{host}:1/", timeout=5.0)
        print(f"  !! {host}: 收到 HTTP {r.status_code} —— 走了代理,no_proxy 没生效")
    except httpx.ConnectError:
        print(f"  OK {host}: ConnectError(直连,未经代理)")
    except Exception as e:
        print(f"  ?  {host}: {type(e).__name__}: {e}")
PY
