#!/usr/bin/env bash
# OpenCode SWE-smith GRPO: k3s Jobs for rollouts, 4-GPU VERL on the host.
# Start order: server → controller → trainer.
set -euo pipefail
ROLE="${1:-}"
if [ "$ROLE" != "server" ] && [ "$ROLE" != "controller" ] && [ "$ROLE" != "trainer" ]; then
  echo "Usage: $0 {server|controller|trainer} [extra args passed to trainer]"
  exit 1
fi
shift || true
cd "$(dirname "$0")/../.."
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
export PATH="${PWD}/.venv/bin:/workspace/.local/bin:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"

EXAMPLE_DIR="examples/swe_smith"
AGL_SERVER_PORT="${AGL_SERVER_PORT:-18082}"
AGL_KEY="${AGL_KEY:-dummy}"
AGL_MODEL_NAME="${AGL_MODEL_NAME:-/workspace/models/Qwen3-8B/Qwen/Qwen3-8B}"
# Keep OpenCode checkpoints off the smith eval/train dir even if .env sets AGL_CKPT_DIR.
AGL_CKPT_DIR="${AGL_OPENCODE_CKPT_DIR:-/workspace/agl-checkpoints/swe_smith_opencode}"
export AGL_CKPT_DIR
export AGL_TRAJ_DIR="${AGL_TRAJ_DIR:-$AGL_CKPT_DIR}"
# Leftover VRAM on 1/2/3 (PCI 0000) + 5 (PCI 0001). Do not touch GPU5 pid <N>.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,5}"
AGL_NAMESPACE="${AGL_NAMESPACE:-default}"
# k3s pods are on 10.42.0.0/24 and cannot reach docker0 (172.17.0.1); use the node IP.
if [ -z "${AGL_AGENT_URL:-}" ]; then
  NODE_IP="$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || true)"
  NODE_IP="${NODE_IP:-<LAN_IP>}"
  AGENT_URL="http://${NODE_IP}:${AGL_SERVER_PORT}"
else
  AGENT_URL="$AGL_AGENT_URL"
fi
TRAIN_DATASET_PATH="${AGL_TRAIN_DATASET_PATH-$EXAMPLE_DIR/train_dataset_mixed.jsonl}"
VAL_DATASET_PATH="${AGL_VAL_DATASET_PATH-$EXAMPLE_DIR/val_dataset_filtered.jsonl}"
export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-0}"
mkdir -p "$AGL_CKPT_DIR/agl-logs" "$AGL_CKPT_DIR/trajectories"

if { [ "$ROLE" = "controller" ] || [ "$ROLE" = "trainer" ]; } && \
   { [ ! -f "$TRAIN_DATASET_PATH" ] || [ ! -f "$VAL_DATASET_PATH" ]; }; then
  echo "ERROR: missing datasets: $TRAIN_DATASET_PATH $VAL_DATASET_PATH" >&2
  exit 1
fi

if [ "$ROLE" = "server" ]; then
  echo "=== OpenCode GRPO :: server ==="
  agl-server \
    port="$AGL_SERVER_PORT" \
    host="${AGL_SERVER_BIND:-0.0.0.0}" \
    key="$AGL_KEY" \
    default_proxy.model_name="$AGL_MODEL_NAME" &
  SERVER_PID=$!
  cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
  trap cleanup EXIT INT TERM
  ready=false
  for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${AGL_SERVER_PORT}/healthz" >/dev/null 2>&1; then
      ready=true
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "ERROR: agl-server exited" >&2
      exit 1
    fi
    sleep 1
  done
  if [ "$ready" != true ]; then
    echo "ERROR: server not healthy after 30s" >&2
    exit 1
  fi
  echo "  server ready: http://127.0.0.1:${AGL_SERVER_PORT}/healthz"
  wait "$SERVER_PID"
elif [ "$ROLE" = "controller" ]; then
  echo "=== OpenCode GRPO :: controller (k3s) ==="
  echo "  store: http://127.0.0.1:$AGL_SERVER_PORT"
  echo "  pods reach gateway at: $AGENT_URL"
  kubectl get nodes
  kubectl create namespace "$AGL_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n "$AGL_NAMESPACE" create configmap swe-smith-opencode-scripts \
    --from-file=smith_agent.py="$EXAMPLE_DIR/agents/smith_agent.py" \
    --from-file=opencode_agent.py="$EXAMPLE_DIR/agents/opencode_agent.py" \
    --dry-run=client -o yaml | kubectl -n "$AGL_NAMESPACE" apply -f -
  agl-controller \
    runner_type=k8s \
    agl_server.url="http://127.0.0.1:${AGL_SERVER_PORT}" \
    agl_server.agent_url="$AGENT_URL" \
    agl_server.key="$AGL_KEY" \
    k8s_runner.namespace="$AGL_NAMESPACE" \
    k8s_runner.ttl_after_finished=600
elif [ "$ROLE" = "trainer" ]; then
  echo "=== OpenCode GRPO :: trainer (4 GPU FSDP) ==="
  echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "  ckpt=$AGL_CKPT_DIR"
  if ! curl -sf "http://127.0.0.1:$AGL_SERVER_PORT/healthz" >/dev/null 2>&1; then
    echo "ERROR: server not reachable at 127.0.0.1:$AGL_SERVER_PORT" >&2
    exit 1
  fi
  export ATTN_IMPLEMENTATION=flash_attention_2
  # nvidia-smi topo is NV18 across all 8 GPUs. Disabling P2P forced a socket path that
  # NCCL 2.21 + H20 rejects with cuda invalid argument. Keep NVLink, skip IB/NVLS/cuMem.
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
  export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  # verl 0.7.1 AsyncLLM is V1; vLLM 0.8.5 errors if the env flag is left unset.
  export VLLM_USE_V1="${VLLM_USE_V1:-1}"
  export RAY_TMPDIR="${RAY_TMPDIR:-/workspace/rt}"
  mkdir -p "$RAY_TMPDIR"
  exec "$PWD/.venv/bin/python" "$EXAMPLE_DIR/train_opencode_agent.py" \
    --agl-base-url "http://127.0.0.1:$AGL_SERVER_PORT" \
    --agl-key "$AGL_KEY" \
    --train-dataset-path "$TRAIN_DATASET_PATH" \
    --val-dataset-path "$VAL_DATASET_PATH" \
    --model "$AGL_MODEL_NAME" \
    --run-name k3s \
    trainer.resume_mode=disable \
    "$@"
fi
