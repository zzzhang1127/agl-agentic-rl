#!/bin/bash
set -u
IMG=jyangballin/swesmith.x86_64.bottlepy_1776_bottle.a8dfef30
LOG=/workspace/agl-checkpoints/swe_smith_smoke/pull.log
exec > >(tee -a "$LOG") 2>&1
echo "pull start $(date) $IMG"
if docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "already local"
  exit 0
fi
if docker pull "$IMG"; then
  echo "pulled $IMG"
  exit 0
fi
echo "direct pull failed, trying docker.1ms.run"
if docker pull "docker.1ms.run/$IMG"; then
  docker tag "docker.1ms.run/$IMG" "$IMG"
  echo "pulled via docker.1ms.run"
  exit 0
fi
echo "trying docker.m.daocloud.io"
if docker pull "docker.m.daocloud.io/$IMG"; then
  docker tag "docker.m.daocloud.io/$IMG" "$IMG"
  echo "pulled via daocloud"
  exit 0
fi
echo "PULL FAILED"
exit 1
