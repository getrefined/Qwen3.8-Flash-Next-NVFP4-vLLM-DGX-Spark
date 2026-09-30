#!/bin/bash
# Qwen3.8-Flash-Next (RadixArk NVFP4) on 2x DGX Spark via vLLM.
# TP2 + EP, MTP3 speculative decode, FULL_DECODE_ONLY CUDA graphs, 262K context.
#
# Two modes:
#   default      official day-0 image vllm/vllm-openai:qwen38-flash-next. Requires ple_layer.py
#                extracted from the image with ple-force-fp8.patch applied, placed next to this
#                script (bind-mounted over the image's copy at runtime).
#   NIGHTLY=1    a vLLM nightly from 2026-09-03 or later (PLE fix upstream, vllm#54882): no patch,
#                PLE offload forced off (--engram-config.cpu_offload false; on a Spark host and GPU
#                memory are the same LPDDR, so offload only shrinks the KV pool). Pin the nightly
#                by digest/tag via IMAGE=... for reproducible numbers.
#
# FP8-dense build (make-fp8-dense.py): NIGHTLY=1 MODEL=/models/<dir> (~/models is mounted ro).
#
# Usage:  ./launch-vllm-fn.sh 1   # worker first
#         ./launch-vllm-fn.sh 0   # then head (serves :8000)
# Env:    HEAD_IP, WORKER_IP, IFACE, IB_HCA must match YOUR fabric (see below).
#         NIGHTLY, IMAGE, MODEL, SEQS, GMU, AUTOTUNE (0 = --no-enable-flashinfer-autotune).
set -euo pipefail
NODE_RANK="${1:?usage: launch-vllm-fn.sh <0|1>}"

NIGHTLY="${NIGHTLY:-0}"
if [ "$NIGHTLY" = "1" ]; then IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"; else IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"; fi
NAME="vllm-fn"
MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"

# ---- fabric: EDIT THESE for your pair ----------------------------------------
HEAD_IP="${HEAD_IP:-192.168.100.10}"       # rank-0 node's inter-node IP
WORKER_IP="${WORKER_IP:-192.168.100.11}"   # rank-1 node's inter-node IP
IFACE="${IFACE:-enp1s0f0np0}"              # ip -o -4 addr | grep <subnet>  (mind np0/np1 suffixes)
IB_HCA="${IB_HCA:-=rocep1s0f0}"            # EXACT-MATCH single device. Multi-pair fleets:
                                           # never list a port cabled to another cluster.
# ------------------------------------------------------------------------------
MPORT="50000"; PORT=8000
case "$NODE_RANK" in
  0) HOST_IP="$HEAD_IP";   EXTRA="--host 0.0.0.0 --port $PORT" ;;
  1) HOST_IP="$WORKER_IP"; EXTRA="--headless" ;;
esac
SEQS="${SEQS:-8}"; GMU="${GMU:-0.80}"
MODEARGS=(); SERVEARGS=()
if [ "$NIGHTLY" = "1" ]; then
  SERVEARGS+=(--engram-config.cpu_offload false)
else
  PATCHED="$(cd "$(dirname "$0")" && pwd)/ple_layer_patched.py"
  test -f "$PATCHED" || { echo "missing $PATCHED (extract from image + apply ple-force-fp8.patch)"; exit 1; }
  MODEARGS+=(-e PLE_FORCE_FP8=1 -v "$PATCHED":/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro)
  [ "${PLE_OFFLOAD:-0}" = "1" ] && MODEARGS+=(-e VLLM_PLE_CPU_OFFLOAD=1)
fi
[ "${AUTOTUNE:-1}" = "0" ] && SERVEARGS+=(--no-enable-flashinfer-autotune)
mkdir -p ~/models

docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p ~/.cache/vllm
docker run -d --name "$NAME" --gpus all --network host --ipc host \
  --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e GLOO_SOCKET_IFNAME="$IFACE" -e NCCL_SOCKET_IFNAME="$IFACE" -e TP_SOCKET_IFNAME="$IFACE" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" \
  -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_AUTO_DETECT=0 -e NCCL_DEBUG=WARN \
  "${MODEARGS[@]}" \
  -v ~/models:/models:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  "$IMAGE" \
  "$MODEL" \
    --served-model-name qwen3.8-flash-next \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --all2all-backend allgather_reducescatter \
    --load-format safetensors --safetensors-load-strategy lazy \
    --max-model-len 262144 \
    --max-num-seqs "$SEQS" \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization "$GMU" \
    --enable-chunked-prefill \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    "${SERVEARGS[@]}" $EXTRA
echo "launched $NAME rank=$NODE_RANK host=$HOST_IP image=$IMAGE model=$MODEL (seqs=$SEQS gmu=$GMU nightly=$NIGHTLY)"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || echo "WARNING: $NAME not running — docker logs $NAME" >&2
