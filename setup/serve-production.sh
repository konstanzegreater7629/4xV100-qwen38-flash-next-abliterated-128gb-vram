#!/bin/bash
# Production launcher — Qwen3.8-Flash-Next-ABLITERATED-NVFP4 on 4x V100 (TP4)
# 1Cat-vLLM 1.5.0. This mirrors the authoritative systemd drop-in.
#
# API key is read from $VLLM_API_KEY (or ~/.config/vllm/apikey.env if present).
# The server is exposed on 0.0.0.0:8000 — protect it with a key and/or a
# firewall; do not leave it open to the internet unauthenticated.
set -euo pipefail
cd /mnt/nvme/1cat-vllm
source venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4   # indices of the four V100s in nvidia-smi
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLE_CPU_OFFLOAD=1         # PLE tables (~52 GB FP8) pinned in RAM — official SM70 route
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL             # P2P only on NVLink; cross-pair falls back to SHM (PLX P2P deadlocks)
export HF_HOME=/mnt/nvme/huggingface-cache
# PR #415 opt-in fast routes (all default-OFF upstream; +13% decode on this contract):
export VLLM_SM70_QWEN38_FP16_GEMV=1
export VLLM_SM70_QWEN38_FUSED_HC_FP16=1
export VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=1
# Triton compiles a C launcher on first run and needs Python.h; if the system
# has no python3-dev, point these at any ABI-compatible conda python package:
# export CPATH=/path/to/python-3.12.x/include/python3.12
# export LIBRARY_PATH=/path/to/python-3.12.x/lib

KEY="${VLLM_API_KEY:-}"
if [ -z "$KEY" ] && [ -f "$HOME/.config/vllm/apikey.env" ]; then
  set -a; source "$HOME/.config/vllm/apikey.env"; set +a
  KEY="${VLLM_API_KEY:-}"
fi
[ -n "$KEY" ] || { echo "set VLLM_API_KEY"; exit 1; }

exec vllm serve /mnt/nvme/models/dealignai-abliterated \
  --served-model-name qwen3.8-flash-next \
  --trust-remote-code --language-model-only --dtype half \
  --tensor-parallel-size 4 --attention-backend FLASH_ATTN_V100 \
  --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 --host 0.0.0.0 --port 8000 \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --api-key "$KEY"
