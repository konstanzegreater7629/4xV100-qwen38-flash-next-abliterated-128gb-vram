#!/bin/bash
# FAST launcher — same speed as production, small max-model-len for a quick
# boot and a larger effective KV pool per concurrent session.
# Usage:  MML=65536 bash serve-fast.sh
set -euo pipefail
MML="${MML:-65536}"
cd /mnt/nvme/1cat-vllm
source venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLE_CPU_OFFLOAD=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export HF_HOME=/mnt/nvme/huggingface-cache
export VLLM_SM70_QWEN38_FP16_GEMV=1
export VLLM_SM70_QWEN38_FUSED_HC_FP16=1
export VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=1

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
  --max-model-len "$MML" --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 --host 0.0.0.0 --port 8000 \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --api-key "$KEY"
