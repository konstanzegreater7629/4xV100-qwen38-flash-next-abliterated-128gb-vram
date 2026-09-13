# The 18-Stage Optimization Program (E0–E17 + E18/CONC-B)

**Model:** dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `@be794b990578`
**Engine:** 1Cat-vLLM 1.5.0 (file-identical to upstream tag v1.5.0 — zero diff;
PRs #345/#389/#415 included; **#398 merged after the tag and is NOT in this build**)
**Methodology, strict:** baseline → one change → measure → KEEP/REVERT.
All measurements on a private rig (warm, n=3, decode from SSE-chunk TPOT +
real `usage`, prefill from `usage`).

## Hardware (verified, not assumed)

| Component | Detail |
|---|---|
| GPU | 4× NVIDIA Tesla V100-SXM2-32GB (**SXM2 reflashed on PCIe**), SM70 |
| Topology | GPU1↔GPU2 NVLink **NV5** (5 bonds), GPU3↔GPU4 NVLink **NV6** (6 bonds), **between pairs = PCIe through PLX (PIX)**; GPU0 = Quadro P400 (display, excluded) |
| CPU | 2× Intel Xeon E5-2680 v2 (Ivy Bridge, 2013, 2.8 GHz, 40 threads) |
| RAM | 251 GB DDR3 ECC, dual-socket NUMA (all GPUs on node 0) |
| Storage | 1 TB NVMe (venv, checkpoint, artifacts) |
| Driver / OS | 580.173.02 · Ubuntu 24.04 |

## Stage log

### E0 — audit (KEEP: knowledge)
Reconstructed the git history of the whole system. Critical discovery: the
**authoritative configuration lived in a systemd drop-in**
(`disable-custom-ar.conf`), not in the main unit — earlier edits to the unit
had been no-ops. Also found 3 dead env vars from the 1.2.2 era
(`SKINNY_NVFP4`, `GDN_CHAIN_SPEC_FAST_BUILD`, `QPN8_MT2`) that don't exist in
1.5.0 — removed everywhere.

### E1 — model verification (KEEP)
Verified the checkpoint: ModelOpt NVFP4 (`quant_algo=NVFP4`, routed experts
512/top-10, group 16), PLE FP8, **MTP complete 31/31 tensors**,
`max_position_embeddings` 262144. Structure byte-identical to the stock
RadixArk checkpoint (config diff = 0). Of 5 candidate repacks studied, the only
drop-in compatible one.

### E2 — target-only baseline (KEEP)
MTP OFF: decode 41.0 tok/s (tg128/tg256/chat identical), KV pool
**439,700 tokens** (vs ~217K with MTP ON — the draft copy eats ~3.5 GiB/GPU).

### E3 — where the ceiling is (KEEP: conclusion; numactl REVERT)
`numactl -N 0` NUMA pinning: **0% difference** → REVERT.
GPUs at 1530 MHz full boost, 92–99% util, ~110 W/250 W, 46–56 °C — no
throttling. Verdict: the ~41 tok/s ceiling is host-bound (2013 Xeon
split-ops/piecewise-eager + cross-pair all-reduce through SHM), **not**
GPU-bound. *(Later corrected: part of the ceiling was actually unactivated
fast routes — see PR #415.)*

### E4 — GMU 0.92 (KEEP)
0.94/0.95 documented OOM risk on this model family (27B OOM at prefill 48K
with 0.95). 0.92 is the sweet spot: 22.0 GiB weights + 5.51 GiB KV per GPU.

### E5 — max-model-len 262,144 (KEEP)
Validated functionally, not just accepted: needle-in-haystack **6/6 PASS**
(32K/64K/128K/196K × depths 10/50/90%), real 256,512-token prompt + reply.
Decode on long context: 40.6 tok/s @32K → 37.7 @196K → 35.7 @256K.
Prefill (unique content, cold): ~1,100 tok/s @32K, 2,123 @139K, 2,335 @209K,
1,631 @256K.

### E6–E8 — FP8 E4M3 KV cache (REJECT, unimplemented)
Ported-in analysis of upstream vLLM PR #54426 for the QSA indexer
(integration points located in 1Cat: `qsa.py`, `ops/qsa.py`). **Rejected
without implementing**: it would disable the fast SM70 path `xqa_page4`
(gated on fp16 KV) — and 262K is already reached in fp16 with 177K headroom.
User directive: no speed compromises. Reference patch archived at
[`../patches/fp8-qsa-kv-upstream-reference.patch`](../patches/fp8-qsa-kv-upstream-reference.patch).

### E9 — MTP depth 4 (REVERT)
MTP4: **18.2 tok/s vs 41 target-only**. Acceptance length 2.17–2.20 of 5;
696 drafted → 204 accepted. Per-pass scaling: 18.2 ≈ 41 ÷ 2.25 → ~2.3 forward
passes per accepted token → each forward pass ≈ **24.4 ms, GPU-bound**.
Conclusion: MTP draft cost is CPU-serialized on the 2013 Xeon and exceeds the
verify gain at every depth on this host. MTP (any depth) is harmful here.
The published 138 tok/s MTP4 number is unreachable on this CPU+topology
combination.

### E10–E11 — all-reduce & CUDA-graph audit (conclusions)
Custom all-reduce livelocks on the PLX bridge (spin-wait at 100% GPU — lived
through it on an earlier session; `--disable-custom-all-reduce` stays).
NCCL with `NCCL_P2P_LEVEL=NVL` is the best available without a CUDA build;
the hierarchical 2+2 all-reduce (PR #398) is not in the 1.5.0 tag.
CUDA-graph policy (capture [1,2] target-only B1) already correct.

### E12 — BFLA sparse prefill (omitted)
Experimental, quality risk on a production LONG server.

### E13 — BALANCED profile (omitted)
Single optimum exists (FAST = LONG with small MML); no balanced tier needed.

### FINAL (pre-CONC-B) — served config
`vllm serve … --tensor-parallel-size 4 --attention-backend FLASH_ATTN_V100
--max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 8192
--gpu-memory-utilization 0.92 --disable-custom-all-reduce
--enable-auto-tool-choice --tool-call-parser qwen3_coder` (MTP OFF, fp16 KV).

Validation: decode 41.0/40.8/41.2, math 4/4, code, RO/EN PASS, determinism
10/10 identical outputs at temp=0.

## Post-program findings

### PR #415 flags (+13% decode, KEEP) — the missed ceiling
External review caught what E3 missed: three **opt-in** fast routes, all
default-OFF upstream (contract-gated: TP4 + no speculative decoding):

```bash
export VLLM_SM70_QWEN38_FP16_GEMV=1
export VLLM_SM70_QWEN38_FUSED_HC_FP16=1
export VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=1
```

Boot log confirms: `288 GEMV projections`, `96 HC modules`, GDN input route
enabled. Result: **41.0 → 46.2 tok/s** (+13%), quality PASS, determinism
10/10. The "41 = hardware ceiling" conclusion from E3 was premature.

### E18 / CONC-B — concurrency (KEEP, max_num_seqs=4)
Full log: [CONCURRENCY.md](CONCURRENCY.md). Summary: 2 streams 33.5/stream
(agg 67.1), 3 → 30.9 (agg 92.8), 4 → 30.5 (agg 122.0); single-stream 46.0
**unchanged on the MNS=4 server** → production moved to `max-num-seqs 4`.

## Honest limitations (carried forward)

1. Decode 46 tok/s « published 65.9/80.7/138 — host-bound (CPU + PCIe/PLX),
   not GPU-bound (110 W, 92–99% util). Next real lever: PR #398 build or CPU swap.
2. MTP unusable with gain — same reason.
3. PR #398 not ported (needs 1–2 h CUDA toolchain build; risk noted).
4. Tokenizing 256K prompts is CPU-bound: 60–90 s per cold request.
5. NVLink traffic counters read N/A on driver 580 (monitoring gap, cosmetic).

## Reproducibility

Every number traces to raw data in
[`../benchmarks/results/`](../benchmarks/results/): E2 (target-only),
E9 (MTP4), E5 (long context), PR415 (flags + quality + determinism),
CONC-A (MNS=2 single-stream check), CONC-B (concurrency matrix),
PROD-MNS4-SMOKE (post-deploy verification).
