# External Consult — open questions (2026-09-13)

Snapshot of the questions handed to external review after the optimization
program, with the corrections that review already produced. Kept here because
several are still open and interesting to anyone reproducing this stack.

## Corrections the review already forced (merged into this repo)

1. **"41 tok/s = hardware ceiling" was premature.** Three opt-in fast routes
   from PR #415 (all default-OFF upstream) had never been activated; with
   them: **46 tok/s (+13%)**. See the main README.
2. **CPU-swap estimate revised down** from +40–75% to **+5–15% decode**. Two
   pieces of evidence: MTP4 scaling is purely GPU-bound per-pass (~24.4 ms),
   and NUMA pinning (`numactl`) moved nothing. CPU gain stays real for
   tokenization (3–5× on 256K prompts) and general responsiveness.
3. **Concurrency test completed** (CONC-B): production now serves
   `max-num-seqs 4`.

## Still open

1. **The 1Cat reference** (65.9 tok/s control / 80.7 optimized / 138 MTP4 on
   4×V100): measured on what rig? Native SXM2 with full NVLink mesh? Modern
   CPU? Prompt length and methodology? Our identical-GPU number is 46.
2. **PLE over PCIe**: every token does a lookup into ~52 GB of FP8 PLE tables
   pinned in RAM. How much per-token latency does that add? Anyone measured
   `VLLM_PLE_CPU_OFFLOAD` on/off at decode? (Partial GPU residency — hot
   bigrams — is theoretically possible but unimplemented.)
3. **EP instead of TP4**: expert-parallelism across the NVLink pairs (EP=2)
   with TP=2 inside each pair would eliminate cross-pair all-reduce for MoE
   layers. Has anyone run 1Cat SM70 that way?
4. **PR #398** (MTP verifier cost, merged ~1 week after the v1.5.0 tag): how
   much does it buy on a weak CPU? Does it justify a 1–2 h CUDA toolchain
   build?
5. **SXM2-reflashed-on-PCIe**: known limitations vs native PCIe V100 (PCIe
   generation, clock management)?
6. **CPU swap E5-v2 → modern**, same vLLM config: does anyone have
   measurements confirming the +5–15% decode / 3–5× tokenize estimate?

## Methodology note

All our numbers: own rig, warm server, n=3, decode from median SSE-chunk gap
(TPOT) cross-checked with `usage.completion_tokens`, prefill from
`usage.prompt_tokens / TTFT`. Short prompts (23 tok) for decode; unique-content
haystacks for long-context; `ignore_eos` for sustained decode. Determinism
verified at temp=0 (10/10 identical outputs).
