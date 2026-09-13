# Concurrency on 4×V100 (CONC-B / E18)

**Question:** what does `max_num_seqs` actually buy on this box, and what does
it cost single-stream performance?

**Short answer:** with `max-num-seqs 4`, up to 4 concurrent requests share the
GPU with per-stream speed degrading 46.0 → 33.5 → 30.9 → 30.5 tok/s, aggregate
throughput rising to **122 tok/s (+165%)** — while a lone request keeps the
full 46.0 tok/s. The tolerance is free.

## Rig

`benchmarks/bench_conc.py` — stdlib-only; N worker threads released
simultaneously by a `threading.Barrier`, each issuing the identical tg-style
request (23-token prompt, 256 output tokens, temp 0.3, `ignore_eos`); decode
speed measured from SSE chunk gaps (median TPOT) cross-checked with real
`usage` completion tokens — the same measurement protocol as every other
stage in this repo.

The test server was an **exact mirror of the effective production
environment** (all three PR #415 routes on, P2P restricted to NVLink), at
MML 65536, `max-num-seqs 4`. Control: n=1 on the test server = 46.0 tok/s —
reproduces production, so the matrix is directly applicable.

## Results (warm, n=3, stdev < 0.2 tok/s)

| Streams | per-stream tok/s | aggregate tok/s | vs single | mean TTFT |
|---|---|---|---|---|
| 1 | 46.0 | 46.0 | 1.00× | 0.33 s |
| 2 | 33.5 | 67.1 | 1.46× | 0.46 s |
| 3 | 30.9 | 92.8 | 2.02× | 0.55 s |
| 4 | 30.5 | 122.0 | 2.65× | 0.58 s |

## Reading

- **Single-stream is not degraded by `max-num-seqs 4`.** 46.0 measured on the
  MNS=4 server itself. (An earlier interrupted test, CONC-A, had verified the
  same for MNS=2 at 41.5 in the pre-PR#415 era.)
- **Aggregate scaling is near-linear to 3 streams** (2.02×) and sub-linear at
  4 (2.65× vs ideal 4×). Per-stream saturates around 30–31 tok/s at n≥3 —
  batching finally feeds the GPU (single-stream ran at ~110 W of 250 W TDP).
- **TTFT stays under a second** at every level for short prompts — no
  latency surprise for interactive use.

## Decision & caveats

Production moved to `--max-num-seqs 4` (systemd drop-in + fast launcher).

- **n > 4 untested.** Measure before raising it (baseline = this document).
- Real agent traffic brings multi-K-token system prompts; concurrent prefill
  shares compute, so the real-world aggregate may differ from this short-prompt
  matrix. Re-measure if it matters.
- Post-deploy smoke on the production box (MML 262144, MNS=4): single-stream
  44.5–45.2 tok/s on a freshly started server (still warming; the stabilized
  number is 46), 2-stream aggregate 62.7–65.2.

## Artifacts

`benchmarks/results/CONC-B-n{1,2,3,4}.jsonl` (one line per stream per round),
`CONC-B-summary.json`, `CONC-A-summary.json`, and `PROD-MNS4-SMOKE*` from the
post-deploy verification.
