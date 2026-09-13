#!/usr/bin/env python3
"""Benchmark concurenta 1Cat-vLLM: N streamuri paralele (stdlib only).

Foloseste chat_req din bench_rig (acelasi protocol de masurare: usage real +
TPOT din chunk-uri). Fiecare runda lanseaza N threaduri pornite simultan
(bariera), fiecare cu acelasi tip de request tg (prompt scurt, ignore_eos).

Iesire: <outdir>/<tag>-n<N>.jsonl (o linie per stream per runda) +
        <outdir>/<tag>-n<N>-summary.json
"""
import argparse, json, os, statistics as st, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_rig import chat_req  # noqa: E402

PROMPT = "Continua povestea unui calator prin munti."


def run_round(base, key, model, nstreams, max_tokens, temp):
    results = [None] * nstreams
    barrier = threading.Barrier(nstreams)

    def worker(i):
        barrier.wait()
        results[i] = chat_req(base, key, model,
                              [{"role": "user", "content": PROMPT}],
                              max_tokens, temp, ignore_eos=True)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(nstreams)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    dec = [r["decode_tok_s"] for r in results if r.get("decode_tok_s")]
    ttft = [r["ttft_s"] for r in results if r.get("ttft_s") is not None]
    ctoks = [r["completion_tokens"] for r in results]
    agg = round(sum(dec), 1) if dec else None
    return {
        "nstreams": nstreams,
        "wall_s": round(wall, 3),
        "decode_per_stream_mean": round(st.mean(dec), 1) if dec else None,
        "decode_per_stream_min": min(dec) if dec else None,
        "decode_per_stream_max": max(dec) if dec else None,
        "decode_agg_tok_s": agg,
        "ttft_mean_s": round(st.mean(ttft), 3) if ttft else None,
        "completion_tokens_total": sum(ctoks),
        "streams": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--levels", default="1,2,3,4")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    summary = {"tag": args.tag, "levels": {}}

    for n in levels:
        # warm: o singura runda N-stream (nu masurata)
        for w in range(args.warm):
            run_round(args.base_url, args.api_key, args.model, n, args.tokens, args.temp)
            print(f"[n={n}] warm {w+1}/{args.warm} ok", flush=True)
        rounds = []
        for i in range(args.iters):
            r = run_round(args.base_url, args.api_key, args.model, n, args.tokens, args.temp)
            rounds.append(r)
            print(f"[n={n}] run{i+1}: per-stream={r['decode_per_stream_mean']} tok/s "
                  f"(min {r['decode_per_stream_min']}, max {r['decode_per_stream_max']}) "
                  f"agg={r['decode_agg_tok_s']} tok/s ttft={r['ttft_mean_s']}s", flush=True)
        aggs = [r["decode_agg_tok_s"] for r in rounds if r["decode_agg_tok_s"]]
        pers = [r["decode_per_stream_mean"] for r in rounds if r["decode_per_stream_mean"]]
        summary["levels"][str(n)] = {
            "per_stream_mean": round(st.mean(pers), 1) if pers else None,
            "agg_mean_tok_s": round(st.mean(aggs), 1) if aggs else None,
            "agg_min": min(aggs) if aggs else None,
            "agg_max": max(aggs) if aggs else None,
        }
        with open(os.path.join(args.outdir, f"{args.tag}-n{n}.jsonl"), "a") as f:
            for rd in rounds:
                f.write(json.dumps(rd, ensure_ascii=False) + "\n")

    spath = os.path.join(args.outdir, f"{args.tag}-summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"REZULTAT {args.tag}: {spath}")


if __name__ == "__main__":
    main()
