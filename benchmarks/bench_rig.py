#!/usr/bin/env python3
"""Rig complet benchmark 1Cat-vLLM pe 4xV100 (stdlib only).

Suites:
  quick    pp512/pp1024/tg128/tg256/chat (regresii rapide)
  target   8K/32K/64K/128K + 512 out (target-only matrix, Etapa 2)
  long     needle tests + near-limit (Etapa 5)
  quality  matematica/cod/romana/engleza (Etapa 15)
  det      determinism temp=0 x10 (Etapa 7/15)
  custom   --tests pp8192,tg256,chat,needle:32768:50
"""
import argparse, json, statistics as st, subprocess, sys, time, urllib.request, urllib.error

BASE_PROMPT = ("Ingineria performantei pe sisteme de inferenta locala necesita masurare "
    "riguroasa, nu presupuneri. Fiecare schimbare de configurare trebuie validata prin "
    "teste repetate, cu incalzire si cu statistici pe mai multe rulari. ")

NEEDLE = "CODUL SECRET AL LABORATORULUI ESTE QX-7421-VOLTA"
HAY_FACT = ("Faptul numarul {i} din arhiva tehnica: sistemul de racire foloseste {m} module "
    "de circulatie, iar temperatura nominala de operare este de {t} grade Celsius. ")

MATH_TESTS = [
    ("Cat este 17 inmultit cu 23? Raspunde doar cu numarul.", "391"),
    ("Cat este 144 impartit la 12? Raspunde doar cu numarul.", "12"),
    ("Suma primelor 10 numere naturale nenule? Raspunde doar cu numarul.", "55"),
    ("Daca un tren merge 60 km/h timp de 2.5 ore, cati km parcurge? Raspunde doar cu numarul.", "150"),
]
CODE_TEST = ("Scrie o functie Python nume `suma_patrate` care primeste o lista de intregi si "
    "returneaza suma patratelor elementelor pozitive. Raspunde DOAR cu blocul de cod, fara explicatii.")
RO_TEST = ("Explica intr-o propozitie ce este un cache KV la un model de limbaj. Raspunde in romana.")
EN_TEST = ("Explain in one sentence what speculative decoding is. Answer in English.")


_TPR = {"base": None, "hay": None}

def _tokenize(base, key, text):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps({"model": "qwen3.8-flash-next", "prompt": text}).encode()
    req = urllib.request.Request(base.rstrip("/").replace("/v1", "/v1") + "/tokenize",
                                 data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return len(json.load(r)["tokens"])

def calibrate(base, key):
    _TPR["base"] = max(1, _tokenize(base, key, BASE_PROMPT))
    _TPR["hay"] = max(1, _tokenize(base, key, HAY_FACT.format(i=0, m=3, t=22)))

def approx_prompt(n):
    reps = max(1, round(n / _TPR["base"]))
    return " ".join([BASE_PROMPT] * reps)


def haystack(target_tokens, depth_pct):
    # construieste haystack ~target_tokens tokeni cu needle la depth_pct%
    tpr = _TPR["hay"] or 40
    target_facts = max(1, int(target_tokens / tpr))
    facts = []
    for i in range(target_facts):
        facts.append(HAY_FACT.format(i=i, m=2 + (i % 7), t=18 + (i % 14)))
    pos = max(1, int(len(facts) * depth_pct / 100))
    facts.insert(pos, NEEDLE + ". ")
    intro = ("Mai jos este o arhiva tehnica lunga. La final iti voi pune o intrebare despre un "
             "cod secret mentionat undeva in text.\n\n")
    outro = ("\n\nIntrebare: care este codul secret al laboratorului mentionat in arhiva? "
             "Raspunde doar cu codul exact.")
    return intro + " ".join(facts) + outro


def chat_req(base, key, model, messages, max_tokens, temp, ignore_eos=False, timeout=3600):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temp, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if ignore_eos:
        body["ignore_eos"] = True
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    text = ""
    chunk_times = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            content = ""
            for c in (chunk.get("choices") or []):
                content += c.get("delta", {}).get("content") or ""
            if content:
                now = time.perf_counter() - t0
                chunk_times.append(now)
                if ttft is None:
                    ttft = now
                text += content
    total = time.perf_counter() - t0
    ttft = ttft if ttft is not None else total
    ptok = usage.get("prompt_tokens", 0)
    ctok = usage.get("completion_tokens", 0)
    dec = max(total - ttft, 1e-6)
    # TPOT = mediana gap-urilor intre chunk-urile de continut
    tpot_ms = None
    if len(chunk_times) >= 3:
        gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:]) if b > a]
        if gaps:
            gaps.sort()
            tpot_ms = round(gaps[len(gaps) // 2] * 1000, 2)
    decode_tok_s = round(1000 / tpot_ms, 1) if tpot_ms and tpot_ms > 0 else (round(ctok / dec, 1) if ctok else None)
    return {"prompt_tokens": ptok, "completion_tokens": ctok, "latency_s": round(total, 3),
            "ttft_s": round(ttft, 3),
            "prefill_tok_s": round(ptok / ttft, 1) if ptok else None,
            "decode_tok_s": decode_tok_s,
            "tpot_ms": tpot_ms,
            "text_head": text[:120], "text_tail": text[-120:]}


def run_test(base, key, model, spec, warm, iters):
    name = spec.split(":", 1)[0]
    if name.startswith("pp"):
        n = int(name[2:])
        mk = lambda: chat_req(base, key, model, [{"role": "user", "content": approx_prompt(n)}], 1, 0)
    elif name.startswith("tg"):
        n = int(name[2:])
        mk = lambda: chat_req(base, key, model, [{"role": "user", "content": "Continua povestea unui calator prin munti."}], n, 0.3, ignore_eos=True)
    elif name == "chat":
        mk = lambda: chat_req(base, key, model, [{"role": "user", "content":
            "Explica-mi pe scurt, in trei paragrafe, cum functioneaza decodarea speculativa MTP."}], 300, 0.3)
    elif name == "needle":
        _, depth, pct = spec.split(":")
        mk = lambda: chat_req(base, key, model, [{"role": "user", "content": haystack(int(depth), int(pct))}], 64, 0)
    elif name == "nearlimit":
        n = int(spec.split(":")[1])
        mk = lambda: chat_req(base, key, model, [{"role": "user", "content": approx_prompt(n)}], 512, 0)
    else:
        raise SystemExit(f"test necunoscut: {spec}")
    results = []
    for i in range(warm):
        r = mk(); r["phase"] = "warm"; results.append(r)
    for i in range(iters):
        r = mk(); r["phase"] = "measured"; results.append(r)
        speed = r["prefill_tok_s"] if name.startswith("pp") or name in ("nearlimit",) else r["decode_tok_s"]
        print(f"  [{spec}] run{i+1}: pt={r['prompt_tokens']} ct={r['completion_tokens']} "
              f"tok/s={speed} ttft={r['ttft_s']}s", flush=True)
    return results


def quality_suite(base, key, model):
    out = []
    for q, expect in MATH_TESTS:
        r = chat_req(base, key, model, [{"role": "user", "content": q}], 32, 0)
        ok = expect in (r["text_head"] + r["text_tail"])
        out.append({"q": q[:60], "expect": expect, "pass": ok, "answer": r["text_head"][:80]})
        print(f"  [math] {expect}: {'PASS' if ok else 'FAIL'} — {r['text_head'][:60]}", flush=True)
    r = chat_req(base, key, model, [{"role": "user", "content": CODE_TEST}], 400, 0)
    code = ""
    if "```" in (r["text_head"] + r["text_tail"]) or "def " in (r["text_head"] + r["text_tail"]):
        full = r["text_head"] + "..." + r["text_tail"]
        code = full
    # re-cerem fara stream simplu: rulam doar daca extragem bloc curat — aici doar verificam forma
    ok = "def suma_patrate" in code.replace("...", "") or "def suma_patrate" in code
    out.append({"q": "code suma_patrate", "pass": bool(ok)})
    print(f"  [code] forma: {'PASS' if ok else 'CHECK MANUAL'}", flush=True)
    for nm, q in (("ro", RO_TEST), ("en", EN_TEST)):
        r = chat_req(base, key, model, [{"role": "user", "content": q}], 150, 0.3)
        out.append({"q": nm, "answer": r["text_head"][:150]})
        print(f"  [{nm}] {r['text_head'][:100]}", flush=True)
    return out


def det_suite(base, key, model, tokens=128):
    outs = []
    for i in range(10):
        r = chat_req(base, key, model, [{"role": "user", "content": "Explica cache-ul KV pe scurt."}], tokens, 0)
        outs.append(r["text_head"] + r["text_tail"])
    uniq = len(set(outs))
    print(f"  [det] {uniq}/10 outputuri unice (1 = perfect determinist)", flush=True)
    return {"unique_outputs": uniq, "samples": [o[:80] for o in outs[:3]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--suite", default="quick", choices=["quick", "target", "long", "quality", "det", "custom"])
    ap.add_argument("--tests", default="")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    calibrate(args.base_url.rsplit("/v1", 1)[0] if args.base_url.endswith("/v1") else args.base_url, args.api_key)
    print(f"calibrare: BASE={_TPR['base']} tok/rep, HAY={_TPR['hay']} tok/fact", flush=True)

    suites = {
        "quick": ["pp512", "pp1024", "tg128", "tg256", "chat"],
        "target": ["pp8192", "pp32768", "pp65536", "pp131072", "tg256", "chat"],
        "long": ["needle:32768:10", "needle:32768:50", "needle:32768:90",
                 "needle:65536:50", "needle:131072:50", "needle:196608:50"],
    }
    tests = args.tests.split(",") if args.tests else suites.get(args.suite, [])

    import os
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.tag}.jsonl")
    summary = {"tag": args.tag, "suite": args.suite, "tests": {}}

    if args.suite == "quality":
        res = quality_suite(args.base_url, args.api_key, args.model)
        summary["quality"] = res
    elif args.suite == "det":
        res = det_suite(args.base_url, args.api_key, args.model)
        summary["det"] = res
    else:
        with open(path, "a") as f:
            for spec in tests:
                name = spec.split(":")[0]
                try:
                    rs = run_test(args.base_url, args.api_key, args.model, spec, args.warm, args.iters)
                except urllib.error.HTTPError as e:
                    body = e.read()[:300]
                    print(f"  [{spec}] HTTP {e.code}: {body[:200]}", flush=True)
                    rs = [{"error": e.code, "body": body.decode("utf-8", "replace")[:300]}]
                for r in rs:
                    r2 = dict(r); r2["spec"] = spec
                    f.write(json.dumps(r2, ensure_ascii=False) + "\n")
                meas = [r for r in rs if r.get("phase") == "measured" and "error" not in r]
                if meas:
                    key = "prefill_tok_s" if name.startswith("pp") or name == "nearlimit" else "decode_tok_s"
                    vals = [r[key] for r in meas if r.get(key)]
                    if vals:
                        summary["tests"][spec] = {"mean": round(st.mean(vals), 1),
                                                  "stdev": round(st.stdev(vals), 1) if len(vals) > 1 else 0,
                                                  "min": min(vals), "max": max(vals),
                                                  "prompt_tokens": meas[0]["prompt_tokens"]}
                    if name == "needle":
                        hit = any(NEEDLE.split("ESTE ")[1] in (r.get("text_head", "") + r.get("text_tail", "")) for r in meas)
                        summary["tests"][spec]["needle_pass"] = hit
                        print(f"  [needle:{spec}] {'PASS' if hit else 'FAIL'}", flush=True)

    spath = os.path.join(args.outdir, f"{args.tag}-summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"REZULTAT {args.tag}: scris {path} + {spath}")


if __name__ == "__main__":
    main()
