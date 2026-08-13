#!/usr/bin/env python3
"""A/B bench: VRouter combo routing vs direct 9router — after dead-model tuning."""
import json, time, urllib.request, urllib.error, sys

ROUTER = "http://127.0.0.1:20129/v1/chat/completions"
NINE = "http://127.0.0.1:20128/v1/chat/completions"
KEY = "hermes-router-2026"

PROMPTS = [
    "Hitung 17*23, jawab singkat.",
    "Tulis satu baris Python untuk balik string.",
    "Apa ibu kota Indonesia? Satu kata.",
]

def call(base, model, prompt, timeout=70):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 80}).encode()
    req = urllib.request.Request(base, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        ms = (time.time() - t0) * 1000
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return {"ok": True, "ms": ms, "len": len(content), "model": data.get("model", "?")}
    except Exception as e:
        return {"ok": False, "ms": (time.time() - t0) * 1000, "err": str(e)[:100]}

print("=== VRouter combo JEROUTER (score-weighted routing) ===")
for i, p in enumerate(PROMPTS):
    r = call(ROUTER, "JEROUTER", p)
    status = f"OK ({r['ms']:.0f}ms, {r['len']}ch, {r.get('model','')})" if r["ok"] else f"FAIL {r.get('err','')} ({r['ms']:.0f}ms)"
    print(f"  [{i+1}] {status}")

print("\n=== VRouter combo CUPANG (ipeenk) ===")
for i, p in enumerate(PROMPTS[:2]):
    r = call(ROUTER, "CUPANG", p)
    status = f"OK ({r['ms']:.0f}ms, {r['len']}ch, {r.get('model','')})" if r["ok"] else f"FAIL {r.get('err','')} ({r['ms']:.0f}ms)"
    print(f"  [{i+1}] {status}")

print("\n=== 9router direct JEROUTER combo (baseline) ===")
for i, p in enumerate(PROMPTS[:2]):
    r = call(NINE, "JEROUTER", p)
    status = f"OK ({r['ms']:.0f}ms, {r['len']}ch, {r.get('model','')})" if r["ok"] else f"FAIL {r.get('err','')} ({r['ms']:.0f}ms)"
    print(f"  [{i+1}] {status}")
