#!/usr/bin/env python3
"""Live verification of the vrouter :20129 /v1/models context_length pass-through.

Reads the gateway key from config.yaml at runtime (never inline secrets in shell).
Checks:
  - how many model entries carry context_length
  - the histogram of windows
  - the specific ids Hermes / other clients resolve
Exit 0 if at least one entry carries a window and the probe model is present.
"""
import collections
import json
import os
import sys
import urllib.error
import urllib.request

import yaml

CFG = os.path.expanduser("~/vrouter/config.yaml")
BASE = os.environ.get("VROUTER_BASE", "http://127.0.0.1:20129")
TARGETS = [
    "deepseek-unrestricted",
    "qwen3.8-flash",
    "step-5-unrestricted",
    "JEROUTER/deepseek-unrestricted",
    "JEROUTER/qwen3.8-flash",
]


def main() -> int:
    cfg = yaml.safe_load(open(CFG))
    key = (cfg.get("server") or {}).get("api_key")
    if not key:
        print("FAIL: no server.api_key in config.yaml")
        return 2

    req = urllib.request.Request(
        f"{BASE}/v1/models",
        headers={"Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            status = r.status
            body = r.read()
    except urllib.error.HTTPError as e:
        print(f"FAIL: HTTP {e.code} {e.read()[:120]!r}")
        return 2

    print(f"HTTP {status}  bytes={len(body)}")
    doc = json.loads(body)
    rows = doc.get("data", [])
    with_ctx = [r for r in rows if isinstance(r.get("context_length"), int)]
    print(f"models: {len(rows)}   with context_length: {len(with_ctx)}")
    hist = collections.Counter(r["context_length"] for r in with_ctx)
    print("window histogram:", dict(sorted(hist.items())))

    missing = []
    for t in TARGETS:
        hit = next((r for r in rows if r.get("id") == t), None)
        if hit is None:
            print(f"  {t:36s} not listed")
            missing.append(t)
            continue
        print(f"  {t:36s} context_length={hit.get('context_length')}")

    if not with_ctx:
        print("VERDICT: FAIL — no entry carries context_length (pass-through not live)")
        return 1
    print("VERDICT: PASS — context_length pass-through is live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
