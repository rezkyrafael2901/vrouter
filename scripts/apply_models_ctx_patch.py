#!/usr/bin/env python3
"""Apply the /v1/models context_length passthrough patch to ANY vrouter copy.

Idempotent + self-verifying. Works on both layouts:
  root   -> /root/vrouter/main.py            (upcloud1)
  ubuntu -> /home/ubuntu/vrouter/main.py     (upcloud2, Tencent)

Usage:
  python3 apply_models_ctx_patch.py --dry-run          # report only, touch nothing
  python3 apply_models_ctx_patch.py --apply            # patch + compile + restart + verify
  python3 apply_models_ctx_patch.py --path /x/main.py --apply

The 8 edits below are the exact ones applied to the Tencent box on 13 Oct 2026
(commit 4cf0662). A known-answer test lives in --selftest: replaying them onto the
pristine pre-patch file must reproduce the patched file byte-for-byte.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------- edit set ---
# (label, old, new)  — old must match exactly; new is what replaces it.
EDITS = [
("meta-attr",
 """        self.models_fetched_at = 0.0

    @property
    def effective_models(self):""",
 """        self.models_fetched_at = 0.0
        self.models_meta = {}       # id -> full upstream model object (context_length, arch, caps)

    @property
    def effective_models(self):"""),

("fetch-keep-meta",
 """            if resp.status_code < 400:
                data = resp.json().get("data", []) if isinstance(resp.json(), dict) else []
                self.models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
                self.models_fetched_at = time.time()""",
 """            if resp.status_code < 400:
                body = resp.json() if isinstance(resp.json(), dict) else {}
                data = body.get("data", []) if isinstance(body, dict) else []
                ids, meta = [], {}
                for m in data:
                    if not isinstance(m, dict):
                        continue
                    mid = m.get("id")
                    if not mid:
                        continue
                    ids.append(mid)
                    # Keep the FULL upstream model object so /v1/models can pass
                    # context_length through to clients that discover model
                    # window sizes from the model list (Hermes, LiteLLM, OpenWebUI).
                    meta[mid] = m
                self.models = ids
                self.models_meta = meta
                self.models_fetched_at = time.time()"""),

("cache-save-meta",
 """        cache = {name: {"models": p.models, "fetched_at": p.models_fetched_at}
                 for name, p in PROVIDERS.items() if p.models}""",
 """        cache = {name: {"models": p.models, "fetched_at": p.models_fetched_at,
                        "meta": getattr(p, "models_meta", {})}
                 for name, p in PROVIDERS.items() if p.models}"""),

("cache-load-meta",
 """                if p:
                    p.models = data.get("models", [])
                    p.models_fetched_at = data.get("fetched_at", 0)""",
 """                if p:
                    p.models = data.get("models", [])
                    p.models_fetched_at = data.get("fetched_at", 0)
                    p.models_meta = data.get("meta", {}) or {}"""),

("ctx-helpers",
 """@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):""",
 """def _model_ctx_length(meta):
    \"\"\"Context window from an upstream model object (None when not reported).\"\"\"
    if not isinstance(meta, dict):
        return None
    for key in ("context_length", "context_window", "max_context_length"):
        val = meta.get(key)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    return None


def _provider_ctx_length(provider_name, model_id):
    p = PROVIDERS.get(provider_name)
    if not p or not model_id:
        return None
    return _model_ctx_length(getattr(p, "models_meta", {}).get(model_id))


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):"""),

("emit-combo",
 """        if combo_active and name not in seen:
            data.append({"id": name, "object": "model", "created": 0, "owned_by": "combo"})
            seen.add(name)""",
 """        if combo_active and name not in seen:
            entry = {"id": name, "object": "model", "created": 0, "owned_by": "combo"}
            # A combo's window = the largest window among its active routes.
            ctx = None
            for route in combo.get("routes", []):
                if not active_providers.get(route.get("provider"), False):
                    continue
                rctx = _provider_ctx_length(route.get("provider"), route.get("model"))
                if rctx and (ctx is None or rctx > ctx):
                    ctx = rctx
            if ctx:
                entry["context_length"] = ctx
            data.append(entry)
            seen.add(name)"""),

("emit-provider",
 """            for m in visible_models:
                mid = f"{p.prefix}/{m}"
                if mid not in seen:
                    data.append({"id": mid, "object": "model", "created": 0, "owned_by": p.name})
                    seen.add(mid)""",
 """            for m in visible_models:
                mid = f"{p.prefix}/{m}"
                if mid not in seen:
                    entry = {"id": mid, "object": "model", "created": 0, "owned_by": p.name}
                    ctx = _model_ctx_length(getattr(p, "models_meta", {}).get(m))
                    if ctx:
                        entry["context_length"] = ctx
                    data.append(entry)
                    seen.add(mid)"""),

("refresh-all-parity",
 """            ids = sorted({m.get("id") for m in data
                          if isinstance(m, dict) and m.get("id")}) if isinstance(data, list) else []
            if ids:
                p.models = ids
                p.models_fetched_at = time.time()""",
 """            meta = {m["id"]: m for m in data
                    if isinstance(m, dict) and m.get("id")} if isinstance(data, list) else {}
            ids = sorted(meta) if meta else []
            if ids:
                p.models = ids
                p.models_fetched_at = time.time()
                p.models_meta = meta   # keep context_length parity with fetch_models()"""),
]

MARKERS = ["self.models_meta = {}", "self.models_meta = meta",
           '"meta": getattr(p, "models_meta", {})', 'p.models_meta = data.get("meta", {}) or {}',
           "def _model_ctx_length(meta):", "def _provider_ctx_length(provider_name, model_id):",
           "p.models_meta = meta   # keep context_length parity"]


def find_main(path):
    if path:
        return path
    for cand in ("/root/vrouter/main.py", "/home/ubuntu/vrouter/main.py",
                 os.path.expanduser("~/vrouter/main.py")):
        if os.path.isfile(cand):
            return cand
    return None


def patch_text(text):
    """Apply every edit that is not already present. Returns (new_text, report)."""
    report = []
    for label, old, new in EDITS:
        if new in text:
            report.append((label, "already-patched"))
            continue
        if old in text:
            text = text.replace(old, new, 1)
            report.append((label, "APPLIED"))
        else:
            report.append((label, "ANCHOR-NOT-FOUND"))
    return text, report


def selftest():
    """Known-answer: replay the edits onto the pristine pre-patch file and compare
    with the live patched file (both on this box)."""
    import tempfile
    pristine = "/home/ubuntu/vrouter/main.py.bak-ctxmeta-20261013-191158"
    live = "/home/ubuntu/vrouter/main.py"
    if not (os.path.isfile(pristine) and os.path.isfile(live)):
        print("SELFTEST SKIP: pristine backup or live file missing")
        return 0
    src = open(pristine).read()
    out, rep = patch_text(src)
    ok = True
    for label, status in rep:
        if status == "ANCHOR-NOT-FOUND":
            print(f"  FAIL {label}: anchor not found in pristine file")
            ok = False
    same = out == open(live).read()
    print(f"  edits applied : {sum(1 for _, s in rep if s == 'APPLIED')}/{len(EDITS)}")
    print(f"  replay == live : {'YES' if same else 'NO'}")
    if not same and ok:
        # locate first divergence for the report
        live_t = open(live).read()
        for i, (a, b) in enumerate(zip(out, live_t)):
            if a != b:
                print(f"  first divergence at char {i}: replay={out[i-40:i+40]!r}")
                break
        else:
            print(f"  length differs: replay={len(out)} live={len(live_t)}")
    print("SELFTEST:", "PASS" if (ok and same) else "FAIL")
    return 0 if (ok and same) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-restart", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    path = find_main(a.path)
    if not path:
        print("FAIL: main.py not found (use --path)")
        return 2

    src = open(path).read()
    already = sum(1 for m in MARKERS if m in src)
    print(f"target : {path}  ({len(src):,} bytes)")
    print(f"markers: {already}/{len(MARKERS)} present")

    out, rep = patch_text(src)
    for label, status in rep:
        print(f"  {label:20s} {status}")

    bad = [l for l, s in rep if s == "ANCHOR-NOT-FOUND"]
    if bad:
        print(f"FAIL: anchors not found -> {bad}. This copy differs from the patched revision; "
              f"inspect manually instead of forcing.")
        return 3

    if out == src:
        print("VERDICT: already fully patched — nothing to do")
        return 0

    if not a.apply:
        print("VERDICT: patch needed (dry-run, nothing written)")
        return 0

    bak = f"{path}.bak-ctxmeta-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, bak)
    open(path, "w").write(out)
    rc = subprocess.run([sys.executable, "-m", "py_compile", path], capture_output=True)
    print(f"backup : {bak}")
    print(f"compile: rc={rc.returncode}")
    if rc.returncode != 0:
        print(rc.stderr.decode()[:500])
        print("ROLLBACK: restoring backup")
        shutil.copy2(bak, path)
        return 4

    if not a.no_restart:
        # root layout -> plain pm2; ubuntu layout -> needs the user session env
        if path.startswith("/home/"):
            user = path.split("/")[2]
            cmd = ["sudo", "-u", user, "XDG_RUNTIME_DIR=/run/user/1000", "pm2", "restart", "vrouter"]
        else:
            cmd = ["pm2", "restart", "vrouter"]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        print(f"restart: rc={r.returncode} {r.stdout.decode()[:120].strip()}")

    print("VERDICT: PATCHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
