#!/usr/bin/env python3
"""VRouter end-to-end test: gateway, proxy, combo."""
import sys
import httpx
import yaml

cfg = yaml.safe_load(open("/home/ubuntu/vrouter/config.yaml"))
srv = cfg.get("server") or {}
key_holder = srv.get("api_key") or ""
BASE = "http://127.0.0.1:20129"
results = []

def check(name, fn):
    try:
        results.append((name, "OK", fn()))
        print(f"  ✅ {name}")
    except Exception as e:
        results.append((name, "FAIL", str(e)))
        print(f"  ❌ {name}: {e}")

c = httpx.Client(timeout=90)
H = {"Authorization": f"Bearer {key_holder}"}

check("health", lambda: (_ for _ in ()).throw(AssertionError(f"{c.get(f'{BASE}/v1/health').json()}")) if False else c.get(f"{BASE}/v1/health").json()["status"] == "ok" or (_ for _ in ()).throw(AssertionError("bad health")))

def models():
    r = c.get(f"{BASE}/v1/models", headers=H)
    assert r.status_code == 200, r.text[:200]
    d = r.json()
    ids = [m["id"] for m in d["data"]]
    assert any(i.startswith("bb/") for i in ids), "no bb/ prefix model"
    assert any(i == "ALL" or i == "BLACKBOX" for i in ids), "combos not in model list"
    return f"{len(ids)} models"
check("models (+combos)", models)

def auth():
    r = c.get(f"{BASE}/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    return "401 ok"
check("auth reject", auth)

def chat_bb():
    r = c.post(f"{BASE}/v1/chat/completions", headers=H,
               json={"model": "bb/z-ai/glm-5.2", "messages": [{"role": "user", "content": "Reply with the single word: PONG"}], "max_tokens": 300})
    assert r.status_code == 200, f"{r.status_code}: {r.text[:400]}"
    content = r.json()["choices"][0]["message"].get("content") or ""
    assert content.strip(), "empty content"
    return f"reply: {content[:30]!r}"
check("chat bb (proxy Geonode)", chat_bb)

def chat_jr():
    r = c.post(f"{BASE}/v1/chat/completions", headers=H,
               json={"model": "jr/f/deepseek-v4-flash-free", "messages": [{"role": "user", "content": "Reply with the single word: PONG"}], "max_tokens": 300})
    assert r.status_code == 200, f"{r.status_code}: {r.text[:400]}"
    content = r.json()["choices"][0]["message"].get("content") or ""
    assert content, "empty content"
    return f"reply: {content[:30]!r}"
check("chat jr (direct)", chat_jr)

def combo_all():
    r = c.post(f"{BASE}/v1/chat/completions", headers=H,
               json={"model": "ALL", "messages": [{"role": "user", "content": "Reply with the single word: PONG"}], "max_tokens": 300})
    assert r.status_code == 200, f"{r.status_code}: {r.text[:400]}"
    content = r.json()["choices"][0]["message"].get("content") or ""
    assert content, "empty content"
    return f"combo ALL reply: {content[:30]!r}"
check("combo ALL", combo_all)

def combo_bb():
    r = c.post(f"{BASE}/v1/chat/completions", headers=H,
               json={"model": "BLACKBOX", "messages": [{"role": "user", "content": "Reply with the single word: PONG"}], "max_tokens": 300})
    assert r.status_code == 200, f"{r.status_code}: {r.text[:400]}"
    content = r.json()["choices"][0]["message"].get("content") or ""
    assert content, "empty content"
    return f"combo BLACKBOX reply: {content[:30]!r}"
check("combo BLACKBOX", combo_bb)

def stream():
    with c.stream("POST", f"{BASE}/v1/chat/completions", headers=H,
                  json={"model": "bb/z-ai/glm-5.2", "stream": True, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 300}) as r:
        assert r.status_code == 200, f"status {r.status_code}"
        chunks = 0
        for line in r.iter_lines():
            if line.startswith("data:") and line != "data: [DONE]":
                chunks += 1
        assert chunks > 0
        return f"SSE chunks: {chunks}"
check("streaming", stream)

def dash():
    r = c.get(f"{BASE}/")
    assert r.status_code == 200 and "VRouter" in r.text
    return "dashboard vrouter served"
check("dashboard", dash)

def dash_login():
    r = c.post(f"{BASE}/api/login", json={"username": srv.get("dashboard_user"), "password": srv.get("dashboard_password")})
    assert r.status_code == 200
    return "login ok"
check("dashboard login", dash_login)

def dash_status():
    r = c.post(f"{BASE}/api/login", json={"username": srv.get("dashboard_user"), "password": srv.get("dashboard_password")})
    r2 = c.get(f"{BASE}/api/status", cookies=r.cookies)
    assert r2.status_code == 200
    j = r2.json()
    assert j["total_providers"] >= 7
    assert j["total_keys"] >= 200
    assert len(j["proxies"]) >= 100
    assert len(j["combos"]) >= 2
    return f"{j['total_providers']} prov, {j['total_keys']} keys, {len(j['proxies'])} proxies, {len(j['combos'])} combos"
check("dashboard status (full)", dash_status)

def add_delete_combo():
    r = c.post(f"{BASE}/api/login", json={"username": srv.get("dashboard_user"), "password": srv.get("dashboard_password")})
    cookies = r.cookies
    # add temp combo
    rr = c.post(f"{BASE}/api/combos", cookies=cookies,
                json={"name": "TESTC", "routes": [{"provider": "blackbox", "model": "z-ai/glm-5.2", "weight": 1}]})
    assert rr.status_code == 200, rr.text[:200]
    # delete it
    rr2 = c.delete(f"{BASE}/api/combos/TESTC", cookies=cookies)
    assert rr2.status_code == 200
    return "add/delete combo ok"
check("combo CRUD", add_delete_combo)

print("\n=== SUMMARY ===")
fails = [r for r in results if r[1] == "FAIL"]
print(f"PASS: {len(results) - len(fails)}/{len(results)}")
for f_ in fails:
    print(f"  FAIL {f_[0]}: {f_[2]}")
sys.exit(1 if fails else 0)