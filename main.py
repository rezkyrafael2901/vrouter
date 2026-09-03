#!/usr/bin/env python3
"""
VRouter — OpenAI-compatible gateway + mobile dashboard.
Replacement for 9router: prefix routing, fallback chain, key rotation,
per-provider proxy pools, model combos, circuit breaker, SSE streaming.
"""
from __future__ import annotations
import asyncio
import json
import os
import random
import re
import secrets
import time
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import sqlite3
from pathlib import Path

import sync_9router
import vrouter_db

CONFIG_PATH = os.environ.get("VROUTER_CONFIG", "/home/ubuntu/vrouter/config.yaml")
DASHBOARD_PATH = os.path.join(os.path.dirname(CONFIG_PATH), "dashboard.html")
ADMIN_PATH = os.path.join(os.path.dirname(CONFIG_PATH), "admin.html")
PUBLIC_DB_PATH = "/home/ubuntu/vrouter-public/public.db"
SITE_NAME = "VRouter"
SITE_URL = "https://vrouter.my.id"
templates = Jinja2Templates(directory=os.path.dirname(CONFIG_PATH))
LANDING_PATH = os.path.join(os.path.dirname(CONFIG_PATH), "landing.html")
DOCS_PATH = os.path.join(os.path.dirname(CONFIG_PATH), "docs.html")
AUTH_PATH = os.path.join(os.path.dirname(CONFIG_PATH), "auth.html")
HISTORY_PATH = os.path.join(os.path.dirname(CONFIG_PATH), "history.jsonl")
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
MODELS_CACHE_PATH = os.path.join(os.path.dirname(CONFIG_PATH), "models_cache.json")
HISTORY_MAX = 5000

# Phase 9 — Cost-aware routing: per-model pricing (USD per 1M tokens)
# Static defaults for known providers; can be overridden per-provider in config
DEFAULT_MODEL_COSTS = {
    # OpenAI
    "gpt-4o": {"in": 5.00, "out": 15.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-4.1-nano": {"in": 0.10, "out": 0.40},
    "o1-preview": {"in": 15.00, "out": 60.00},
    "o1-mini": {"in": 3.00, "out": 12.00},
    "o3-mini": {"in": 1.10, "out": 4.40},
    # Anthropic
    "claude-3.5-sonnet": {"in": 3.00, "out": 15.00},
    "claude-3.5-haiku": {"in": 0.80, "out": 4.00},
    "claude-3-opus": {"in": 15.00, "out": 75.00},
    # Google
    "gemini-1.5-pro": {"in": 3.50, "out": 10.50},
    "gemini-1.5-flash": {"in": 0.075, "out": 0.30},
    "gemini-2.0-flash": {"in": 0.10, "out": 0.40},
    # XAI
    "grok-2": {"in": 2.00, "out": 10.00},
    "grok-3": {"in": 3.00, "out": 15.00},
    # DeepSeek
    "deepseek-chat": {"in": 0.27, "out": 1.10},
    "deepseek-reasoner": {"in": 0.55, "out": 2.20},
    # Qwen / Alibaba
    "qwen2.5-72b": {"in": 0.40, "out": 0.40},
    "qwen2.5-32b": {"in": 0.20, "out": 0.20},
    "qwen2.5-14b": {"in": 0.10, "out": 0.10},
    "qwen2.5-7b": {"in": 0.05, "out": 0.05},
    "qwen3-coder": {"in": 0.30, "out": 0.30},
    # GLM / Z.ai
    "glm-4": {"in": 0.50, "out": 0.50},
    "glm-4.5": {"in": 0.50, "out": 0.50},
    "glm-5": {"in": 0.80, "out": 0.80},
    # Minimax
    "minimax-m3": {"in": 1.00, "out": 1.00},
    # Nemotron
    "nemotron-3-ultra": {"in": 0.50, "out": 0.50},
    # Mimo / Xiaomi
    "mimo-v2.5-pro": {"in": 0.30, "out": 0.30},
    # Free models (OpenRouter style)
    "free": {"in": 0.0, "out": 0.0},
}

# Provider-specific cost overrides (set in config per provider)
PROVIDER_MODEL_COSTS: dict[str, dict] = {}


def get_model_cost(provider: Provider, model: str) -> tuple[float, float]:
    """Return (input_cost_per_1M, output_cost_per_1M) for a model.
    Priority: provider override > provider default_model cost > DEFAULT_MODEL_COSTS > (0.0, 0.0)."""
    # 1) Provider-specific override (exact match)
    p_costs = PROVIDER_MODEL_COSTS.get(provider.name, {})
    if model in p_costs:
        c = p_costs[model]
        return float(c.get("in", 0.0)), float(c.get("out", 0.0))
    # 2) Provider-specific override (case-insensitive fuzzy match)
    model_lower = model.lower()
    for cost_model, cost_data in p_costs.items():
        if cost_model.lower() in model_lower or model_lower in cost_model.lower():
            return float(cost_data.get("in", 0.0)), float(cost_data.get("out", 0.0))
    # 3) Default model cost from provider
    if provider.default_model and provider.default_model in p_costs:
        c = p_costs[provider.default_model]
        return float(c.get("in", 0.0)), float(c.get("out", 0.0))
    # 4) Static defaults (fuzzy match by basename)
    base = model.split("/")[-1].lower()
    for k, v in DEFAULT_MODEL_COSTS.items():
        if k in base or base in k:
            return v["in"], v["out"]
    # 5) Free marker
    if "free" in base or "free" in model.lower():
        return 0.0, 0.0
    # 6) Unknown — assume free (conservative for cost-aware routing)
    return 0.0, 0.0


def estimate_request_cost(provider: Provider, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a request based on token usage."""
    in_cost, out_cost = get_model_cost(provider, model)
    return (prompt_tokens * in_cost + completion_tokens * out_cost) / 1_000_000.0


# --- Local token counting fallback (many free providers omit usage on stream) ---
_TOKEN_ENC = None
def _get_encoder():
    """Lazy tiktoken encoder; None if tiktoken unavailable (falls back to heuristic)."""
    global _TOKEN_ENC
    if _TOKEN_ENC is None:
        try:
            import tiktoken
            _TOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKEN_ENC = False  # sentinel: tried and failed
    return _TOKEN_ENC or None


def count_tokens(text: str) -> int:
    """Count tokens in a string. tiktoken if available, else ~chars/4 heuristic."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def count_prompt_tokens(body: dict) -> int:
    """Estimate prompt tokens from an OpenAI-style chat body (messages or prompt).
    Adds a small per-message overhead to approximate the ChatML framing."""
    try:
        total = 0
        msgs = body.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                total += 4  # role/format framing overhead per message
                c = m.get("content")
                if isinstance(c, str):
                    total += count_tokens(c)
                elif isinstance(c, list):  # multimodal content parts
                    for part in c:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            total += count_tokens(part["text"])
                if isinstance(m.get("name"), str):
                    total += count_tokens(m["name"])
            return total + 2  # priming
        p = body.get("prompt")
        if isinstance(p, str):
            return count_tokens(p)
    except Exception:
        pass
    return 0


def _extract_delta_text(obj: dict) -> str:
    """Pull assistant text out of a streamed chat.completion.chunk."""
    try:
        ch = obj.get("choices")
        if not ch:
            return ""
        delta = ch[0].get("delta") or {}
        txt = delta.get("content")
        if isinstance(txt, str):
            return txt
        # some providers stream reasoning separately; count it too
        rc = delta.get("reasoning_content")
        return rc if isinstance(rc, str) else ""
    except Exception:
        return ""


def pick_cost_aware_route(combo_name: str, exclude_reasoning: bool = False):
    """Cost-aware combo strategy: pick cheapest healthy model that meets quality.
    Returns [(provider, model)] ordered by estimated cost per 1K tokens (cheapest first).
    Skips unhealthy/locked providers + dead models. Only considers models with known pricing."""
    combo = COMBOS.get(combo_name)
    if not combo or not combo["routes"]:
        return []
    routes = list(combo["routes"])
    healthy_routes = []
    for r in routes:
        p = PROVIDERS.get(r.get("provider", ""))
        if not p or not p.is_active or not p.keys:
            continue
        if p.locked_until > time.time():
            continue
        model = r.get("model", p.default_model or "auto")
        model_id = f"{p.name}/{model}"
        dm = DEAD_MODELS.get(model_id)
        if dm and dm.get("disabled_at", 0) > 0 and time.time() - dm.get("disabled_at", 0) < 900:
            continue
        if exclude_reasoning and model in REASONING_MODELS:
            continue
        # Must have pricing (known or free)
        in_c, out_c = get_model_cost(p, model)
        if in_c == 0.0 and out_c == 0.0 and "free" not in model.lower():
            # Unknown pricing — deprioritize but don't exclude (assume free-ish)
            est_cost_per_1k = 999.0  # high placeholder
        else:
            # Estimate cost for 1K prompt + 1K completion tokens
            est_cost_per_1k = (in_c + out_c) / 1000.0
        healthy_routes.append((r, p, model, est_cost_per_1k))
    if not healthy_routes:
        # Fallback: any active provider with keys
        healthy_routes = [
            (r, PROVIDERS[r["provider"]], r.get("model", PROVIDERS[r["provider"]].default_model or "auto"), 0.0)
            for r in routes
            if PROVIDERS.get(r.get("provider", "")) and PROVIDERS[r["provider"]].is_active and PROVIDERS[r["provider"]].keys
        ]
    # Sort by estimated cost (cheapest first)
    healthy_routes.sort(key=lambda x: x[3])
    return [(p, m) for (_, p, m, _) in healthy_routes]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

SERVER_CFG = CONFIG["server"]
ROUTER_CFG = CONFIG["router"]
API_KEY = SERVER_CFG["api_key"]
DASH_USER = SERVER_CFG.get("dashboard_user", "admin")
DASH_PASS = SERVER_CFG.get("dashboard_password", "admin")

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
class Provider:
    def __init__(self, name, base_url, prefix, type_, keys, weight, default_model="", proxy="", is_active=True, manual_models=None, keep_prefix=False):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.prefix = prefix
        self.type = type_          # apikey | oauth | none
        self.keys = list(keys)
        self.weight = weight
        self.default_model = default_model
        self.proxy = proxy          # proxy pool name or full URL ("" = direct)
        self.is_active = is_active  # on/off toggle
        self.keep_prefix = keep_prefix  # send prefix/model upstream (9router-style aliases)
        self._key_idx = 0
        self.failures = 0
        self.last_error = None
        self.last_used = None
        self.total_requests = 0
        self.total_errors = 0
        self.locked_until = 0.0
        self.models = []            # cached model list from provider
        self.manual_models = list(manual_models) if manual_models else []  # user-specified models
        self.models_fetched_at = 0.0

    @property
    def effective_models(self):
        """Merge auto-fetched + manual models, dedupe, prefer manual order first."""
        seen = set()
        result = []
        for m in self.manual_models:
            if m and m not in seen:
                seen.add(m)
                result.append(m)
        for m in self.models:
            if m and m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def fetch_models(self, force=False, timeout=25):
        """Fetch /models from provider. Returns (ok, models|error). Cached 10 min."""
        if not force and self.models and (time.time() - self.models_fetched_at) < 600:
            return True, self.models
        key = self.next_key()
        if not key:
            return False, "no keys"
        url = f"{self.base_url}/models"
        headers = {**self.auth_header(key)}
        proxy_url = resolve_proxy_url(self.proxy)
        try:
            if proxy_url:
                with httpx.Client(timeout=timeout, proxy=proxy_url) as c:
                    resp = c.get(url, headers=headers)
            else:
                with httpx.Client(timeout=timeout) as c:
                    resp = c.get(url, headers=headers)
            if resp.status_code < 400:
                data = resp.json().get("data", []) if isinstance(resp.json(), dict) else []
                self.models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
                self.models_fetched_at = time.time()
                _save_models_cache()
                return True, self.models
            return False, f"HTTP {resp.status_code}: {resp.text[:150]}"
        except Exception as e:
            return False, str(e)[:150]

    def next_key(self):
        if not self.keys:
            return None
        # Gateway-style provider (keep_prefix=True, upstream = 9router/local gateway):
        # keys[0] adalah gateway API key (mis. "123456"); sisanya PAT upstream yang
        # hanya dipakai 9router di balik layar — jangan kirim PAT ke gateway lokal.
        if getattr(self, "keep_prefix", False):
            return self.keys[0]
        now = time.time()
        k = len(self.keys)
        # Phase 4 anti-429 + Phase E weighted: pilih key termeterai tercepat (EMA latency×success).
        scored = []   # (score, key, kk) — lower score = lebih cepat+reliable
        cold = []     # valid key tapi belum cukup sample — round-robin index
        for _ in range(k):
            idx = self._key_idx % k
            self._key_idx = (self._key_idx + 1) % k
            key = self.keys[idx]
            kk = f"{self.name}|{key}"
            if KEY_COOLDOWN.get(kk, 0) > now:
                continue  # key masih cooldown 429 — lewati
            if now - KEY_LAST_SENT.get(kk, 0.0) < KEY_MIN_INTERVAL:
                continue  # terlalu cepat buat key ini — lewati
            sc = key_score(kk)
            if sc is not None:
                scored.append((sc, key, kk))
            else:
                cold.append((key, kk))
        if scored:
            scored.sort(key=lambda t: t[0])  # tercepat & paling reliable dulu
            _, key, kk = scored[0]
            KEY_LAST_SENT[kk] = now
            return key
        if cold:
            key, kk = cold[0]
            KEY_LAST_SENT[kk] = now
            return key
        # semua key cool — best-effort (key dengan cooldown paling dekat)
        idx = self._key_idx % k
        self._key_idx = (self._key_idx + 1) % k
        return self.keys[idx]

    def auth_header(self, key):
        if self.type in ("apikey", "oauth", "Authorization"):
            return {"Authorization": f"Bearer {key}"}
        return {}

    def to_dict(self, include_key=False):
        d = {
            "name": self.name,
            "base_url": self.base_url,
            "prefix": self.prefix,
            "type": self.type,
            "weight": self.weight,
            "default_model": self.default_model,
            "proxy": self.proxy,
            "is_active": self.is_active,
            "key_count": len(self.keys),
            "models_count": len(self.effective_models),
            "models": self.effective_models[:300],
            "manual_models": self.manual_models[:300],
            "failures": self.failures,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "last_error": self.last_error,
            "last_used": self.last_used,
            "locked_until": self.locked_until,
            "locked": self.locked_until > time.time(),
            "healthy": self.is_active and self.failures < 5 and self.locked_until <= time.time(),
        }
        if include_key:
            d["keys"] = self.keys
        return d

PROVIDERS: dict[str, Provider] = {}
PROXIES: dict[str, str] = {}    # pool name -> proxy URL
COMBOS: dict[str, dict] = {}    # combo name -> {routes: [{provider, model, weight}]}
HISTORY: deque = deque(maxlen=HISTORY_MAX)   # [ {ts, provider, model, status, code, ms, err} ]
# Track provider names the user explicitly deleted — sync_9router must NOT re-add these
DELETED_PROVIDERS: set[str] = set()

# Phase 1 — Smart routing infrastructure
MODEL_STATS: dict[str, dict] = {}    # "provider/model" -> stats
REASONING_MODELS: set[str] = set()   # models known to return reasoning-only output
DEAD_MODELS: dict[str, dict] = {}    # model_id -> {failures, last_checked, last_error, disabled_at}
# Phase 4 — per-key anti-429 + EMA adaptive scoring
KEY_COOLDOWN: dict = {}               # "provider|key" -> unlock ts (exponential backoff after 429)
KEY_429_BURST: dict = {}             # "provider|key" -> consecutive 429 count
KEY_LAST_SENT: dict = {}            # "provider|key" -> ts (min-interval gate)
KEY_MIN_INTERVAL = ROUTER_CFG.get("key_min_interval", 0.5)   # min gap per key (anti-burst)
HEDGE_RACE_TIMEOUT = ROUTER_CFG.get("hedge_race_timeout", 3.0)
SPECULATIVE_HEDGE = ROUTER_CFG.get("speculative_hedge", True)
# Phase E — per-key weighted routing: pick the fastest-healthy key per provider, not round-robin.
KEY_STATS: dict = {}              # "provider|key" -> {ema_ms, ema_ok, samples, last_latency_ms}
KEY_ALPHA = ROUTER_CFG.get("key_ema_alpha", 0.25)
KEY_MIN_SAMPLES = ROUTER_CFG.get("key_min_samples", 5)  # baru weighted setelah N hasil
# Circuit-breaker tunables (configurable via config.yaml router.*)
CB_FAIL_THRESHOLD = ROUTER_CFG.get("cb_fail_threshold", 3)      # failures before trip
CB_LOCK_SECONDS = ROUTER_CFG.get("cb_lock_seconds", 30)         # lock on 429/all-keys-cooldown
CB_HEALTH_LOCK_SECONDS = ROUTER_CFG.get("cb_health_lock_seconds", 60)  # lock from health-check fails
CB_EVENTS: deque = deque(maxlen=200)   # circuit-breaker trip/reset audit log
# Tok/s meter — per "provider/model" streaming throughput (TTFT + tokens/sec)
THROUGHPUT_STATS: dict = {}   # "provider/model" -> {samples, ttft_ema, toks_ema, ttft_last, toks_last, best_toks, tok_total, last_used}
LAST_HEALTH_CHECK = 0.0
HEALTH_CHECK_TASK = None
MODEL_PROBE_TASK = None
SYNC_STATE = {
    "enabled": False,
    "interval_seconds": 0,
    "last_run": 0.0,
    "last_result": None,
    "next_run": 0.0,
    "running": False,
    "last_fingerprint": None,
}


def _init_model_stats(key: str):
    if key not in MODEL_STATS:
        MODEL_STATS[key] = {
            "total": 0, "ok": 0, "err": 0,
            "latency_sum": 0, "latency_min": 999999, "latency_max": 0,
            "reasoning": False, "last_error": None, "last_used": 0.0,
            # Phase 4 — EMA: reaktif ke slowdown/lembah suhu terbaru
            "ema_latency_ms": 0.0, "ema_success": 0.5, "samples": 0,
        }


def update_model_stats(provider: str, model: str, status: str, latency_ms: float, is_reasoning: bool = False):
    key = f"{provider}/{model}"
    _init_model_stats(key)
    s = MODEL_STATS[key]
    s["total"] += 1
    s["latency_sum"] += latency_ms
    s["latency_min"] = min(s["latency_min"], int(latency_ms))
    s["latency_max"] = max(s["latency_max"], int(latency_ms))
    s["last_used"] = time.time()
    if status == "ok":
        s["ok"] += 1
        s["last_error"] = None
    else:
        s["err"] += 1
        s["last_error"] = status
    # Phase 4 — EMA update (alpha ~0.25; ~5-sample effective memory).
    # Defensive .get() karena MODEL_STATS bisa di-reload dari state lama (tanpa field baru).
    alpha = 0.25
    lat_ms = float(latency_ms)
    s["ema_latency_ms"] = alpha * lat_ms + (1 - alpha) * s.get("ema_latency_ms", lat_ms)
    s["ema_success"] = alpha * (1.0 if status == "ok" else 0.0) + (1 - alpha) * s.get("ema_success", 0.5)
    s["samples"] = s.get("samples", 0) + 1
    if is_reasoning:
        s["reasoning"] = True
        REASONING_MODELS.add(model)


def trip_circuit(provider, reason: str, lock_seconds: int = None):
    """Open the circuit breaker for a provider: lock it for N seconds and log
    the trip event for the dashboard audit panel. Returns the unlock timestamp."""
    secs = CB_LOCK_SECONDS if lock_seconds is None else lock_seconds
    provider.locked_until = time.time() + secs
    CB_EVENTS.append({
        "ts": time.time(), "provider": provider.name, "action": "trip",
        "reason": reason[:120], "lock_seconds": secs,
        "failures": provider.failures, "until": provider.locked_until,
    })
    return provider.locked_until


def reset_circuit(provider, reason: str = "manual"):
    """Force-close the circuit breaker for a provider (dashboard button / recovery)."""
    was_locked = provider.locked_until > time.time()
    provider.locked_until = 0.0
    provider.failures = 0
    provider.last_error = None
    CB_EVENTS.append({
        "ts": time.time(), "provider": provider.name, "action": "reset",
        "reason": reason[:120], "was_locked": was_locked,
    })


def record_throughput(provider_name: str, model: str, ttft_ms: float, tok_s: float, c_tok: int):
    """Tok/s meter: accumulate per provider/model streaming throughput.
    Uses EMA (alpha=0.3) so the ranking reflects recent behaviour, keeps a
    best-ever tok/s and total tokens streamed."""
    if not tok_s and not ttft_ms:
        return
    key = f"{provider_name}/{model}"
    s = THROUGHPUT_STATS.get(key)
    a = 0.3
    if not s:
        s = {"provider": provider_name, "model": model, "samples": 0,
             "ttft_ema": ttft_ms, "toks_ema": tok_s, "ttft_last": ttft_ms,
             "toks_last": tok_s, "best_toks": tok_s, "tok_total": 0, "last_used": 0.0}
        THROUGHPUT_STATS[key] = s
    s["samples"] += 1
    s["ttft_last"] = ttft_ms
    s["toks_last"] = tok_s
    s["ttft_ema"] = round(a * ttft_ms + (1 - a) * s["ttft_ema"], 1) if ttft_ms else s["ttft_ema"]
    s["toks_ema"] = round(a * tok_s + (1 - a) * s["toks_ema"], 2) if tok_s else s["toks_ema"]
    s["best_toks"] = max(s["best_toks"], tok_s)
    s["tok_total"] += int(c_tok or 0)
    s["last_used"] = time.time()


def detect_and_fix_reasoning(data: dict) -> bool:
    """Check if response is reasoning-only (content empty, reasoning present).
    If so, extract reasoning as content. Returns True if reasoning-only."""
    try:
        choices = data.get("choices", [])
        if not choices:
            return False
        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning", "") or msg.get("reasoning_content", "") or ""
        if not content.strip() and reasoning.strip():
            msg["content"] = reasoning[:4000]
            choices[0]["message"] = msg
            data["choices"] = choices
            return True
    except Exception:
        pass
    return False


def parse_upstream_json(resp) -> dict:
    """Robust JSON parse of upstream responses.
    Some upstreams (ipeenk) return SSE-shaped bodies (`data: {...}`) even for
    stream=False, or append trailing `data: [DONE]` after JSON. If the body is
    a sequence of chat.completion.chunk objects (SSE), merge deltas into a
    single completion. Returns None if nothing parses."""
    text = resp.text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except Exception:
        data = None
    if isinstance(data, dict) and data.get("object") == "chat.completion.chunk":
        # single streaming-style chunk returned as JSON (ipeenk quirk)
        choices = data.get("choices") or []
        content = ""
        finish_reason = None
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content") or ""
            finish_reason = choices[0].get("finish_reason")
        return {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "created": data.get("created") or int(time.time()),
            "model": data.get("model", ""),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason or "stop",
            }],
        }
    if isinstance(data, dict):
        return data
    if data is not None:
        return None
    # SSE-shaped: lines starting with "data:"
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                parts.append(payload)
    # If any chunk looks like a streaming delta, merge all deltas into a
    # standard chat.completion object.
    parsed = []
    for candidate in parts:
        try:
            parsed.append(json.loads(candidate))
        except Exception:
            continue
    if parsed and parsed[-1].get("object") == "chat.completion.chunk":
        full_content = ""
        full_reasoning = ""
        finish_reason = None
        merged_model = parsed[-1].get("model", "")
        merged_id = parsed[-1].get("id", "")
        for chunk in parsed:
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            c = delta.get("content")
            if c:
                full_content += c
            rr = delta.get("reasoning_content") or delta.get("reasoning")
            if rr:
                full_reasoning += rr
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
        return {
            "id": merged_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": merged_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_content,
                            "reasoning_content": full_reasoning},
                "finish_reason": finish_reason or "stop",
            }],
        }
    for candidate in parsed:
        if candidate.get("choices"):
            return candidate
    # trailing "data: [DONE]" after JSON object
    try:
        idx = text.rfind("}")
        if idx > 0:
            return json.loads(text[: idx + 1])
    except Exception:
        pass
    return None


async def _ping_provider(p: 'Provider'):
    """Ping a single provider /models endpoint for health check."""
    try:
        key = p.next_key()
        if not key:
            return
        url = f"{p.base_url}/models"
        headers = {**p.auth_header(key)}
        proxy_url = resolve_proxy_url(p.proxy)
        t0 = time.time()
        if proxy_url:
            async with httpx.AsyncClient(timeout=15, proxy=proxy_url) as c:
                resp = await c.get(url, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=15) as c:
                resp = await c.get(url, headers=headers)
        latency = (time.time() - t0) * 1000
        if resp.status_code < 400:
            if p.locked_until > time.time():
                reset_circuit(p, "health-check recovered")
            else:
                p.failures = 0
                p.locked_until = 0.0
                p.last_error = None
            # refresh model list
            rdata = resp.json()
            data = rdata.get("data", []) if isinstance(rdata, dict) else []
            new_models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
            if new_models:
                p.models = new_models
                p.models_fetched_at = time.time()
                _save_models_cache()
        else:
            p.failures += 1
            p.last_error = f"[{resp.status_code}] {resp.text[:150]}"
            if p.failures >= CB_FAIL_THRESHOLD:
                trip_circuit(p, f"health-check [{resp.status_code}]", CB_HEALTH_LOCK_SECONDS)
    except httpx.TimeoutException:
        p.failures += 1
        p.last_error = "health check timeout"
        if p.failures >= CB_FAIL_THRESHOLD:
            trip_circuit(p, "health-check timeout", CB_HEALTH_LOCK_SECONDS)
    except Exception as e:
        p.failures += 1
        p.last_error = str(e)[:150]
        if p.failures >= CB_FAIL_THRESHOLD:
            trip_circuit(p, f"health-check exc:{str(e)[:60]}", CB_HEALTH_LOCK_SECONDS)


async def health_check_loop(app: FastAPI):
    """Background task: ping all providers every N seconds."""
    global LAST_HEALTH_CHECK
    interval = ROUTER_CFG.get("health_check_interval", 300)
    # initial check after 10s startup
    await asyncio.sleep(10)
    while True:
        tasks = []
        for p in PROVIDERS.values():
            if not p.keys:
                continue
            tasks.append(_ping_provider(p))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        LAST_HEALTH_CHECK = time.time()
        await asyncio.sleep(interval)


async def _probe_model(p: 'Provider', model: str) -> dict:
    """Test a single model with a simple prompt. Returns {ok, latency, error, reasoning}."""
    key = p.next_key()
    if not key:
        return {"ok": False, "latency": 0, "error": "no keys", "reasoning": False}
    url = f"{p.base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 10,
        "stream": False,
    }
    headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA, **p.auth_header(key)}
    proxy_url = resolve_proxy_url(p.proxy)
    t0 = time.time()
    try:
        if proxy_url:
            async with httpx.AsyncClient(timeout=20, proxy=proxy_url) as c:
                resp = await c.post(url, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=20) as c:
                resp = await c.post(url, json=payload, headers=headers)
        latency = (time.time() - t0) * 1000
        if resp.status_code < 400:
            data = resp.json()
            is_reasoning = detect_and_fix_reasoning(data)
            return {"ok": True, "latency": int(latency), "error": None, "reasoning": is_reasoning}
        return {"ok": False, "latency": int(latency), "error": f"HTTP {resp.status_code}: {resp.text[:100]}", "reasoning": False}
    except httpx.TimeoutException:
        return {"ok": False, "latency": int((time.time() - t0) * 1000), "error": "timeout", "reasoning": False}
    except Exception as e:
        return {"ok": False, "latency": int((time.time() - t0) * 1000), "error": str(e)[:100], "reasoning": False}


async def model_probe_loop(app: FastAPI):
    """Background task: probe models periodically. 3x fail → mark dead.
    Rate-limit friendly: max CONCURRENT probes globally (3), per-provider cap 6,
    prioritize combo-used models + dead-model revives. Dead models revive after
    DEAD_GRACE via next probe."""
    await asyncio.sleep(30)
    dead_grace = 600  # seconds before a dead model gets probed again
    sem = asyncio.Semaphore(3)  # never burst an upstream

    def _priority(pname: str, model: str, now: float) -> int:
        mid = f"{pname}/{model}"
        dm = DEAD_MODELS.get(mid)
        if dm and dm.get("disabled_at", 0) > 0 and now - dm["disabled_at"] >= dead_grace:
            return 0  # revive first
        # combo-used models get probed before never-seen models
        for c in COMBOS.values():
            for r in c.get("routes", []):
                if r.get("provider") == pname and r.get("model") == model:
                    return 1
        s = MODEL_STATS.get(mid)
        if not s or s["total"] == 0:
            return 2  # unprobed
        return 3  # already probed — lower priority

    async def _probe_sem(p, model):
        async with sem:
            return await _probe_model(p, model)

    while True:
        tasks = []
        now = time.time()
        for p in PROVIDERS.values():
            if not p.keys or not p.models:
                continue
            candidates = sorted(p.models, key=lambda m: _priority(p.name, m, now))[:6]
            for model in candidates:
                tasks.append((p.name, model, _probe_sem(p, model)))
        results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
        for (prov_name, model, _), result in zip(tasks, results):
            model_id = f"{prov_name}/{model}"
            if isinstance(result, Exception):
                result = {"ok": False, "latency": 0, "error": str(result)[:100], "reasoning": False}
            if result["ok"]:
                # Model is alive — clear dead status, track reasoning
                DEAD_MODELS.pop(model_id, None)
                if result["reasoning"]:
                    REASONING_MODELS.add(model)
                update_model_stats(prov_name, model, "ok", result["latency"], result["reasoning"])
            else:
                if model_id not in DEAD_MODELS:
                    DEAD_MODELS[model_id] = {"failures": 0, "last_checked": 0, "last_error": None, "disabled_at": 0}
                dm = DEAD_MODELS[model_id]
                dm["failures"] += 1
                dm["last_checked"] = time.time()
                dm["last_error"] = result["error"]
                if dm["failures"] >= 3:
                    dm["disabled_at"] = time.time()
                update_model_stats(prov_name, model, result["error"], result["latency"])
        await asyncio.sleep(300)  # 5 minutes


def log_history(entry: dict):
    entry["ts"] = entry.get("ts", time.time())
    entry["ms"] = round(entry.get("ms", 0))
    HISTORY.append(entry)
    try:
        with open(HISTORY_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def load_history():
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            HISTORY.append(json.loads(line))
                        except Exception:
                            pass
    except Exception:
        pass


def _save_models_cache():
    try:
        cache = {name: {"models": p.models, "fetched_at": p.models_fetched_at}
                 for name, p in PROVIDERS.items() if p.models}
        with open(MODELS_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _load_models_cache():
    try:
        if os.path.exists(MODELS_CACHE_PATH):
            with open(MODELS_CACHE_PATH) as f:
                cache = json.load(f)
            changed = False
            for name, data in cache.items():
                # Skip deleted providers — don't let stale cache resurrect them
                if name in DELETED_PROVIDERS:
                    cache.pop(name, None)
                    changed = True
                    continue
                p = PROVIDERS.get(name)
                if p:
                    p.models = data.get("models", [])
                    p.models_fetched_at = data.get("fetched_at", 0)
                else:
                    # Provider doesn't exist anymore — clean cache entry
                    cache.pop(name, None)
                    changed = True
            if changed:
                try:
                    with open(MODELS_CACHE_PATH, "w") as f:
                        json.dump(cache, f)
                except Exception:
                    pass
    except Exception:
        pass


def load_config():
    """(Re)load providers, proxies and combos from CONFIG."""
    global CONFIG, SERVER_CFG, ROUTER_CFG, API_KEY, DASH_USER, DASH_PASS, PROVIDER_MODEL_COSTS, DELETED_PROVIDERS
    PROVIDERS.clear()
    PROXIES.clear()
    COMBOS.clear()
    PROVIDER_MODEL_COSTS.clear()
    # Restore deleted provider names so sync_9router doesn't re-add them
    DELETED_PROVIDERS = set(CONFIG.get("deleted_providers", []))
    for p in CONFIG.get("providers", []):
        PROVIDERS[p["name"]] = Provider(
            name=p["name"],
            base_url=p["base_url"],
            prefix=p.get("prefix", ""),
            type_=p.get("type", "apikey"),
            keys=p.get("keys", []),
            weight=p.get("weight", 5),
            default_model=p.get("default_model", ""),
            proxy=p.get("proxy", ""),
            is_active=p.get("is_active", True),
            manual_models=p.get("manual_models", []),
            keep_prefix=p.get("keep_prefix", False),
        )
        # Phase 9 — provider-specific model cost overrides
        if "model_costs" in p:
            PROVIDER_MODEL_COSTS[p["name"]] = p["model_costs"]
    for pr in CONFIG.get("proxies", []):
        PROXIES[pr["name"]] = pr.get("url", "")
    for cb in CONFIG.get("combos", []):
        COMBOS[cb["name"]] = {"routes": cb.get("routes", []), "strategy": cb.get("strategy", "random"), "rr_idx": 0}


load_config()
load_history()
_load_models_cache()
vrouter_db.init_db()


def resolve_proxy_url(proxy_ref: str) -> Optional[str]:
    """proxy_ref can be a pool name (PROXIES key) or a full URL."""
    if not proxy_ref:
        return None
    if proxy_ref in PROXIES:
        return PROXIES[proxy_ref]
    if "://" in proxy_ref:
        return proxy_ref
    return None


def find_provider_by_prefix(prefix: str) -> Optional[Provider]:
    for p in PROVIDERS.values():
        if p.prefix and p.prefix == prefix:
            return p
    return None


def find_provider_by_model(model: str) -> Optional[Provider]:
    for p in PROVIDERS.values():
        if p.default_model and p.default_model == model:
            return p
    return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=ROUTER_CFG.get("request_timeout", 60))
    app.state.proxy_clients = {}   # proxy URL -> AsyncClient
    # Restore learned stats + dead models from disk (survive restarts)
    load_router_state()
    # Start background health check scheduler
    app.state.health_task = asyncio.create_task(health_check_loop(app))
    # Start background model probe (dead model detection)
    app.state.probe_task = asyncio.create_task(model_probe_loop(app))
    # Start 9router auto-sync scheduler
    app.state.sync_task = asyncio.create_task(sync_9router_loop(app))
    yield
    # Persist learned stats + dead models so restarts don't lose tuning
    save_router_state()
    app.state.health_task.cancel()
    app.state.probe_task.cancel()
    app.state.sync_task.cancel()
    await app.state.client.aclose()
    for pc in app.state.proxy_clients.values():
        await pc.aclose()


STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "router_state.json")


def save_router_state():
    """Persist MODEL_STATS + DEAD_MODELS + REASONING_MODELS atomically."""
    try:
        payload = {
            "model_stats": MODEL_STATS,
            "dead_models": DEAD_MODELS,
            "reasoning_models": sorted(REASONING_MODELS),
            "saved_at": time.time(),
        }
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        print(f"[state] save failed: {e}")


def load_router_state():
    """Load persisted state at startup. Missing/corrupt file → fresh state."""
    try:
        if not os.path.exists(STATE_FILE):
            return
        with open(STATE_FILE) as f:
            data = json.load(f)
        for key, s in (data.get("model_stats") or {}).items():
            MODEL_STATS[key] = s
        # Drop dead entries older than 1 day so reliably-broken models still
        # get a chance to revive (upstreams do recover).
        cutoff = time.time() - 86400
        for key, dm in (data.get("dead_models") or {}).items():
            if dm.get("disabled_at", 0) >= cutoff:
                DEAD_MODELS[key] = dm
        for m in data.get("reasoning_models") or []:
            REASONING_MODELS.add(m)
        print(f"[state] loaded {len(MODEL_STATS)} model stats, "
              f"{len(DEAD_MODELS)} dead models, {len(REASONING_MODELS)} reasoning")
    except Exception as e:
        print(f"[state] load failed (starting fresh): {e}")


def get_proxy_client(app: FastAPI, proxy_url: str) -> httpx.AsyncClient:
    """Reuse a cached AsyncClient per proxy URL (avoids closing mid-stream)."""
    pool = app.state.proxy_clients
    if proxy_url not in pool:
        pool[proxy_url] = httpx.AsyncClient(timeout=ROUTER_CFG.get("request_timeout", 60), proxy=proxy_url)
    return pool[proxy_url]


app = FastAPI(title="VRouter", version="2.0.0", lifespan=lifespan)


def check_gateway_key(authorization: Optional[str]):
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    key = authorization
    if key.lower().startswith("bearer "):
        key = key[7:]
    if key != API_KEY:
        raise HTTPException(401, "Invalid API key")


def check_dashboard_auth(request: Request):
    token = request.cookies.get("vr_token")
    if not token or token not in VALID_SESSIONS:
        raise HTTPException(401, "Dashboard login required")


def check_dashboard_or_gateway(request: Request):
    """Terima cookie dashboard ATAU gateway API key (Authorization: Bearer <key>)."""
    token = request.cookies.get("vr_token")
    if token and token in VALID_SESSIONS:
        return
    authz = request.headers.get("authorization", "")
    key = authz[7:] if authz.lower().startswith("bearer ") else authz
    if key and key == API_KEY:
        return
    raise HTTPException(401, "Dashboard login or API key required")


# --- Brute force protection ---
LOGIN_ATTEMPTS = defaultdict(list)  # ip -> [timestamps]
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 300  # 5 menit

def get_client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def check_rate_limit(ip: str):
    now = time.time()
    attempts = LOGIN_ATTEMPTS[ip]
    # hapus attempt yang lebih lama dari lockout window
    LOGIN_ATTEMPTS[ip] = [t for t in attempts if now - t < LOCKOUT_SECONDS]
    if len(LOGIN_ATTEMPTS[ip]) >= MAX_ATTEMPTS:
        oldest = LOGIN_ATTEMPTS[ip][0]
        remaining = int(LOCKOUT_SECONDS - (now - oldest))
        raise HTTPException(429, f"Too many attempts. Coba lagi dalam {remaining}s.")

def record_failed_attempt(ip: str):
    LOGIN_ATTEMPTS[ip].append(time.time())

def clear_attempts(ip: str):
    LOGIN_ATTEMPTS.pop(ip, None)

# --- Secure session tokens (persisted to file) ---
SESSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.vr_sessions.json')

def _load_sessions():
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, 'r') as f:
                data = json.load(f)
                # Only keep sessions from last 7 days
                now = time.time()
                return {k: v for k, v in data.items() if v > now - 604800}
    except Exception:
        pass
    return {}

def _save_sessions():
    try:
        with open(SESSION_FILE, 'w') as f:
            json.dump(VALID_SESSIONS, f)
    except Exception:
        pass

VALID_SESSIONS = _load_sessions()  # dict of token -> created_at


# ---------------------------------------------------------------------------
# Model routing (prefix, default, combo)
# ---------------------------------------------------------------------------
def parse_model(model: str):
    """Return (provider, upstream_model) — combos are resolved by caller."""
    if "/" in model:
        head, _, tail = model.partition("/")
        p = find_provider_by_prefix(head)
        if p:
            return p, tail
        p = PROVIDERS.get(head)
        if p:
            return p, tail
    p = find_provider_by_model(model)
    if p:
        return p, model
    candidates = sorted(PROVIDERS.values(), key=lambda x: x.weight)
    for p in candidates:
        if p.keys:
            return p, model
    return None, model


def health_score_for(model_id: str, now: Optional[float] = None) -> float:
    """Composite 0-100 health score for a "provider/model" key.
    Single source of truth shared by /api/health-score (display) and
    pick_combo_route (routing). Weights:
      reliability 45 (EMA success) + latency 25 (EMA, 300ms→full/8s→0)
      + throughput 20 (tok/s, 60→full) + freshness 10 (<5min→full/>6h→0).
    Circuit-open provider → score halved. No stats → returns neutral 50.0."""
    s = MODEL_STATS.get(model_id)
    if not s:
        return 50.0
    total = s.get("total", 0)
    if not total:
        return 50.0
    if now is None:
        now = time.time()
    prov = model_id.split("/", 1)[0]
    ema_succ = s.get("ema_success", s.get("ok", 0) / total if total else 0.0)
    ema_lat = s.get("ema_latency_ms", (s["latency_sum"] / total) if total else 0.0)
    rel = ema_succ * 45.0
    lat_norm = max(0.0, min(1.0, (8000.0 - ema_lat) / (8000.0 - 300.0)))
    lat_score = lat_norm * 25.0
    tp = THROUGHPUT_STATS.get(model_id)
    tok_s = tp["toks_ema"] if tp else 0.0
    tp_score = min(1.0, tok_s / 60.0) * 20.0 if tok_s else 0.0
    age = now - s.get("last_used", 0)
    fresh = max(0.0, min(1.0, (21600.0 - age) / (21600.0 - 300.0)))
    fresh_score = fresh * 10.0
    score = rel + lat_score + tp_score + fresh_score
    p = PROVIDERS.get(prov)
    if p and p.locked_until > now:
        score *= 0.5
    return round(max(0.0, min(100.0, score)), 1)


def pick_combo_route(combo_name: str, exclude_reasoning: bool = False):
    """Ordered combo candidates per strategy -> [(provider, model)].
    Strategy: random (score-weighted), round_robin (rotate), fallback (strict priority),
    cost_aware (cheapest healthy model first).
    Skips unhealthy/locked providers + dead models. Uses MODEL_STATS (latency/success)
    to rank routes so slow/dead-ish models fall behind fast ones."""
    combo = COMBOS.get(combo_name)
    if not combo or not combo["routes"]:
        return []
    routes = list(combo["routes"])
    strategy = combo.get("strategy", "random")

    # cost_aware: use dedicated picker
    if strategy == "cost_aware":
        return pick_cost_aware_route(combo_name, exclude_reasoning)

    # Filter out unhealthy routes + dead models (but keep unprobed = neutral)
    healthy_routes = []
    for r in routes:
        p = PROVIDERS.get(r.get("provider", ""))
        if not p or not p.is_active or not p.keys:
            continue
        if p.locked_until > time.time():
            continue
        model = r.get("model", p.default_model or "auto")
        model_id = f"{p.name}/{model}"
        dm = DEAD_MODELS.get(model_id)
        if dm and dm.get("disabled_at", 0) > 0 and time.time() - dm.get("disabled_at", 0) < 900:
            continue  # recently dead → skip (grace 15 min sebelum revive probe)
        if exclude_reasoning and model in REASONING_MODELS:
            continue
        healthy_routes.append(r)

    # If all routes unhealthy/dead, fall back to any provider with keys
    if not healthy_routes:
        healthy_routes = [r for r in routes if PROVIDERS.get(r.get("provider", "")) and PROVIDERS[r.get("provider", "")].is_active and PROVIDERS[r.get("provider", "")].keys]
        if not healthy_routes:
            return []

    def route_score(r):
        p = PROVIDERS.get(r.get("provider", ""))
        if not p or not p.is_active:
            return 0.0
        w = r.get("weight", 1) * max(1, p.weight)
        model_id = f"{p.name}/{r.get('model', p.default_model or 'auto')}"
        # Phase 5 — opt-in composite health-score routing. When enabled, blend the
        # 0-100 health score (reliability+latency+throughput+freshness) into the
        # weight so genuinely healthier models win, not just fast-but-flaky ones.
        if ROUTER_CFG.get("health_score_routing"):
            hs = health_score_for(model_id) / 100.0  # 0..1
            s = MODEL_STATS.get(model_id)
            if not s or s.get("samples", 0) < 2:
                return 0.5 * w  # unprobed → neutral, still tried
            return round(hs * w, 3)
        s = MODEL_STATS.get(model_id)
        if not s or s.get("samples", 0) < 2:
            return 0.5 * w  # belum ada data cukup, netral — tetap dicoba
        # Phase 4 — pakai EMA (reaktif ke slowdown/erro) bukan rata-rata kumulatif
        sr = s.get("ema_success", s["ok"] / max(s["total"], 1))
        avg = max(s.get("ema_latency_ms", 1), 1)
        # 2s -> 0.9, 5s -> 0.75, 15s -> 0.5, 30s -> 0.2 — kurva latency
        lat = max(0.05, 1.0 - avg / 8000.0)
        if s["err"] >= 3 and sr < 0.6:
            lat *= 0.2  # error beruntun — drop jauh
        return round(sr * lat * w, 3)

    if strategy == "round_robin":
        idx = combo.get("rr_idx", 0) % len(healthy_routes)
        combo["rr_idx"] = (idx + 1) % len(healthy_routes)
        order = healthy_routes[idx:] + healthy_routes[:idx]
    elif strategy == "fallback":
        order = healthy_routes  # priority as listed
    else:  # random / smart — score-weighted
        scored = sorted(healthy_routes, key=route_score, reverse=True)
        top = scored[:max(2, len(scored) // 3)]
        if not top:
            top = scored
        rnd = random.uniform(0, 1)
        # 70%: roulette-wheel pick from scored (score = probability weight);
        # 30%: explore any route (anti-stuck bias).
        # Roulette wheel makes 5x-faster models ~ (score ratio) more likely,
        # instead of flat random.choice over a small top pool.
        pool = scored
        if rnd >= 0.7:
            # exploration: uniform pick among all
            first = random.choice(healthy_routes)
        else:
            scores = [max(route_score(r), 0.001) for r in pool]
            total = sum(scores)
            pick = random.uniform(0, total)
            acc = 0
            first = pool[-1]
            for r, sc in zip(pool, scores):
                acc += sc
                if pick <= acc:
                    first = r
                    break
        rest = [r for r in pool if r is not first]
        order = [first] + rest

    result = []
    for r in order:
        p = PROVIDERS.get(r.get("provider", ""))
        if p and p.keys:
            result.append((p, r.get("model", p.default_model or "auto")))
    return result


def mark_model_error(provider_name: str, model: str, reason: str) -> None:
    """Live-track model failure dari request path (bukan cuma probe).
    3x error → disabled_at set, combo skipping mulai berlaku."""
    model_id = f"{provider_name}/{model}"
    if model_id not in DEAD_MODELS:
        DEAD_MODELS[model_id] = {"failures": 0, "last_checked": time.time(), "last_error": None, "disabled_at": 0}
    dm = DEAD_MODELS[model_id]
    dm["failures"] += 1
    dm["last_checked"] = time.time()
    dm["last_error"] = reason[:120]
    if dm["failures"] >= 3:
        dm["disabled_at"] = time.time()


def mark_key_429(provider_name: str, key: str) -> None:
    """Per-key exponential cooldown setelah 429 — provider lain & key lain tetap hidup.
    (Ganti lagi lock-provider 30s yang ngerjamadiluhungin semua key sekaligus.)"""
    kk = f"{provider_name}|{key}"
    b = KEY_429_BURST.get(kk, 0) + 1
    KEY_429_BURST[kk] = b
    cool = min(60.0, max(1.0, 2.0 ** b))   # 2s,4s,...,60s
    KEY_COOLDOWN[kk] = time.time() + cool


def mark_key_ok(provider_name: str, key: str) -> None:
    """Reset cooldown/burst setelah key respon 200."""
    kk = f"{provider_name}|{key}"
    if kk in KEY_429_BURST:
        KEY_429_BURST[kk] = max(0, KEY_429_BURST[kk] - 1)
    KEY_COOLDOWN.pop(kk, None)
    KEY_LAST_SENT[kk] = time.time()


def record_key_result(provider_name: str, key: str, latency_ms: float, success: bool) -> None:
    """Phase E — EMA per-key untuk weighted selection. success=True memutihkan burst penalty."""
    kk = f"{provider_name}|{key}"
    a = KEY_ALPHA
    ok = 1.0 if success else 0.0
    s = KEY_STATS.get(kk)
    if s:
        s["ema_ms"] = a * latency_ms + (1 - a) * s["ema_ms"]
        s["ema_ok"] = a * ok + (1 - a) * s["ema_ok"]
        s["samples"] = s.get("samples", 0) + 1
    else:
        KEY_STATS[kk] = {"ema_ms": latency_ms, "ema_ok": ok, "samples": 1, "last_latency_ms": latency_ms}
    KEY_STATS[kk]["last_latency_ms"] = latency_ms


def key_score(kk: str):
    """Return score = latency_ms / (ema_ok + 0.1) — lower = lebih cepat+reliable. None = belum cukup sample."""
    s = KEY_STATS.get(kk)
    if not s or s.get("samples", 0) < KEY_MIN_SAMPLES:
        return None
    return s["ema_ms"] / (s["ema_ok"] + 0.1)


# ---------------------------------------------------------------------------
# Phase 4 — Speculative hedge (stream=true)
# ---------------------------------------------------------------------------
async def _stream_worker(provider: Provider, upstream_model: str, body: dict,
                         base_headers: dict, app: FastAPI, signal: asyncio.Event,
                         out_q: asyncio.Queue, box: dict):
    """Hedge worker: stream SATU candidate upstream ke (signal, out_q).
    `signal` set saat byte pertama sampai (winner election). Error/429 dicatat."""
    proxy_url = resolve_proxy_url(provider.proxy)
    client = get_proxy_client(app, proxy_url) if proxy_url else app.state.client
    url = f"{provider.base_url}/chat/completions"
    key = provider.next_key()
    if not key:
        await out_q.put(("err", 0, "no keys"))
        return
    headers = {**base_headers, **provider.auth_header(key)}
    payload = {**body, "model": upstream_model}          # stream=True tetap di body
    if provider.keep_prefix and provider.prefix and not upstream_model.startswith(f"{provider.prefix}/"):
        payload["model"] = f"{provider.prefix}/{upstream_model}"
    req = client.build_request("POST", url, json=payload, headers=headers)
    t0 = time.time()
    resp = None
    try:
        resp = await client.send(req, stream=True)
        if resp.status_code >= 400:
            err_text = ""
            try:
                err_text = resp.text[:200]
            except Exception:
                pass
            await out_q.put(("err", resp.status_code, f"HTTP {resp.status_code}"))
            mark_model_error(provider.name, upstream_model, f"HTTP {resp.status_code}")
            if resp.status_code == 429:
                mark_key_429(provider.name, key)
            return
        async for raw in resp.aiter_bytes():
            if not signal.is_set():
                signal.set()
                box["ms"] = int((time.time() - t0) * 1000)   # first-byte latency
            await out_q.put(("data", raw))
        await out_q.put(("end", None))
    except Exception as e:
        try:
            await out_q.put(("err", 0, str(e)[:200]))
        except Exception:
            pass
        mark_model_error(provider.name, upstream_model, f"exc:{str(e)[:60]}")
    finally:
        if resp is not None:
            try:
                await resp.aclose()
            except Exception:
                pass


async def _hedge_stream(app: FastAPI, candidates: list, body: dict,
                        base_headers: dict, race_timeout: float):
    """Speculative hedge: top-2 combo candidates dikirim PARALEL.
    Pemenang = yang pertama kirim byte. Loser dibatalkan. Kalau tidak ada
    pemenang dalam race_timeout → fallback ke jalur sekunder (sequential).
    Return (True, gen, info) | (False, None, None)."""
    workers = []
    for (p, m) in candidates:
        sig = asyncio.Event()
        q = asyncio.Queue(maxsize=256)
        box = {"ms": 0}
        t = asyncio.create_task(_stream_worker(p, m, body, base_headers, app, sig, q, box))
        workers.append({"task": t, "p": p, "m": m, "sig": sig, "q": q, "box": box})
    # race: Event mana yang set lebih dulu (FIRST_COMPLETED)
    probes = {asyncio.ensure_future(w["sig"].wait()): w for w in workers}
    done, pending = await asyncio.wait(list(probes.keys()), timeout=race_timeout,
                                       return_when=asyncio.FIRST_COMPLETED)
    winner = None
    if done:
        wt = next(iter(done))
        winner = probes[wt]
    for pr in pending:
        pr.cancel()
    for w in workers:
        if w is not winner:
            w["task"].cancel()
    if not winner:
        for w in workers:
            if not w["task"].done():
                w["task"].cancel()
        return False, None, None

    async def gen():
        q = winner["q"]
        buf = b""
        p_tok = 0
        c_tok = 0
        _completion_text = []   # local fallback when upstream omits usage
        first_byte_ms = winner["box"].get("ms", 0)
        try:
            while True:
                try:
                    item = await q.get()
                except asyncio.CancelledError:
                    return
                kind = item[0]
                if kind == "end":
                    return
                if kind == "err":
                    err = json.dumps({"id": "hedge_error", "object": "chat.completion.chunk",
                                      "created": int(time.time()), "model": winner["m"],
                                      "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                    yield f"data: {err}\n\n".encode()
                    return
                chunk = item[1]      # ("data", raw_bytes)
                yield chunk
                # sniff usage from SSE tail
                try:
                    buf = (buf + chunk)[-8192:]
                    for line in buf.split(b"\n"):
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if payload in (b"[DONE]", b""):
                            continue
                        try:
                            obj = json.loads(payload)
                        except Exception:
                            continue
                        u = obj.get("usage") if isinstance(obj, dict) else None
                        if isinstance(u, dict):
                            p_tok = u.get("prompt_tokens", p_tok) or p_tok
                            c_tok = u.get("completion_tokens", c_tok) or c_tok
                        # local fallback: accumulate streamed assistant text
                        if isinstance(obj, dict):
                            dt = _extract_delta_text(obj)
                            if dt:
                                _completion_text.append(dt)
                except Exception:
                    pass
        finally:
            t = winner["task"]
            if not t.done():
                t.cancel()
            try:
                # Many free upstreams never emit a usage chunk on streams →
                # p_tok/c_tok stay 0. Fill from local token counting so the
                # Logs In/Out/Cost columns aren't blank.
                if not p_tok:
                    p_tok = count_prompt_tokens(body)
                if not c_tok and _completion_text:
                    c_tok = count_tokens("".join(_completion_text))
                est = estimate_request_cost(winner["p"], winner["m"], p_tok, c_tok)
                log_history({"provider": winner["p"].name, "model": winner["m"],
                             "req_model": body.get("model", ""), "status": "hedge_ok",
                             "code": 200, "ms": first_byte_ms,
                             "proxy": winner["p"].proxy or "direct", "stream": True,
                             "prompt_tokens": p_tok, "completion_tokens": c_tok,
                             "total_tokens": p_tok + c_tok, "est_cost_usd": round(est, 8)})
            except Exception:
                pass

    info = {"provider": winner["p"], "model": winner["m"], "first_byte_ms": winner["box"]["ms"]}
    return True, gen(), info


# ---------------------------------------------------------------------------
# Endpoints — API (gateway)
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    check_gateway_key(authorization)
    data = []
    seen = set()
    # Combos first — they lead the /model list ahead of individual provider models
    for name in COMBOS:
        if name not in seen:
            data.append({"id": name, "object": "model", "created": 0, "owned_by": "combo"})
            seen.add(name)
    for p in PROVIDERS.values():
        if p.prefix:
            # Include all effective models (manual + auto) with prefix
            for m in p.effective_models:
                mid = f"{p.prefix}/{m}"
                if mid not in seen:
                    data.append({"id": mid, "object": "model", "created": 0, "owned_by": p.name})
                    seen.add(mid)
    # Anthropic aliases (so /v1/messages clients see claude-* models)
    for alias in list(ANTHROPIC_MODEL_MAP.keys()) + ([ANTHROPIC_DEFAULT_MODEL] if ANTHROPIC_DEFAULT_MODEL else []):
        if alias and alias not in seen:
            data.append({"id": alias, "object": "model", "created": 0, "owned_by": "anthropic"})
            seen.add(alias)
    return {"object": "list", "data": data}


# ---------------------------------------------------------------------------
# Anthropic-format compatibility layer (/v1/messages)
# Translate Anthropic request <-> OpenAI internally, reusing the mature
# chat_completions core (hedge, failover, retries, stats, logging).
# ---------------------------------------------------------------------------
ANTHROPIC_CFG = CONFIG.get("anthropic", {})
ANTHROPIC_MODEL_MAP = ANTHROPIC_CFG.get("model_map", {})
ANTHROPIC_DEFAULT_MODEL = ANTHROPIC_CFG.get("default_model", "")


def _resolve_anthropic_model(model: str) -> str:
    """Map an Anthropic model name (claude-*) to a VRouter combo / provider model.
    1) combo passthrough  2) exact map  3) fuzzy map  4) default for claude-*  5) as-is."""
    if not model:
        return ANTHROPIC_DEFAULT_MODEL or ""
    if model in COMBOS:
        return model  # allow direct combo calls through /v1/messages
    if model in ANTHROPIC_MODEL_MAP:
        return ANTHROPIC_MODEL_MAP[model]
    base = model.lower()
    for k, v in ANTHROPIC_MODEL_MAP.items():
        if k.lower() in base or base in k.lower():
            return v
    if base.startswith("claude") and ANTHROPIC_DEFAULT_MODEL:
        return ANTHROPIC_DEFAULT_MODEL
    return model


def _anthropic_content_to_openai(content):
    """Anthropic content (str OR block list) -> OpenAI content (str OR parts list).
    Handles text / image (base64 data URL) / tool_result blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        text_buf = []
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text":
                text_buf.append(block.get("text", ""))
            elif t == "image":
                if text_buf:
                    parts.append({"type": "text", "text": "\n".join(text_buf)})
                    text_buf = []
                src = block.get("source", {})
                if src.get("type") == "base64":
                    media = src.get("media_type", "image/png")
                    data = src.get("data", "")
                    parts.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}})
                elif src.get("type") == "url":
                    parts.append({"type": "image_url", "image_url": {"url": src.get("url", "")}})
            elif t == "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, list):
                    rc = " ".join(b.get("text", "") for b in rc if isinstance(b, dict))
                text_buf.append(f"[tool_result]\n{rc}")
        if text_buf:
            parts.append({"type": "text", "text": "\n".join(text_buf)})
        if not parts:
            return ""
        if len(parts) == 1 and parts[0]["type"] == "text":
            return parts[0]["text"]
        return parts
    return str(content)


def _anthropic_assistant_to_openai(msg: dict) -> dict:
    """Convert an Anthropic assistant message (text + tool_use blocks) to OpenAI."""
    content = msg.get("content")
    tool_calls = []
    text_parts = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text":
                text_parts.append(block.get("text", ""))
            elif t == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })
    out = {"role": "assistant"}
    out["content"] = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _anthropic_tools_to_openai(tools):
    """Anthropic tools [{name,description,input_schema}] -> OpenAI [{type:function,...}]."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"})
            }
        })
    return out if out else None


def _anthropic_tool_choice_to_openai(tc):
    """Anthropic tool_choice {auto|any|tool:{name}} -> OpenAI tool_choice."""
    if tc is None:
        return None
    if isinstance(tc, str):
        return tc  # "auto" / "required" passthrough
    if isinstance(tc, dict):
        t = tc.get("type")
        if t == "auto":
            return "auto"
        if t == "any":
            return "required"
        if t == "tool":
            name = tc.get("name", "")
            return {"type": "function", "function": {"name": name}}
    return None


def _anthropic_user_to_openai(m: dict) -> list:
    """Anthropic user message may mix text + tool_result blocks. OpenAI requires
    tool results as separate {role: tool, tool_call_id} messages, so split them."""
    content = m.get("content")
    if not isinstance(content, list):
        return [{"role": "user", "content": _anthropic_content_to_openai(content)}]
    text_buf = []
    out = []
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "tool_result":
            if text_buf:
                out.append({"role": "user", "content": "\n".join(text_buf)})
                text_buf = []
            rc = block.get("content", "")
            if isinstance(rc, list):
                rc = " ".join(b.get("text", "") for b in rc if isinstance(b, dict))
            tool_msg = {"role": "tool", "tool_call_id": block.get("tool_use_id", ""), "content": rc}
            if block.get("is_error"):
                tool_msg["content"] = f"[ERROR] {rc}"
            out.append(tool_msg)
        else:
            text_buf.append(str(block.get("text", "")))
    if text_buf:
        out.append({"role": "user", "content": "\n".join(text_buf)})
    return out


def _extract_resp_text(resp_data, is_chunk: bool = False) -> str:
    """Pull assistant plaintext out of an OpenAI chat.completion (or stream chunk)."""
    if not isinstance(resp_data, dict):
        return ""
    ch = resp_data.get("choices") or []
    if ch:
        slot = ch[0].get("delta") if is_chunk else (ch[0].get("message") or {})
        txt = slot.get("content") if is_chunk else (slot.get("content") or slot.get("reasoning_content") or "")
        if is_chunk:
            txt = slot.get("content") or slot.get("reasoning_content") or ""
        if isinstance(txt, list):
            txt = " ".join(b.get("text", "") for b in txt if isinstance(b, dict))
        return txt or ""
    return ""


# Response quality gate (upgrade #2). Detect empty / refusal / repetition /
# garbage so a bad upstream result cascades to the next healthy provider
# instead of surfacing to the client.
_REFUSAL_PATTERNS = [
    "i'm sorry", "i am sorry", "sorry, but", "i cannot", "i can't",
    "i won't", "i will not", "i'm not able to", "i am not able to",
    "i'm unable to", "i am unable to", "i must decline", "i must refuse",
    "i'm not allowed", "as an ai", "as an artificial intelligence",
    "i'm not permitted to", "i cannot provide", "i can only provide",
    "against my guidelines", "designed to be helpful", "i'm a language model",
]
_REFUSAL_RE = re.compile("|".join(re.escape(p) for p in _REFUSAL_PATTERNS), re.IGNORECASE)
# Start-anchored variant: refusal where the assistant *opens* with an apology/decline.
_REFUSAL_START_RE = re.compile(
    r"^(?:\s*(?:" + "|".join(re.escape(p) for p in _REFUSAL_PATTERNS) + r"))",
    re.IGNORECASE)
_REPEAT_RE = re.compile(r"(\b[\w.,?!']{2,}\b\s*)\1{3,}", re.IGNORECASE)


def quality_check(resp_data, is_chunk: bool = False):
    """Return (ok: bool, reason: str). ok=False if response is empty/refusal/repeat/garbage."""
    text = _extract_resp_text(resp_data, is_chunk)
    if not text or not text.strip():
        return False, "empty content"
    low = text.lower().strip()
    words = low.split()
    # Flag a refusal only when the reply *starts* with a refusal phrase (permulaan),
    # OR it is a short apology-only message (<=14 words) that contains one.
    # Long, substantive answers that merely mention "I can't" mid-sentence pass.
    if _REFUSAL_START_RE.search(low):
        return False, "refusal pattern"
    if len(words) <= 10 and _REFUSAL_RE.search(low):
        return False, "refusal pattern"
    # unicode/garbage ratio: >10% non-printable junk in a real response
    if len(text) > 30:
        non_text = sum(1 for c in text if not (c.isprintable() or c in "\n\r\t "))
        if non_text / len(text) > 0.10:
            return False, "unicode garbage ratio"
    if _REPEAT_RE.search(text):
        return False, "repetition loop"
    return True, ""


def _anthropic_to_openai_body(body: dict) -> dict:
    """Translate an Anthropic /v1/messages request into an OpenAI chat.completions body."""
    obody = dict(body)
    # Model resolution: combo passthrough > map > default > as-is
    obody["model"] = _resolve_anthropic_model(body.get("model", ""))
    # system top-level -> prepend system message
    system = body.get("system")
    msgs = []
    if system:
        if isinstance(system, list):
            system = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
        msgs.append({"role": "system", "content": system})
    for m in body.get("messages", []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            msgs.append(_anthropic_assistant_to_openai(m))
        elif role == "user":
            msgs.extend(_anthropic_user_to_openai(m))
        else:
            msgs.append({"role": role, "content": _anthropic_content_to_openai(m.get("content"))})
    obody["messages"] = msgs
    # tools + tool_choice
    tools = _anthropic_tools_to_openai(body.get("tools"))
    if tools:
        obody["tools"] = tools
    else:
        obody.pop("tools", None)
    tc = _anthropic_tool_choice_to_openai(body.get("tool_choice"))
    if tc:
        obody["tool_choice"] = tc
    else:
        obody.pop("tool_choice", None)
    # stop_sequences -> stop; drop Anthropic-only fields that break OpenAI upstreams
    if body.get("stop_sequences"):
        obody["stop"] = body["stop_sequences"]
    obody.pop("stop_sequences", None)
    obody.pop("thinking", None)          # not universal upstream; strip to avoid breakage
    obody.pop("metadata", None)
    obody["max_tokens"] = body.get("max_tokens", 1024)
    return obody


def _safe_json_load(s: str):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def _openai_to_anthropic_response(data: dict, req_model: str) -> dict:
    """Convert an OpenAI chat.completion (non-stream) into an Anthropic message."""
    choices = data.get("choices") or []
    content = []
    stop_reason = "end_turn"
    if choices:
        ch = choices[0]
        msg = ch.get("message") or {}
        txt = msg.get("content") or ""
        if txt:
            content.append({"type": "text", "text": txt})
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", "toolu_1"),
                "name": fn.get("name", ""),
                "input": _safe_json_load(fn.get("arguments", "{}"))
            })
        fr = ch.get("finish_reason")
        stop_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                    "function_call": "tool_use", "content_filter": "end_turn"}
        stop_reason = stop_map.get(fr, "end_turn")
    usage = data.get("usage") or {}
    return {
        "id": "msg_" + str(data.get("id", "msg_1")).replace("chatcmpl-", ""),
        "type": "message",
        "role": "assistant",
        "content": content if content else [{"type": "text", "text": ""}],
        "model": req_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0) or 0,
            "output_tokens": usage.get("completion_tokens", 0) or 0,
        }
    }


class AnthropicSSE:
    """Translate OpenAI chat.completion.chunk SSE stream -> Anthropic message SSE events.
    State machine: emits message_start -> content_block_start/delta/stop per block ->
    message_delta -> message_stop. Handles text, reasoning_content, and tool_calls."""

    def __init__(self, req_model: str, prompt_tokens: int = 0):
        self.msg_id = "msg_" + secrets.token_hex(12)
        self.req_model = req_model
        self.prompt_tokens = prompt_tokens
        self.content_blocks = []   # list of (kind, index)
        self.tool_blocks = {}      # openai tool index -> anthropic block index
        self._tool_names = {}      # openai tool index -> name (may arrive late)
        self.block_index = 0
        self.started = False
        self.stopped = False
        self.output_tokens = 0
        self.acc_text = []

    def _event(self, event: str, data: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    def message_start(self) -> bytes:
        return self._event("message_start", {
            "type": "message_start",
            "message": {
                "id": self.msg_id, "type": "message", "role": "assistant",
                "content": [], "model": self.req_model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": self.prompt_tokens, "output_tokens": 0}
            }})

    def _start_text_block(self) -> bytes:
        idx = self.block_index
        self.block_index += 1
        self.content_blocks.append(("text", idx))
        return self._event("content_block_start", {
            "type": "content_block_start", "index": idx,
            "content_block": {"type": "text", "text": ""}})

    def _start_tool_block(self, tidx: int, name: str) -> bytes:
        idx = self.block_index
        self.block_index += 1
        self.tool_blocks[tidx] = idx
        self.content_blocks.append(("tool", idx))
        tid = f"toolu_{tidx}"
        return self._event("content_block_start", {
            "type": "content_block_start", "index": idx,
            "content_block": {"type": "tool_use", "id": tid, "name": name, "input": {}}})

    def _text_delta(self, idx: int, text: str) -> bytes:
        return self._event("content_block_delta", {
            "type": "content_block_delta", "index": idx,
            "delta": {"type": "text_delta", "text": text}})

    def _input_delta(self, idx: int, partial: str) -> bytes:
        return self._event("content_block_delta", {
            "type": "content_block_delta", "index": idx,
            "delta": {"type": "input_json_delta", "partial_json": partial}})

    def _stop_blocks(self) -> bytes:
        out = b""
        for _kind, idx in self.content_blocks:
            out += self._event("content_block_stop", {"type": "content_block_stop", "index": idx})
        self.content_blocks = []
        return out

    def finish(self, stop_reason: str = "end_turn") -> bytes:
        out = self._stop_blocks()
        out += self._event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": self.output_tokens}})
        out += self._event("message_stop", {"type": "message_stop"})
        self.stopped = True
        return out

    def feed(self, obj: dict) -> bytes:
        """Process one OpenAI chunk; return translated SSE bytes (may be empty)."""
        out = b""
        if not self.started:
            self.started = True
            out += self.message_start()
        choices = obj.get("choices") or []
        usage = obj.get("usage") or {}
        if usage:
            self.output_tokens = max(self.output_tokens, usage.get("completion_tokens", 0) or 0)
        if not choices:
            return out
        ch = choices[0]
        delta = ch.get("delta") or {}
        txt = delta.get("content")
        if txt:
            open_text = [i for k, i in self.content_blocks if k == "text"]
            if not open_text:
                out += self._start_text_block()
                open_text = [i for k, i in self.content_blocks if k == "text"]
            out += self._text_delta(open_text[-1], txt)
            self.acc_text.append(txt)
        rc = delta.get("reasoning_content") or delta.get("reasoning")
        if rc and not txt:
            open_text = [i for k, i in self.content_blocks if k == "text"]
            if not open_text:
                out += self._start_text_block()
                open_text = [i for k, i in self.content_blocks if k == "text"]
            out += self._text_delta(open_text[-1], rc)
            self.acc_text.append(rc)
        for tci in delta.get("tool_calls") or []:
            if not isinstance(tci, dict):
                continue
            tidx = tci.get("index", 0)
            fn = tci.get("function") or {}
            if tidx not in self.tool_blocks:
                # OpenAI tool_calls are keyed by index; id/name can arrive on a
                # later chunk. Reserve the block by index, fill id/name when seen.
                if self._tool_names.get(tidx) is None:
                    self._tool_names[tidx] = fn.get("name", "") or ""
                out += self._start_tool_block(tidx, self._tool_names[tidx])
            # id/name may arrive after block start — update stored identity
            if fn.get("name"):
                self._tool_names[tidx] = fn["name"]
            args = fn.get("arguments")
            if args:
                out += self._input_delta(self.tool_blocks[tidx], args)
        fr = ch.get("finish_reason")
        if fr and not self.stopped:
            stop_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                        "function_call": "tool_use", "content_filter": "end_turn"}
            if not self.output_tokens and self.acc_text:
                self.output_tokens = count_tokens("".join(self.acc_text))
            out += self.finish(stop_map.get(fr, "end_turn"))
        return out


class _ShimRequest:
    """Minimal stand-in for FastAPI Request — lets /v1/messages reuse chat_completions
    with a pre-translated body while keeping the same app.state (clients, stats)."""
    def __init__(self, body: dict, app: FastAPI):
        self._body = body
        self.app = app
        self.client = getattr(app.state, "client", None)
        self.state = app.state

    async def json(self):
        return self._body


@app.post("/v1/messages")
async def anthropic_messages(request: Request,
                             x_api_key: Optional[str] = Header(None),
                             authorization: Optional[str] = Header(None),
                             anthropic_version: Optional[str] = Header(None)):
    """Anthropic Messages API (Claude Code / Anthropic SDK compatible).
    Accepts x-api-key (Anthropic style) OR Authorization: Bearer (OpenAI style)."""
    key = x_api_key or authorization or ""
    check_gateway_key(key)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(400, "Body must be a JSON object")
    stream = bool(body.get("stream", False))
    req_model = body.get("model", "")
    openai_body = _anthropic_to_openai_body(body)
    openai_body["stream"] = stream
    if stream and "stream_options" not in openai_body:
        openai_body["stream_options"] = {"include_usage": True}
    # Reuse the full chat_completions core (hedge/failover/retry/stats/log).
    try:
        resp = await chat_completions(_ShimRequest(openai_body, request.app),
                                      authorization=f"Bearer {API_KEY}")
    except HTTPException as e:
        # OpenAI-style 502 detail -> Anthropic-style error envelope
        try:
            detail = json.loads(e.detail) if isinstance(e.detail, str) else e.detail
        except Exception:
            detail = {"message": str(e.detail)}
        err = detail.get("error", {}) if isinstance(detail, dict) else {}
        return JSONResponse(
            {"type": "error", "error": {
                "type": err.get("type", "api_error"),
                "message": err.get("message", str(e.detail))}},
            status_code=e.status_code)
    if not stream:
        data = json.loads(resp.body) if isinstance(resp.body, (bytes, bytearray)) else resp.body
        if isinstance(data, dict) and data.get("object") == "chat.completion":
            return JSONResponse(_openai_to_anthropic_response(data, req_model))
        return resp  # error passthrough

    translator = AnthropicSSE(req_model, count_prompt_tokens(openai_body))

    async def _translate():
        try:
            async for chunk in resp.body_iterator:
                text = chunk.decode("utf-8", "ignore") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                for line in text.splitlines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        if not translator.stopped:
                            yield translator.finish("end_turn")
                        return
                    try:
                        obj = json.loads(payload)
                    except Exception:
                        continue
                    ev = translator.feed(obj)
                    if ev:
                        yield ev
            if not translator.stopped:
                yield translator.finish("end_turn")
        except Exception:
            if not translator.stopped:
                yield translator.finish("end_turn")

    return StreamingResponse(_translate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    check_gateway_key(authorization)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    model = body.get("model", "")
    stream = body.get("stream", False)
    # Ask upstreams to emit a usage chunk on streaming responses so we can log
    # real in/out tokens (OpenAI-compatible stream_options). Harmless if ignored.
    if stream and "stream_options" not in body:
        body["stream_options"] = {"include_usage": True}
    base_headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA}
    max_attempts = ROUTER_CFG.get("max_fallback_attempts", 3)
    route_timeout = ROUTER_CFG.get("route_timeout", 45)  # per-attempt timeout non-stream
    client = request.app.state.client

    # Resolve model: combo first, then prefix/default routing
    candidates = []
    if model in COMBOS:
        # Pass exclude_reasoning=True if request doesn't explicitly ask for reasoning
        candidates = pick_combo_route(model, exclude_reasoning=True)
        # If all routes were reasoning-only, fall back to include them
        if not candidates:
            candidates = pick_combo_route(model, exclude_reasoning=False)
    else:
        primary, upstream_model = parse_model(model)
        if primary:
            candidates.append((primary, upstream_model))
        for p in sorted(PROVIDERS.values(), key=lambda x: x.weight):
            if p is not primary and p.keys:
                candidates.append((p, upstream_model if not primary else p.default_model or upstream_model))

    # dedupe by (provider, model) pair — combo routes share one provider but
    # differ by model; deduping by provider alone would collapse the whole
    # combo to a single model (root cause of "all_failed after 1 model")
    seen_pm = set()
    candidates = [(p, m) for p, m in candidates
                  if not ((p.name, m) in seen_pm or seen_pm.add((p.name, m)))]

    # Phase 4 — speculative hedge: stream=true + combo → fire top-2 parallel,
    # winner = first byte. Jika tidak ada pemenang dalam race_timeout, jatuh
    # ke jalur sekunder (sequential) di bawah.
    if stream and model in COMBOS and len(candidates) >= 2 and SPECULATIVE_HEDGE:
        ok, gen, winner = await _hedge_stream(
            request.app, candidates[:2], body, base_headers, HEDGE_RACE_TIMEOUT)
        if ok:
            update_model_stats(winner["provider"].name, winner["model"],
                               "ok", winner["first_byte_ms"], False)
            # NOTE: history is logged inside gen()'s finally block once the
            # stream completes, so we capture real in/out tokens there.
            return StreamingResponse(gen, media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    errors = []
    t_start = time.time()
    max_retries_per_provider = ROUTER_CFG.get("max_retries", 2)
    backoff_base = ROUTER_CFG.get("backoff_base_seconds", 2)
    
    for attempt, (provider, upstream_model) in enumerate(candidates[:max_attempts]):
        if provider.locked_until > time.time():
            errors.append(f"{provider.name}: circuit breaker open (locked {int(provider.locked_until - time.time())}s)")
            continue
        if not provider.keys:
            errors.append(f"{provider.name}: no keys")
            continue

        # Retry loop per provider (for 429/500 with backoff)
        for retry in range(max_retries_per_provider + 1):
            key = provider.next_key()
            url = f"{provider.base_url}/chat/completions"
            payload = {**body, "model": upstream_model}
            if provider.keep_prefix and provider.prefix and not upstream_model.startswith(f"{provider.prefix}/"):
                payload["model"] = f"{provider.prefix}/{upstream_model}"
            # Browser UA: some upstreams (ipeenk/Cloudflare) 403 default python-httpx UA
            headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA, **provider.auth_header(key)}
            proxy_url = resolve_proxy_url(provider.proxy)
            t_attempt = time.time()

            provider.total_requests += 1
            provider.last_used = time.time()
            try:
                if proxy_url:
                    pclient = get_proxy_client(request.app, proxy_url)
                    if stream:
                        req = pclient.build_request("POST", url, json=payload, headers=headers)
                        resp = await pclient.send(req, stream=True)
                    else:
                        resp = await asyncio.wait_for(
                            pclient.post(url, json=payload, headers=headers),
                            timeout=route_timeout)
                else:
                    if stream:
                        req = client.build_request("POST", url, json=payload, headers=headers)
                        resp = await client.send(req, stream=True)
                    else:
                        resp = await asyncio.wait_for(
                            client.post(url, json=payload, headers=headers),
                            timeout=route_timeout)

                if resp.status_code >= 400:
                    # Streaming responses can't read .text without read() —
                    # read a bounded chunk first so the real error surfaces.
                    if stream:
                        try:
                            err_text = (await asyncio.wait_for(resp.aread(), timeout=5)).decode("utf-8", "ignore")[:500]
                        except Exception:
                            err_text = f"HTTP {resp.status_code} (stream error body)"
                    else:
                        err_text = resp.text[:500]
                    provider.last_error = f"[{resp.status_code}] {err_text}"
                    provider.total_errors += 1
                    latency_ms = (time.time() - t_attempt) * 1000
                    update_model_stats(provider.name, upstream_model, f"HTTP {resp.status_code}", latency_ms)
                    log_history({"provider": provider.name, "model": upstream_model, "req_model": model,
                                 "status": "error", "code": resp.status_code, "ms": latency_ms,
                                 "err": err_text[:120], "proxy": provider.proxy or "direct", "retry": retry})

                    # Live dead-model tracking: permanent-ish errors mark model,
                    # so combo routing skips it on the NEXT request (fast feedback).
                    if resp.status_code in (401, 403, 404, 405, 429, 500, 502, 503, 504):
                        mark_model_error(provider.name, upstream_model, f"HTTP {resp.status_code}")

                    # Retry logic: 429 → per-key exponential cooldown + rotate key;
                    # 500 → retry once; 502/503 → skip to next route immediately.
                    if resp.status_code == 429 and retry < max_retries_per_provider:
                        record_key_result(provider.name, key, latency_ms, False)
                        mark_key_429(provider.name, key)
                        errors.append(f"{provider.name}: HTTP 429 (key rotated, retry {retry+1})")
                        await asyncio.sleep(KEY_MIN_INTERVAL)  # pendek — next_key loncat key yang cooldown
                        continue  # retry same provider w/ key berikutunya
                    elif resp.status_code == 500 and retry < max_retries_per_provider:
                        backoff = backoff_base * (2 ** retry)
                        errors.append(f"{provider.name}: HTTP 500 (retry {retry+1} in {backoff}s)")
                        await asyncio.sleep(min(backoff, 8))
                        continue  # retry same provider
                    else:
                        if resp.status_code == 429:
                            # semua key ini cooldown → circuit-break provider sebentar
                            trip_circuit(provider, f"HTTP 429 all-keys-cooldown", CB_LOCK_SECONDS)
                            record_key_result(provider.name, key, latency_ms, False)
                            mark_key_429(provider.name, key)
                        errors.append(f"{provider.name}: HTTP {resp.status_code} {err_text[:200]}")
                        break  # move to next provider candidate

                # Success
                DEAD_MODELS.pop(f"{provider.name}/{upstream_model}", None)
                mark_key_ok(provider.name, key)
                latency_ms = (time.time() - t_attempt) * 1000
                record_key_result(provider.name, key, latency_ms, True)

                if not stream:
                    resp_data = parse_upstream_json(resp)
                    if resp_data is None:
                        raise ValueError(f"non-JSON body {resp.text[:100]!r}")
                    # --- Response quality gate (upgrade #2) ---
                    # On empty / refusal / repetition / garbage, fail over to the
                    # next healthy provider instead of returning a bad response.
                    _qok, _qreason = quality_check(resp_data)
                    if not _qok:
                        provider.total_errors += 1
                        latency_ms = (time.time() - t_attempt) * 1000
                        mark_model_error(provider.name, upstream_model, f"quality:{_qreason}")
                        update_model_stats(provider.name, upstream_model, "quality_rejected", latency_ms)
                        log_history({"provider": provider.name, "model": upstream_model, "req_model": model,
                                     "status": "error", "code": resp.status_code, "ms": latency_ms,
                                     "err": f"quality:{_qreason}", "proxy": provider.proxy or "direct",
                                     "stream": False, "failover": True, "retry": retry})
                        errors.append(f"{provider.name}: quality={_qreason} (failover)")
                        break  # next provider candidate

                    # Extract token usage
                    prompt_tokens = 0
                    completion_tokens = 0
                    if isinstance(resp_data, dict):
                        usage = resp_data.get("usage")
                        if isinstance(usage, dict):
                            prompt_tokens = usage.get("prompt_tokens", 0) or 0
                            completion_tokens = usage.get("completion_tokens", 0) or 0
                    # Local fallback if upstream omitted usage
                    if not prompt_tokens:
                        prompt_tokens = count_prompt_tokens(body)
                    if not completion_tokens and isinstance(resp_data, dict):
                        try:
                            _ch = resp_data.get("choices") or []
                            _txt = ""
                            if _ch:
                                _msg = _ch[0].get("message") or {}
                                _txt = _msg.get("content") or _msg.get("reasoning_content") or ""
                            if _txt:
                                completion_tokens = count_tokens(_txt)
                        except Exception:
                            pass

                    is_reasoning = detect_and_fix_reasoning(resp_data)
                    update_model_stats(provider.name, upstream_model, "ok", latency_ms, is_reasoning)
                    est_cost = estimate_request_cost(provider, upstream_model, prompt_tokens, completion_tokens)
                    log_history({
                        "provider": provider.name, "model": upstream_model, "req_model": model,
                        "status": "ok", "code": resp.status_code, "ms": latency_ms,
                        "proxy": provider.proxy or "direct", "stream": False,
                        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "est_cost_usd": round(est_cost, 8)
                    })
                    return JSONResponse(resp_data, status_code=resp.status_code)

                # Streaming with pre-first-token failover: PRIME the first upstream
                # chunk here (still inside the candidate loop) so a connection that
                # drops, times out, or closes empty BEFORE emitting any token can
                # cascade to the next healthy provider — the client hasn't received
                # a byte yet, so the switch is invisible. Once the first byte is in
                # hand we commit the StreamingResponse and can no longer fail over.
                _prov_name = provider.name
                _proxy = provider.proxy or "direct"
                _provider_ref = provider
                _stream_iter = resp.aiter_bytes()
                _first_chunk = b""
                _t_first = None
                _t_gen_start = time.time()
                prime_timeout = ROUTER_CFG.get("stream_prime_timeout", 20)
                try:
                    while True:
                        _c = await asyncio.wait_for(_stream_iter.__anext__(), timeout=prime_timeout)
                        if _c:
                            _first_chunk = _c
                            _t_first = time.time()
                            break
                except StopAsyncIteration:
                    # upstream closed with zero bytes before any token → failover
                    try: await resp.aclose()
                    except Exception: pass
                    provider.total_errors += 1
                    latency_ms = (time.time() - t_attempt) * 1000
                    mark_model_error(provider.name, upstream_model, "empty stream")
                    update_model_stats(provider.name, upstream_model, "error", latency_ms)
                    log_history({"provider": provider.name, "model": upstream_model, "req_model": model,
                                 "status": "error", "code": 0, "ms": latency_ms,
                                 "err": "empty stream before first token", "proxy": provider.proxy or "direct",
                                 "stream": True, "failover": True})
                    errors.append(f"{provider.name}: empty stream (failover)")
                    break  # next provider candidate
                except (asyncio.TimeoutError, httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout):
                    try: await resp.aclose()
                    except Exception: pass
                    provider.total_errors += 1
                    latency_ms = (time.time() - t_attempt) * 1000
                    mark_model_error(provider.name, upstream_model, "ttft timeout")
                    update_model_stats(provider.name, upstream_model, "timeout", latency_ms)
                    log_history({"provider": provider.name, "model": upstream_model, "req_model": model,
                                 "status": "error", "code": 0, "ms": latency_ms,
                                 "err": f"TTFT timeout >{prime_timeout}s", "proxy": provider.proxy or "direct",
                                 "stream": True, "failover": True})
                    provider.weight = max(1, provider.weight - 1)
                    errors.append(f"{provider.name}: TTFT timeout (failover)")
                    break  # next provider candidate
                except Exception as e:
                    try: await resp.aclose()
                    except Exception: pass
                    provider.total_errors += 1
                    latency_ms = (time.time() - t_attempt) * 1000
                    mark_model_error(provider.name, upstream_model, f"stream-prime:{str(e)[:50]}")
                    update_model_stats(provider.name, upstream_model, "exception", latency_ms)
                    log_history({"provider": provider.name, "model": upstream_model, "req_model": model,
                                 "status": "error", "code": 0, "ms": latency_ms,
                                 "err": f"stream prime: {str(e)[:110]}", "proxy": provider.proxy or "direct",
                                 "stream": True, "failover": True})
                    errors.append(f"{provider.name}: stream prime error (failover)")
                    break  # next provider candidate

                # Streaming quality gate (upgrade #2): peek the primed first chunk.
                # If it already contains a refusal / repetition / garbage pattern,
                # abort before committing the StreamingResponse and fail over.
                _peek_text = ""
                try:
                    for _pl in _first_chunk.decode("utf-8", "ignore").split("\n"):
                        if _pl.startswith("data:"):
                            _payload = _pl[5:].strip()
                            if _payload and _payload != "[DONE]":
                                _obj = json.loads(_payload)
                                _peek_text += _extract_resp_text(_obj, is_chunk=True)
                except Exception:
                    pass
                if _peek_text:
                    _pok, _preason = quality_check(
                        {"choices": [{"delta": {"content": _peek_text}}]}, is_chunk=True)
                    if not _pok:
                        try: await resp.aclose()
                        except Exception: pass
                        provider.total_errors += 1
                        latency_ms = (time.time() - t_attempt) * 1000
                        mark_model_error(provider.name, upstream_model, f"quality:{_preason}")
                        update_model_stats(provider.name, upstream_model, "quality_rejected", latency_ms)
                        log_history({"provider": provider.name, "model": upstream_model, "req_model": model,
                                     "status": "error", "code": 0, "ms": latency_ms,
                                     "err": f"quality:{_preason}", "proxy": provider.proxy or "direct",
                                     "stream": True, "failover": True})
                        errors.append(f"{provider.name}: quality={_preason} (stream failover)")
                        break  # next provider candidate (inner retry loop)

                # First byte in hand → commit. Real TTFT-based success stat.
                DEAD_MODELS.pop(f"{provider.name}/{upstream_model}", None)
                _ttft_prime_ms = round((_t_first - _t_gen_start) * 1000, 1) if _t_first else latency_ms
                update_model_stats(provider.name, upstream_model, "ok", _ttft_prime_ms, False)

                async def gen():
                    buf = b""
                    p_tok = 0
                    c_tok = 0
                    _completion_text = []   # local fallback when upstream omits usage
                    _t_gen_start_local = _t_gen_start
                    _t_first_local = _t_first
                    async def _chained():
                        # replay the primed first chunk, then drain the rest
                        if _first_chunk:
                            yield _first_chunk
                        async for c in _stream_iter:
                            yield c
                    try:
                        async for chunk in _chained():
                            yield chunk
                            # accumulate tail to sniff usage from SSE data lines
                            try:
                                buf = (buf + chunk)[-8192:]
                                for line in buf.split(b"\n"):
                                    line = line.strip()
                                    if not line.startswith(b"data:"):
                                        continue
                                    payload = line[5:].strip()
                                    if payload in (b"[DONE]", b""):
                                        continue
                                    try:
                                        obj = json.loads(payload)
                                    except Exception:
                                        continue
                                    u = obj.get("usage") if isinstance(obj, dict) else None
                                    if isinstance(u, dict):
                                        p_tok = u.get("prompt_tokens", p_tok) or p_tok
                                        c_tok = u.get("completion_tokens", c_tok) or c_tok
                                    if isinstance(obj, dict):
                                        dt = _extract_delta_text(obj)
                                        if dt:
                                            _completion_text.append(dt)
                            except Exception:
                                pass
                    finally:
                        # Runs even if the client disconnects mid-stream → always logged.
                        if not p_tok:
                            p_tok = count_prompt_tokens(body)
                        if not c_tok and _completion_text:
                            c_tok = count_tokens("".join(_completion_text))
                        est = estimate_request_cost(_provider_ref, upstream_model, p_tok, c_tok)
                        _now_end = time.time()
                        ttft_ms = round((_t_first_local - _t_gen_start_local) * 1000, 1) if _t_first_local else 0.0
                        gen_secs = max(1e-6, _now_end - (_t_first_local or _t_gen_start_local))
                        tok_per_s = round(c_tok / gen_secs, 2) if c_tok else 0.0
                        record_throughput(_prov_name, upstream_model, ttft_ms, tok_per_s, c_tok)
                        log_history({
                            "provider": _prov_name, "model": upstream_model, "req_model": model,
                            "status": "ok", "code": resp.status_code, "ms": latency_ms,
                            "proxy": _proxy, "stream": True,
                            "prompt_tokens": p_tok, "completion_tokens": c_tok,
                            "total_tokens": p_tok + c_tok,
                            "est_cost_usd": round(est, 8),
                            "ttft_ms": ttft_ms, "tok_s": tok_per_s
                        })
                return StreamingResponse(gen(), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

            except (asyncio.TimeoutError, httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout):
                provider.last_error = "timeout"
                provider.total_errors += 1
                latency_ms = (time.time() - t_attempt) * 1000
                mark_model_error(provider.name, upstream_model, "timeout")
                update_model_stats(provider.name, upstream_model, "timeout", latency_ms)
                log_history({"provider": provider.name, "model": upstream_model, "req_model": model,
                             "status": "error", "code": 0, "ms": latency_ms, "err": "timeout",
                             "proxy": provider.proxy or "direct", "retry": retry})
                # Timeout → deprioritize (increase weight penalty) + try next provider
                provider.weight = max(1, provider.weight - 1)
                errors.append(f"{provider.name}: timeout (deprioritized)")
                break  # move to next provider
            except Exception as e:
                provider.last_error = str(e)[:200]
                provider.total_errors += 1
                latency_ms = (time.time() - t_attempt) * 1000
                mark_model_error(provider.name, upstream_model, f"exc:{str(e)[:60]}")
                update_model_stats(provider.name, upstream_model, "exception", latency_ms)
                log_history({"provider": provider.name, "model": upstream_model, "req_model": model,
                             "status": "error", "code": 0, "ms": latency_ms, "err": str(e)[:120],
                             "proxy": provider.proxy or "direct", "retry": retry})
                errors.append(f"{provider.name}: {str(e)[:200]}")
                break  # move to next provider

    log_history({"provider": "none", "model": model, "req_model": model,
                 "status": "all_failed", "code": 502, "ms": (time.time()-t_start)*1000,
                 "err": errors[0][:120] if errors else "no candidates", "proxy": ""})
    raise HTTPException(502, json.dumps({"error": {"message": "All providers failed", "type": "router_error",
                                                    "attempts": errors}}, indent=2))


@app.get("/v1/health")
async def health():
    return {"status": "ok", "providers": len(PROVIDERS), "combos": len(COMBOS),
            "proxies": len(PROXIES), "ts": time.time()}


@app.get("/v1/model-stats")
async def model_stats(authorization: Optional[str] = Header(None)):
    """Per-model latency, success rate, reasoning flag."""
    check_gateway_key(authorization)
    stats = []
    for key, s in MODEL_STATS.items():
        avg = round(s["latency_sum"] / s["total"], 1) if s["total"] else 0
        success_rate = round(s["ok"] / s["total"] * 100, 1) if s["total"] else 0
        stats.append({
            "model": key, "total": s["total"], "ok": s["ok"], "err": s["err"],
            "avg_ms": avg, "min_ms": s["latency_min"] if s["latency_min"] < 999999 else 0,
            "max_ms": s["latency_max"], "success_rate": success_rate,
            "reasoning": s["reasoning"], "last_error": s["last_error"],
            "last_used": s["last_used"],
            # Phase 4 — EMA (reaktif ke kondisi terkini)
            "ema_latency_ms": round(s.get("ema_latency_ms", avg), 1),
            "ema_success": round(s.get("ema_success", success_rate / 100.0), 3),
            "samples": s.get("samples", 0),
        })
    stats.sort(key=lambda x: x["total"], reverse=True)
    return {"stats": stats, "total_models": len(stats),
            "reasoning_models": list(REASONING_MODELS)}


@app.get("/v1/key-stats")
async def key_stats(authorization: Optional[str] = Header(None)):
    """Phase E — per-key EMA latency/success for weighted routing verification."""
    check_gateway_key(authorization)
    ks = []
    for kk, s in KEY_STATS.items():
        if s.get("samples", 0) == 0:
            continue
        # mask key: first 4 + last 4
        parts = kk.split("|", 1)
        provider = parts[0] if len(parts) > 0 else ""
        raw_key = parts[1] if len(parts) > 1 else ""
        masked = raw_key[:4] + "..." + raw_key[-4:] if len(raw_key) > 8 else raw_key[:4] + "..."
        ks.append({
            "provider": provider,
            "key": masked,
            "ema_latency_ms": round(s["ema_ms"], 1),
            "ema_success": round(s["ema_ok"], 3),
            "samples": s["samples"],
            "last_latency_ms": round(s["last_latency_ms"], 1),
        })
    ks.sort(key=lambda x: x["ema_latency_ms"])
    return {"key_stats": ks, "total_keys": len(ks)}


@app.get("/v1/cost-stats")
async def cost_stats(authorization: Optional[str] = Header(None)):
    """Phase 9 — Cost-aware routing stats: estimated spend per model/provider/combo."""
    check_gateway_key(authorization)
    # Aggregate from history (last 5000 entries = HISTORY_MAX)
    by_model = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "est_cost_usd": 0.0})
    by_provider = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "est_cost_usd": 0.0})
    by_combo = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "est_cost_usd": 0.0})
    total_cost = 0.0
    total_requests = 0
    for entry in HISTORY:
        if entry.get("est_cost_usd", 0) == 0 and entry.get("stream", False):
            # Streaming: cost not tracked in history yet (no token count from SSE)
            continue
        model = entry.get("model", "unknown")
        provider_name = entry.get("provider", "unknown")
        req_model = entry.get("req_model", model)
        prompt_t = entry.get("prompt_tokens", 0)
        completion_t = entry.get("completion_tokens", 0)
        cost = entry.get("est_cost_usd", 0.0)
        by_model[model]["requests"] += 1
        by_model[model]["prompt_tokens"] += prompt_t
        by_model[model]["completion_tokens"] += completion_t
        by_model[model]["est_cost_usd"] += cost
        by_provider[provider_name]["requests"] += 1
        by_provider[provider_name]["prompt_tokens"] += prompt_t
        by_provider[provider_name]["completion_tokens"] += completion_t
        by_provider[provider_name]["est_cost_usd"] += cost
        by_combo[req_model]["requests"] += 1
        by_combo[req_model]["prompt_tokens"] += prompt_t
        by_combo[req_model]["completion_tokens"] += completion_t
        by_combo[req_model]["est_cost_usd"] += cost
        total_cost += cost
        total_requests += 1
    def fmt(d):
        return {k: {**v, "est_cost_usd": round(v["est_cost_usd"], 6)} for k, v in sorted(d.items(), key=lambda x: -x[1]["est_cost_usd"])}
    return {
        "total_est_cost_usd": round(total_cost, 6),
        "total_requests": total_requests,
        "by_model": fmt(by_model),
        "by_provider": fmt(by_provider),
        "by_combo": fmt(by_combo),
        "pricing_coverage": {
            "known_models": sum(1 for m in by_model if any(get_model_cost(p, m) != (0.0, 0.0) for p in PROVIDERS.values())),
            "unknown_models": sum(1 for m in by_model if all(get_model_cost(p, m) == (0.0, 0.0) for p in PROVIDERS.values()))
        }
    }


@app.post("/v1/health-check")
async def trigger_health_check(authorization: Optional[str] = Header(None)):
    """Manually trigger provider health check."""
    check_gateway_key(authorization)
    tasks = [_ping_provider(p) for p in PROVIDERS.values() if p.keys]
    await asyncio.gather(*tasks, return_exceptions=True)
    results = [{"provider": p.name, "healthy": p.failures < 3 and p.locked_until <= time.time(),
                "failures": p.failures, "last_error": p.last_error,
                "models": len(p.models)} for p in PROVIDERS.values()]
    return {"ok": True, "checked": len(results), "results": results, "ts": time.time()}


@app.get("/v1/dead-models")
async def dead_models_api(authorization: Optional[str] = Header(None)):
    """List models marked as dead (3x probe failure)."""
    check_gateway_key(authorization)
    dead = []
    for model_id, info in DEAD_MODELS.items():
        if info.get("disabled_at", 0) > 0:
            dead.append({"model": model_id, "failures": info["failures"],
                         "last_error": info["last_error"],
                         "disabled_at": info["disabled_at"],
                         "last_checked": info.get("last_checked", 0)})
    return {"dead_models": dead, "total": len(dead)}


# ---------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if os.path.exists(DASHBOARD_PATH):
        # no-store: dashboard HTML changes often; never let the browser serve
        # a stale cached copy (root cause of "fitur belum muncul" after updates)
        return FileResponse(DASHBOARD_PATH, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        })
    return HTMLResponse("<h1>dashboard.html missing</h1>")


@app.get("/landing", response_class=HTMLResponse)
async def landing(request: Request):
    if os.path.exists(LANDING_PATH):
        return FileResponse(LANDING_PATH, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        })
    return HTMLResponse("<h1>landing.html missing</h1>")


@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):
    if os.path.exists(DOCS_PATH):
        return FileResponse(DOCS_PATH, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        })
    return HTMLResponse("<h1>docs.html missing</h1>")


@app.get("/auth", response_class=HTMLResponse)
async def auth(request: Request):
    if os.path.exists(AUTH_PATH):
        return FileResponse(AUTH_PATH, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        })
    return HTMLResponse("<h1>auth.html missing</h1>")


@app.post("/api/login")
async def login(request: Request, response: Response):
    ip = get_client_ip(request)
    check_rate_limit(ip)
    body = await request.json()
    if body.get("password") == DASH_PASS:
        token = secrets.token_urlsafe(32)
        VALID_SESSIONS[token] = time.time()
        _save_sessions()
        clear_attempts(ip)
        response.set_cookie("vr_token", token, httponly=True, max_age=86400 * 7, samesite="lax")
        return {"ok": True}
    record_failed_attempt(ip)
    raise HTTPException(401, "Invalid credentials")


@app.post("/api/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("vr_token")
    if token:
        VALID_SESSIONS.pop(token, None)
        _save_sessions()
    response.delete_cookie("vr_token")
    return {"ok": True}


@app.get("/api/status")
async def api_status(request: Request):
    check_dashboard_auth(request)
    # Model stats summary
    total_model_requests = sum(s["total"] for s in MODEL_STATS.values())
    reasoning_count = len(REASONING_MODELS)
    healthy_count = sum(1 for p in PROVIDERS.values() if p.failures < 3 and p.locked_until <= time.time())
    unhealthy_count = len(PROVIDERS) - healthy_count

    # --- Aggregate token/cost totals (Point 1) ---
    # Full window (all retained history) + today (local midnight)
    import time as _t
    _now = _t.time()
    _lt = _t.localtime(_now)
    _midnight = _now - (_lt.tm_hour * 3600 + _lt.tm_min * 60 + _lt.tm_sec)
    agg_in = agg_out = 0
    agg_cost = 0.0
    agg_ok = 0
    today_in = today_out = 0
    today_cost = 0.0
    today_ok = 0
    for e in HISTORY:
        pt = e.get("prompt_tokens", 0) or 0
        ct = e.get("completion_tokens", 0) or 0
        c = e.get("est_cost_usd", 0.0) or 0.0
        is_ok = e.get("status") in ("ok", "hedge_ok")
        agg_in += pt; agg_out += ct; agg_cost += c
        if is_ok: agg_ok += 1
        if e.get("ts", 0) >= _midnight:
            today_in += pt; today_out += ct; today_cost += c
            if is_ok: today_ok += 1

    # --- Per-provider latency series for sparkline (Point 3) ---
    # last ~40 successful latencies per provider, oldest→newest
    _lat_series = defaultdict(list)
    for e in HISTORY:
        if e.get("status") in ("ok", "hedge_ok"):
            ms = e.get("ms")
            if isinstance(ms, (int, float)) and ms > 0:
                _lat_series[e.get("provider", "?")].append(int(ms))
    latency_series = {k: v[-40:] for k, v in _lat_series.items()}

    return {
        "ok": True,
        "ts": time.time(),
        "version": "2.3.0",
        "total_providers": len(PROVIDERS),
        "total_keys": sum(len(p.keys) for p in PROVIDERS.values()),
        "total_requests": sum(p.total_requests for p in PROVIDERS.values()),
        "total_errors": sum(p.total_errors for p in PROVIDERS.values()),
        "healthy_providers": healthy_count,
        "unhealthy_providers": unhealthy_count,
        "last_health_check": LAST_HEALTH_CHECK,
        "model_stats_count": len(MODEL_STATS),
        "total_model_requests": total_model_requests,
        "reasoning_models_count": reasoning_count,
        "dead_models_count": sum(1 for d in DEAD_MODELS.values() if d.get("disabled_at", 0) > 0),
        # aggregate token/cost (Point 1)
        "agg": {
            "total": {"in": agg_in, "out": agg_out, "cost": round(agg_cost, 6), "ok": agg_ok},
            "today": {"in": today_in, "out": today_out, "cost": round(today_cost, 6), "ok": today_ok},
        },
        "latency_series": latency_series,  # per-provider sparkline data (Point 3)
        "providers": [p.to_dict(include_key=True) for p in PROVIDERS.values()],
        "proxies": [{"name": k, "url": v, "masked": v[:12] + "..." if len(v) > 15 else v} for k, v in PROXIES.items()],
        "combos": [{"name": k, "routes": v["routes"], "strategy": v.get("strategy", "random"), "route_count": len(v["routes"])} for k, v in COMBOS.items()],
        "history": list(HISTORY)[-200:],
    }


def _canonical_model_stats_key(key: str):
    """Map stale provider prefixes to the active provider they belong to.

    MODEL_STATS is keyed 'provider/model'. Old provider names (e.g.
    'openai-compatible-jerouter', 'openai-compatible-chat-1bb0...') survive in
    router_state.json even after the provider was deleted/renamed. We remap
    them by prefix so stats show under the active provider name (VRouter).
    Returns None if the key's provider is not active anymore.
    """
    if "/" not in key:
        return None
    prov_part, _, model_part = key.partition("/")
    # Direct active provider match
    if prov_part in PROVIDERS:
        return key
    # Prefix alias: match active provider whose prefix == old provider name
    # (e.g. prefix JEROUTER -> provider VRouter) or whose name is the prefix
    # source (openai-compatible-jerouter -> VRouter).
    for name, p in PROVIDERS.items():
        if p.prefix and p.prefix.lower() == prov_part.lower():
            return f"{name}/{model_part}"
        # openai-compatible-<prefix> was the auto-generated name for a
        # provider whose prefix is <prefix> (e.g. openai-compatible-jerouter).
        if prov_part.lower().startswith("openai-compatible-"):
            stripped = prov_part[len("openai-compatible-"):].lower()
            if stripped and (stripped == p.prefix.lower() or stripped == name.lower()):
                return f"{name}/{model_part}"
    return None


@app.get("/api/model-stats")
async def api_model_stats(request: Request):
    check_dashboard_auth(request)
    # Active model universe: provider effective/manual models + combo routes.
    # Anything outside it is a stale model from a deleted config.
    active_models: set[str] = set()
    for p in PROVIDERS.values():
        active_models.update(p.effective_models)
        active_models.update(p.manual_models)
    for combo in COMBOS.values():
        for r in combo.get("routes", []):
            if r.get("model"):
                active_models.add(r["model"])
    merged: dict[str, dict] = {}
    for key, s in MODEL_STATS.items():
        canon = _canonical_model_stats_key(key)
        if not canon:
            continue  # stale provider, not active anymore
        # drop models no longer in the active universe (deleted from config)
        if "/" in canon:
            _, _, model_part = canon.partition("/")
            if model_part not in active_models:
                continue
        # merge under canonical key
        if canon not in merged:
            merged[canon] = {
                "total": 0, "ok": 0, "err": 0,
                "latency_sum": 0.0, "latency_min": 999999, "latency_max": 0,
                "reasoning": False, "last_error": None, "last_used": 0.0,
                "ema_latency_ms": 0.0, "ema_success": 0.5, "samples": 0,
            }
        m = merged[canon]
        m["total"] += s["total"]
        m["ok"] += s["ok"]
        m["err"] += s["err"]
        m["latency_sum"] += s["latency_sum"]
        m["latency_min"] = min(m["latency_min"], s["latency_min"])
        m["latency_max"] = max(m["latency_max"], s["latency_max"])
        m["reasoning"] = m["reasoning"] or s["reasoning"]
        m["last_used"] = max(m["last_used"], s.get("last_used", 0))
        if s.get("last_error"):
            m["last_error"] = s["last_error"]
        m["ema_latency_ms"] = m["ema_latency_ms"] + s.get("ema_latency_ms", 0)
        m["ema_success"] = m["ema_success"] + s.get("ema_success", 0)
        m["samples"] += s.get("samples", 0)
    stats = []
    for key, m in merged.items():
        avg = round(m["latency_sum"] / m["total"], 1) if m["total"] else 0
        success_rate = round(m["ok"] / m["total"] * 100, 1) if m["total"] else 0
        n = max(1, len([k for k in MODEL_STATS if _canonical_model_stats_key(k) == key]))
        stats.append({
            "model": key, "total": m["total"], "ok": m["ok"], "err": m["err"],
            "avg_ms": avg, "min_ms": m["latency_min"] if m["latency_min"] < 999999 else 0,
            "max_ms": m["latency_max"], "success_rate": success_rate,
            "reasoning": m["reasoning"], "last_error": m["last_error"],
            "last_used": m["last_used"],
            "ema_latency_ms": round(m["ema_latency_ms"] / n, 1),
            "ema_success": round(m["ema_success"] / n, 3),
            "samples": m["samples"],
        })
    stats.sort(key=lambda x: x["total"], reverse=True)
    return {"stats": stats, "total_models": len(stats),
            "reasoning_models": list(REASONING_MODELS)}


@app.get("/api/dead-models")
async def api_dead_models(request: Request):
    check_dashboard_auth(request)
    dead = []
    for model_id, info in DEAD_MODELS.items():
        dead.append({"model": model_id, "failures": info["failures"],
                     "last_error": info["last_error"],
                     "disabled_at": info.get("disabled_at", 0),
                     "last_checked": info.get("last_checked", 0)})
    dead.sort(key=lambda x: x.get("disabled_at", 0), reverse=True)
    return {"dead_models": dead, "total": len(dead)}


@app.post("/api/smart-combo")
async def api_smart_combo(request: Request):
    """Auto-build combo based on criteria.
    Body: {name, criteria: fast|balanced|coding|reliable, max_models, exclude_reasoning}
    Uses model stats (latency, success rate) to pick best models."""
    check_dashboard_auth(request)
    body = await request.json()
    name = body.get("name", "").strip()
    criteria = body.get("criteria", "balanced")
    max_models = int(body.get("max_models", 5))
    exclude_reasoning = body.get("exclude_reasoning", True)

    if not name:
        raise HTTPException(400, "Combo name required")
    if criteria not in ("fast", "balanced", "coding", "reliable"):
        raise HTTPException(400, "Invalid criteria. Use: fast, balanced, coding, reliable")
    if not COMBO_NAME_RE.match(name):
        raise HTTPException(400, "Invalid combo name")

    # Collect all candidate models from all providers
    candidates = []
    for p in PROVIDERS.values():
        if not p.keys or p.locked_until > time.time():
            continue
        for model in p.models:
            if not model:
                continue
            model_id = f"{p.name}/{model}"
            # Skip dead models
            dm = DEAD_MODELS.get(model_id)
            if dm and dm.get("disabled_at", 0) > 0:
                continue
            # Skip reasoning models if excluded
            if exclude_reasoning and model in REASONING_MODELS:
                continue

            # Get stats
            s = MODEL_STATS.get(model_id, {})
            avg_ms = s.get("latency_sum", 0) / s.get("total", 1) if s.get("total") else 999999
            success_rate = s.get("ok", 0) / s.get("total", 1) * 100 if s.get("total") else 0
            total_reqs = s.get("total", 0)

            # Score based on criteria
            if criteria == "fast":
                # Prioritize latency, then success rate
                score = -avg_ms + (success_rate * 10)
            elif criteria == "reliable":
                # Prioritize success rate, then latency
                score = success_rate * 100 - avg_ms / 100
            elif criteria == "coding":
                # Filter for coding-related models
                coding_keywords = ["code", "coder", "codestral", "devstral", "deepseek", "kimi", "gpt"]
                if any(kw in model.lower() for kw in coding_keywords):
                    score = success_rate * 50 - avg_ms / 10
                else:
                    score = -999999  # exclude non-coding
            else:  # balanced
                score = (success_rate * 50) - (avg_ms / 50) + min(total_reqs, 10)

            candidates.append({
                "provider": p.name,
                "model": model,
                "score": score,
                "avg_ms": round(avg_ms, 1) if avg_ms < 999999 else 0,
                "success_rate": round(success_rate, 1),
                "total_reqs": total_reqs,
            })

    # Sort by score descending, take top N
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = candidates[:max_models]

    if not selected:
        raise HTTPException(400, "No suitable models found. Try different criteria or send requests first to build stats.")

    # Build combo routes
    routes = [{"provider": c["provider"], "model": c["model"], "weight": max(1, int(c["score"] / 10))}
              for c in selected]
    COMBOS[name] = {"routes": routes, "strategy": "fallback", "rr_idx": 0}
    _save_config()

    return {"ok": True, "combo": {"name": name, "routes": routes, "strategy": "fallback"},
            "criteria": criteria, "candidates_evaluated": len(candidates),
            "selected": [{"model": c["model"], "provider": c["provider"],
                          "avg_ms": c["avg_ms"], "success_rate": c["success_rate"],
                          "score": round(c["score"], 1)} for c in selected]}


@app.post("/api/health-check")
async def api_health_check(request: Request):
    check_dashboard_auth(request)
    global LAST_HEALTH_CHECK
    tasks = [_ping_provider(p) for p in PROVIDERS.values() if p.keys]
    await asyncio.gather(*tasks, return_exceptions=True)
    LAST_HEALTH_CHECK = time.time()
    results = [{"provider": p.name, "healthy": p.failures < 3 and p.locked_until <= time.time(),
                "failures": p.failures, "last_error": p.last_error,
                "models": len(p.models)} for p in PROVIDERS.values()]
    return {"ok": True, "checked": len(results), "results": results, "ts": LAST_HEALTH_CHECK}


@app.get("/api/provider-health")
async def api_provider_health(request: Request):
    """Provider health dashboard — per-provider stats for UI."""
    check_dashboard_or_gateway(request)
    now = time.time()
    providers_data = []
    for p in PROVIDERS.values():
        # Provider health status
        healthy = p.failures < 3 and p.locked_until <= now
        degraded = p.failures > 0 and p.failures < 3
        status = "healthy" if healthy else ("degraded" if degraded else "down")

        # Per-provider stats
        avg_latency = 0
        total_reqs = p.total_requests
        total_errs = p.total_errors
        if total_reqs > 0:
            # Try to compute from MODEL_STATS for this provider's models
            provider_models = [k for k in MODEL_STATS.keys() if k.startswith(p.name + "/")]
            lat_sum = sum(MODEL_STATS[m].get("latency_sum", 0) for m in provider_models)
            req_sum = sum(MODEL_STATS[m].get("total", 0) for m in provider_models)
            if req_sum > 0:
                avg_latency = round(lat_sum / req_sum, 1)

        # Keys status
        active_keys = 0
        cooldown_keys = 0
        for key in p.keys:
            kk = f"{p.name}|{key}"
            if KEY_COOLDOWN.get(kk, 0) > now:
                cooldown_keys += 1
            else:
                active_keys += 1

        # Last health check
        last_check_ago = int(now - LAST_HEALTH_CHECK) if LAST_HEALTH_CHECK > 0 else None

        providers_data.append({
            "name": p.name,
            "status": status,
            "healthy": healthy,
            "failures": p.failures,
            "last_error": p.last_error,
            "models_count": len(p.effective_models),
            "total_requests": total_reqs,
            "total_errors": total_errs,
            "avg_latency_ms": avg_latency,
            "error_rate": round(total_errs / total_reqs * 100, 1) if total_reqs > 0 else 0,
            "keys_total": len(p.keys),
            "keys_active": active_keys,
            "keys_cooldown": cooldown_keys,
            "keys": p.keys,
            "manual_models": p.manual_models,
            "default_model": p.default_model,
            "weight": p.weight,
            "is_active": p.is_active,
            "locked_until": p.locked_until,
            "last_used": p.last_used,
            "proxy": p.proxy or "direct",
            "base_url": p.base_url,
            "prefix": p.prefix,
            "type": p.type,
            "models": p.effective_models,
            "last_check_ago": last_check_ago,
        })

    # Summary
    healthy_count = sum(1 for p in providers_data if p["healthy"])
    degraded_count = sum(1 for p in providers_data if p["status"] == "degraded")
    down_count = sum(1 for p in providers_data if p["status"] == "down")
    total_requests = sum(p["total_requests"] for p in providers_data)
    total_errors = sum(p["total_errors"] for p in providers_data)

    return {
        "ok": True,
        "ts": now,
        "summary": {
            "total_providers": len(providers_data),
            "healthy": healthy_count,
            "degraded": degraded_count,
            "down": down_count,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "overall_error_rate": round(total_errors / total_requests * 100, 1) if total_requests > 0 else 0,
            "last_health_check": LAST_HEALTH_CHECK,
            "last_check_ago": last_check_ago,
        },
        "providers": providers_data,
    }


# --- providers ---
@app.post("/api/providers")
async def add_provider(request: Request):
    check_dashboard_auth(request)
    body = await request.json()
    name = body.get("name", "").strip()
    if not name or name in PROVIDERS:
        raise HTTPException(400, "Provider name missing or already exists")
    PROVIDERS[name] = Provider(
        name=name,
        base_url=body["base_url"],
        prefix=body.get("prefix", ""),
        type_=body.get("type", "apikey"),
        keys=body.get("keys", []),
        weight=body.get("weight", 5),
        default_model=body.get("default_model", ""),
        proxy=body.get("proxy", ""),
    )
    _save_config()
    return {"ok": True, "provider": PROVIDERS[name].to_dict()}


@app.post("/api/providers/{name}/keys")
async def add_key(name: str, request: Request):
    check_dashboard_auth(request)
    p = PROVIDERS.get(name)
    if not p:
        raise HTTPException(404, "Provider not found")
    body = await request.json()
    key = body.get("key", "").strip()
    if not key:
        raise HTTPException(400, "Empty key")
    p.keys.append(key)
    _save_config()
    return {"ok": True, "key_count": len(p.keys)}


@app.delete("/api/providers/{name}/keys/{idx}")
async def delete_key(name: str, idx: int, request: Request):
    check_dashboard_auth(request)
    p = PROVIDERS.get(name)
    if not p:
        raise HTTPException(404, "Provider not found")
    if 0 <= idx < len(p.keys):
        p.keys.pop(idx)
        _save_config()
    return {"ok": True, "key_count": len(p.keys)}


@app.delete("/api/providers/{name}")
async def delete_provider(name: str, request: Request):
    check_dashboard_auth(request)
    if name not in PROVIDERS:
        raise HTTPException(404, "Provider not found")
    # Cascade: hapus provider dari semua combo dulu, baru hapus provider
    removed_from = [c for c, v in COMBOS.items() if any(r.get("provider") == name for r in v.get("routes", []))]
    for c in removed_from:
        COMBOS[c]["routes"] = [r for r in COMBOS[c]["routes"] if r.get("provider") != name]
    # Combo yang routes-nya habis → hapus combo itu
    for c in [c for c in COMBOS if not COMBOS[c]["routes"]]:
        del COMBOS[c]
    # Hapus models cache provider ini supaya tidak re-appear di restart/dashboard
    p = PROVIDERS[name]
    p.models = []
    p.models_fetched_at = 0.0
    _save_models_cache()
    del PROVIDERS[name]
    DELETED_PROVIDERS.add(name)
    _save_config()
    return {"ok": True, "removed_from": removed_from}


@app.post("/api/providers/{name}/toggle")
async def toggle_provider(name: str, request: Request):
    """Toggle provider on/off (active/inactive)."""
    check_dashboard_auth(request)
    p = PROVIDERS.get(name)
    if not p:
        raise HTTPException(404, "Provider not found")
    body = await request.json()
    p.is_active = body.get("is_active", not p.is_active)
    _save_config()
    return {"ok": True, "is_active": p.is_active}


@app.put("/api/providers/{name}/proxy")
async def set_provider_proxy(name: str, request: Request):
    """Set proxy untuk satu provider (9router-style per-connection).

    value:
      ""                       -> direct (tanpa proxy)
      "<pool name>"            -> pakai pool dari registry PROXIES
      "http://user:pass@ip:port" -> full URL langsung (bypass registry)
    """
    check_dashboard_auth(request)
    p = PROVIDERS.get(name)
    if not p:
        raise HTTPException(404, "Provider not found")
    body = await request.json()
    val = (body.get("proxy") or "").strip()
    if val and "://" not in val and val not in PROXIES:
        raise HTTPException(400, f"Unknown proxy pool '{val}' — tambahkan dulu ke /api/proxies atau pakai full URL")
    p.proxy = val
    _save_config()
    return {"ok": True, "proxy": p.proxy, "resolved": resolve_proxy_url(p.proxy) or "direct"}


@app.post("/api/providers/{name}/test")
async def test_provider(name: str, request: Request):
    check_dashboard_auth(request)
    p = PROVIDERS.get(name)
    if not p:
        raise HTTPException(404, "Provider not found")
    key = p.next_key()
    url = f"{p.base_url}/models"
    headers = {"User-Agent": BROWSER_UA, **p.auth_header(key)}
    proxy_url = resolve_proxy_url(p.proxy)
    try:
        if proxy_url:
            async with httpx.AsyncClient(timeout=20, proxy=proxy_url) as pc:
                resp = await pc.get(url, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=20) as ac:
                resp = await ac.get(url, headers=headers)
        if resp.status_code < 400:
            models = resp.json().get("data", []) if isinstance(resp.json(), dict) else []
            return {"ok": True, "status": resp.status_code, "models": len(models),
                    "proxy": p.proxy or "direct",
                    "sample": [m.get("id") for m in models[:5]]}
        return {"ok": False, "status": resp.status_code, "error": resp.text[:200], "proxy": p.proxy or "direct"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "proxy": p.proxy or "direct"}


@app.post("/api/providers/{name}/fetch-models")
async def fetch_provider_models(name: str, request: Request):
    """Import-models: fetch FULL live model list from upstream /models.
    Returns every model id (not just a sample) so the dashboard can present
    a checklist for import into manual_models (9router-style)."""
    check_dashboard_auth(request)
    p = PROVIDERS.get(name)
    if not p:
        raise HTTPException(404, "Provider not found")
    key = p.next_key()
    url = f"{p.base_url}/models"
    headers = {"User-Agent": BROWSER_UA, **p.auth_header(key)}
    proxy_url = resolve_proxy_url(p.proxy)
    try:
        if proxy_url:
            async with httpx.AsyncClient(timeout=25, proxy=proxy_url) as pc:
                resp = await pc.get(url, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=25) as ac:
                resp = await ac.get(url, headers=headers)
        if resp.status_code >= 400:
            return {"ok": False, "status": resp.status_code,
                    "error": resp.text[:200], "proxy": p.proxy or "direct"}
        try:
            j = resp.json()
        except Exception:
            return {"ok": False, "error": "non-JSON /models response", "proxy": p.proxy or "direct"}
        data = j.get("data", j) if isinstance(j, dict) else j
        ids = sorted({m.get("id") for m in data
                      if isinstance(m, dict) and m.get("id")}) if isinstance(data, list) else []
        # Gateway-style provider (keep_prefix=True, upstream = 9router/local gateway):
        # gateway /models returns models dari SEMUA provider — filter hanya yang
        # prefix-nya milik provider ini (mis. qd/* untuk QODER). Kalau kosong
        # (upstream filter beda format), jangan timpa cache yang sudah valid.
        if getattr(p, "keep_prefix", False) and p.prefix:
            own = [m for m in ids if m.startswith(f"{p.prefix}/")]
            if own:
                ids = own
        # refresh the auto-fetch cache too
        if ids:
            p.models = ids
            p.models_fetched_at = time.time()
            _save_models_cache()
        return {"ok": True, "status": resp.status_code, "count": len(ids),
                "models": ids, "manual": p.manual_models, "proxy": p.proxy or "direct"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "proxy": p.proxy or "direct"}


@app.post("/api/refresh-all-models")
async def refresh_all_models(request: Request):
    """Point 2 — Bulk refresh: fetch live /models for EVERY active provider in
    parallel and update each provider's auto-fetch cache. Returns a per-provider
    summary (count / ok / error) so the dashboard can show one-click results."""
    check_dashboard_auth(request)

    async def _one(p: Provider):
        if not p.keys:
            return {"provider": p.name, "ok": False, "error": "no keys", "count": 0}
        key = p.next_key()
        url = f"{p.base_url}/models"
        headers = {"User-Agent": BROWSER_UA, **p.auth_header(key)}
        proxy_url = resolve_proxy_url(p.proxy)
        try:
            if proxy_url:
                async with httpx.AsyncClient(timeout=25, proxy=proxy_url) as pc:
                    resp = await pc.get(url, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=25) as ac:
                    resp = await ac.get(url, headers=headers)
            if resp.status_code >= 400:
                return {"provider": p.name, "ok": False,
                        "error": f"HTTP {resp.status_code}", "count": len(p.models or [])}
            try:
                j = resp.json()
            except Exception:
                return {"provider": p.name, "ok": False, "error": "non-JSON", "count": len(p.models or [])}
            data = j.get("data", j) if isinstance(j, dict) else j
            ids = sorted({m.get("id") for m in data
                          if isinstance(m, dict) and m.get("id")}) if isinstance(data, list) else []
            if ids:
                p.models = ids
                p.models_fetched_at = time.time()
            return {"provider": p.name, "ok": True, "count": len(ids)}
        except Exception as e:
            return {"provider": p.name, "ok": False, "error": str(e)[:120], "count": len(p.models or [])}

    targets = [p for p in PROVIDERS.values() if p.is_active and p.keys]
    results = await asyncio.gather(*[_one(p) for p in targets], return_exceptions=False)
    # persist any cache updates once
    try:
        _save_models_cache()
    except Exception:
        pass
    ok_n = sum(1 for r in results if r.get("ok"))
    total_models = sum(r.get("count", 0) for r in results)
    results.sort(key=lambda r: (not r.get("ok"), r.get("provider", "")))
    return {"ok": True, "providers_ok": ok_n, "providers_total": len(targets),
            "total_models": total_models, "results": results}


@app.get("/api/circuit-breakers")
async def circuit_breakers(request: Request):
    """Circuit-breaker panel data: per-provider breaker state + recent trip/reset events."""
    check_dashboard_auth(request)
    now = time.time()
    rows = []
    for p in PROVIDERS.values():
        locked = p.locked_until > now
        rows.append({
            "provider": p.name,
            "state": "open" if locked else ("half_open" if p.failures > 0 else "closed"),
            "locked": locked,
            "remaining_s": int(p.locked_until - now) if locked else 0,
            "failures": p.failures,
            "threshold": CB_FAIL_THRESHOLD,
            "last_error": p.last_error,
            "total_requests": p.total_requests,
            "total_errors": p.total_errors,
            "is_active": p.is_active,
        })
    # open breakers first, then by failure count
    rows.sort(key=lambda r: (not r["locked"], -r["failures"], r["provider"]))
    events = list(CB_EVENTS)[-50:][::-1]
    return {"ok": True, "ts": now,
            "config": {"fail_threshold": CB_FAIL_THRESHOLD,
                       "lock_seconds": CB_LOCK_SECONDS,
                       "health_lock_seconds": CB_HEALTH_LOCK_SECONDS},
            "open_count": sum(1 for r in rows if r["locked"]),
            "breakers": rows, "events": events}


@app.post("/api/circuit-breakers/{name}/reset")
async def circuit_breaker_reset(name: str, request: Request):
    """Force-close a provider's breaker (manual recovery button)."""
    check_dashboard_auth(request)
    p = PROVIDERS.get(name)
    if not p:
        raise HTTPException(404, "Provider not found")
    reset_circuit(p, "manual dashboard reset")
    return {"ok": True, "provider": name, "state": "closed"}


@app.post("/api/circuit-breakers/reset-all")
async def circuit_breaker_reset_all(request: Request):
    """Force-close every open breaker at once."""
    check_dashboard_auth(request)
    n = 0
    for p in PROVIDERS.values():
        if p.locked_until > time.time() or p.failures > 0:
            reset_circuit(p, "manual reset-all")
            n += 1
    return {"ok": True, "reset": n}


@app.get("/api/throughput")
async def throughput(request: Request):
    """Tok/s meter: per provider/model streaming performance ranked by tok/s (EMA)."""
    check_dashboard_auth(request)
    now = time.time()
    rows = []
    for s in THROUGHPUT_STATS.values():
        rows.append({
            "provider": s["provider"], "model": s["model"],
            "samples": s["samples"],
            "ttft_ms": s["ttft_ema"], "ttft_last": s["ttft_last"],
            "tok_s": s["toks_ema"], "tok_s_last": s["toks_last"],
            "best_tok_s": round(s["best_toks"], 2),
            "tok_total": s["tok_total"],
            "age_s": int(now - s["last_used"]) if s["last_used"] else None,
        })
    # rank fastest first
    rows.sort(key=lambda r: (-r["tok_s"], r["ttft_ms"]))
    return {"ok": True, "ts": now, "count": len(rows), "models": rows}


@app.post("/api/throughput/reset")
async def throughput_reset(request: Request):
    """Clear the tok/s meter accumulators."""
    check_dashboard_auth(request)
    n = len(THROUGHPUT_STATS)
    THROUGHPUT_STATS.clear()
    return {"ok": True, "cleared": n}


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"


@app.get("/api/health-score")
async def health_score(request: Request):
    """Model Health Scoring — composite 0-100 per provider/model combining:
      • reliability (EMA success rate)      weight 45
      • latency    (EMA latency, lower=better) weight 25
      • throughput (tok/s from stream meter)   weight 20
      • freshness  (recently used / not stale)  weight 10
    Circuit-open providers are penalised. Ranked best-first.

    Optional ?provider=<name> filters to a single provider (e.g. ?provider=VRouter).
    By default only shows models from providers currently registered & active."""
    check_dashboard_auth(request)
    now = time.time()
    filter_prov = (request.query_params.get("provider") or "").strip()
    # provider lock state for penalty
    locked = {p.name: (p.locked_until > now) for p in PROVIDERS.values()}
    active_providers = set(PROVIDERS.keys())
    rows = []
    for key, s in MODEL_STATS.items():
        prov = key.split("/", 1)[0]
        total = s.get("total", 0)
        if not total:
            continue
        # Filter: only active registered providers, or specific provider if requested
        if filter_prov:
            if prov != filter_prov:
                continue
        else:
            if prov not in active_providers:
                continue
        ema_succ = s.get("ema_success", s.get("ok", 0) / total if total else 0.0)
        ema_lat = s.get("ema_latency_ms", (s["latency_sum"] / total) if total else 0.0)
        # reliability: 0-45
        rel = ema_succ * 45.0
        # latency: 0-25 — 300ms→full, 8000ms→0 (linear clamp)
        lat_norm = max(0.0, min(1.0, (8000.0 - ema_lat) / (8000.0 - 300.0)))
        lat_score = lat_norm * 25.0
        # throughput: 0-20 — 60 tok/s→full
        tp = THROUGHPUT_STATS.get(key)
        tok_s = tp["toks_ema"] if tp else 0.0
        tp_score = min(1.0, tok_s / 60.0) * 20.0 if tok_s else 0.0
        # freshness: 0-10 — used < 5min→full, > 6h→0
        age = now - s.get("last_used", 0)
        fresh = max(0.0, min(1.0, (21600.0 - age) / (21600.0 - 300.0)))
        fresh_score = fresh * 10.0
        # Score via shared helper (single source of truth w/ routing path)
        score = health_score_for(key, now)
        rows.append({
            "model": key, "provider": prov, "score": score, "grade": _grade(score),
            "reliability": round(ema_succ * 100, 1),
            "ema_latency_ms": round(ema_lat, 1),
            "tok_s": round(tok_s, 1),
            "total": total, "err": s.get("err", 0),
            "circuit_open": locked.get(prov, False),
            "age_s": int(age) if s.get("last_used") else None,
        })
    rows.sort(key=lambda r: -r["score"])
    return {"ok": True, "ts": now, "count": len(rows), "models": rows, "filtered_provider": filter_prov or None}


@app.get("/api/costs")
async def costs(request: Request):
    """Cost dashboard — aggregate est_cost_usd + tokens from request history.
    Query: ?window=<seconds> (default 86400 = 24h; 0 = all-time)."""
    check_dashboard_auth(request)
    try:
        window = int(request.query_params.get("window", "86400"))
    except ValueError:
        window = 86400
    now = time.time()
    cutoff = 0 if window <= 0 else now - window
    by_prov: dict = {}
    by_model: dict = {}
    tot_cost = tot_in = tot_out = tot_req = 0
    for e in HISTORY:
        if e.get("ts", 0) < cutoff:
            continue
        if e.get("status") not in ("ok", "hedge_ok"):
            continue
        prov = e.get("provider", "?")
        model = f'{prov}/{e.get("model", "?")}'
        c = float(e.get("est_cost_usd", 0) or 0)
        pin = int(e.get("prompt_tokens", 0) or 0)
        pout = int(e.get("completion_tokens", 0) or 0)
        tot_cost += c; tot_in += pin; tot_out += pout; tot_req += 1
        for agg, k in ((by_prov, prov), (by_model, model)):
            d = agg.setdefault(k, {"cost": 0.0, "in": 0, "out": 0, "req": 0})
            d["cost"] += c; d["in"] += pin; d["out"] += pout; d["req"] += 1
    def _rows(agg, label):
        out = []
        for k, d in agg.items():
            out.append({label: k, "cost_usd": round(d["cost"], 6),
                        "prompt_tokens": d["in"], "completion_tokens": d["out"],
                        "total_tokens": d["in"] + d["out"], "requests": d["req"]})
        out.sort(key=lambda r: -r["cost_usd"])
        return out
    return {"ok": True, "ts": now, "window_s": window,
            "totals": {"cost_usd": round(tot_cost, 6), "prompt_tokens": tot_in,
                       "completion_tokens": tot_out, "total_tokens": tot_in + tot_out,
                       "requests": tot_req},
            "by_provider": _rows(by_prov, "provider"),
            "by_model": _rows(by_model, "model")}


@app.post("/api/providers/{name}")
async def update_provider_alias(name: str, request: Request):
    # legacy POST alias kept for compatibility
    return await update_provider(name, request)


@app.put("/api/providers/{name}")
async def update_provider(name: str, request: Request):
    """Edit provider: base_url, prefix, default_model, proxy, weight, keys, manual_models, rename."""
    check_dashboard_auth(request)
    p = PROVIDERS.get(name)
    if not p:
        raise HTTPException(404, "Provider not found")
    body = await request.json()
    # Rename support: new_name != current name
    new_name = body.get("new_name")
    if new_name and new_name.strip() and new_name.strip() != name:
        new_name = new_name.strip()
        if new_name in PROVIDERS and new_name != name:
            raise HTTPException(409, f"Provider '{new_name}' already exists")
        # Re-key in runtime dict
        PROVIDERS[new_name] = PROVIDERS.pop(name)
        p = PROVIDERS[new_name]
        p.name = new_name
        # Re-map model cost overrides
        if name in PROVIDER_MODEL_COSTS:
            PROVIDER_MODEL_COSTS[new_name] = PROVIDER_MODEL_COSTS.pop(name)
    # Update fields if provided
    if "base_url" in body:
        p.base_url = body["base_url"].rstrip("/")
    if "prefix" in body:
        p.prefix = body["prefix"]
    if "default_model" in body:
        p.default_model = body["default_model"]
    if "proxy" in body:
        p.proxy = body["proxy"]
    if "weight" in body:
        p.weight = int(body["weight"])
    if "keys" in body and isinstance(body["keys"], list):
        p.keys = [k for k in body["keys"] if k]
    if "manual_models" in body and isinstance(body["manual_models"], list):
        p.manual_models = [m for m in body["manual_models"] if m]
    if "is_active" in body:
        p.is_active = bool(body["is_active"])
    _save_config()
    return {"ok": True, "provider": p.to_dict(include_key=False)}


# --- proxies ---
@app.post("/api/proxies")
async def add_proxy(request: Request):
    check_dashboard_auth(request)
    body = await request.json()
    name = body.get("name", "").strip()
    url = body.get("url", "").strip()
    if not name or not url:
        raise HTTPException(400, "Name and URL required")
    if name in PROXIES:
        raise HTTPException(400, f"Proxy pool '{name}' sudah ada — pakai nama lain atau hapus dulu")
    if "://" not in url:
        raise HTTPException(400, "URL harus full proxy URL (mis. http://user:pass@ip:port)")
    PROXIES[name] = url
    _save_config()
    return {"ok": True, "proxies": [{"name": k, "url": v, "masked": v[:12] + "..." if len(v) > 15 else v} for k, v in PROXIES.items()]}


@app.delete("/api/proxies/{name}")
async def delete_proxy(name: str, request: Request):
    check_dashboard_auth(request)
    if name not in PROXIES:
        raise HTTPException(404, "Proxy not found")
    del PROXIES[name]
    _save_config()
    return {"ok": True}


# --- combos ---
import re as _re
COMBO_NAME_RE = _re.compile(r"^[A-Za-z0-9._-]+$")

@app.post("/api/combos")
async def add_combo(request: Request):
    check_dashboard_auth(request)
    body = await request.json()
    name = body.get("name", "").strip()
    routes = body.get("routes", [])
    strategy = body.get("strategy", "random")
    if strategy not in ("random", "round_robin", "fallback"):
        strategy = "random"
    if not name or not routes:
        raise HTTPException(400, "Combo name and routes required")
    if not COMBO_NAME_RE.match(name):
        raise HTTPException(400, "Combo name only allows letters, numbers, '-', '_' and '.'")
    # normalize routes: [{provider, model, weight}]
    normalized = []
    for r in routes:
        prov = PROVIDERS.get(r.get("provider", ""))
        if prov:
            normalized.append({
                "provider": prov.name,
                "model": r.get("model", prov.default_model or "auto"),
                "weight": int(r.get("weight", 1)),
            })
    if not normalized:
        raise HTTPException(400, "No valid providers in routes")
    COMBOS[name] = {"routes": normalized, "strategy": strategy, "rr_idx": 0}
    _save_config()
    return {"ok": True, "combo": {"name": name, "routes": normalized, "strategy": strategy}}


@app.patch("/api/combos/{name}")
async def patch_combo(name: str, request: Request):
    check_dashboard_auth(request)
    if name not in COMBOS:
        raise HTTPException(404, "Combo not found")
    body = await request.json()
    # Rename support: new_name != current name
    new_name = body.get("new_name")
    if new_name and new_name.strip() and new_name.strip() != name:
        new_name = new_name.strip()
        if new_name in COMBOS:
            raise HTTPException(409, f"Combo '{new_name}' already exists")
        COMBOS[new_name] = COMBOS.pop(name)
        name = new_name
    strategy = body.get("strategy")
    routes = body.get("routes")
    if strategy is not None:
        if strategy not in ("random", "round_robin", "fallback"):
            raise HTTPException(400, "Invalid strategy")
        COMBOS[name]["strategy"] = strategy
        COMBOS[name]["rr_idx"] = 0
    if routes is not None:
        normalized = []
        for r in routes:
            prov = PROVIDERS.get(r.get("provider", ""))
            if prov:
                normalized.append({
                    "provider": prov.name,
                    "model": r.get("model", prov.default_model or "auto"),
                    "weight": int(r.get("weight", 1)),
                })
        if not normalized:
            raise HTTPException(400, "No valid providers in routes")
        COMBOS[name]["routes"] = normalized
        COMBOS[name]["rr_idx"] = 0
    _save_config()
    return {"ok": True, "combo": {"name": name, "routes": COMBOS[name]["routes"],
                                  "strategy": COMBOS[name].get("strategy", "random")}}


@app.post("/api/combos/{name}/copy")
async def copy_combo(name: str, request: Request):
    check_dashboard_auth(request)
    if name not in COMBOS:
        raise HTTPException(404, "Combo not found")
    src = COMBOS[name]
    new_name = name
    i = 1
    while new_name in COMBOS:
        i += 1
        new_name = f"{name}-copy{i}" if i > 2 else f"{name}-copy"
    COMBOS[new_name] = {
        "routes": [dict(r) for r in src["routes"]],
        "strategy": src.get("strategy", "random"),
        "rr_idx": 0,
    }
    _save_config()
    return {"ok": True, "combo": {"name": new_name, "routes": COMBOS[new_name]["routes"],
                                  "strategy": COMBOS[new_name].get("strategy", "random")}}


@app.delete("/api/combos/{name}")
async def delete_combo(name: str, request: Request):
    check_dashboard_auth(request)
    if name not in COMBOS:
        raise HTTPException(404, "Combo not found")
    del COMBOS[name]
    _save_config()
    return {"ok": True}


@app.post("/api/reload")
async def reload_config(request: Request):
    check_dashboard_auth(request)
    global CONFIG
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
    SERVER_CFG = CONFIG["server"]
    ROUTER_CFG = CONFIG["router"]
    load_config()
    _load_models_cache()
    return {"ok": True, "providers": len(PROVIDERS), "combos": len(COMBOS), "proxies": len(PROXIES)}


# --- history ---
@app.get("/api/history")
async def api_history(request: Request):
    check_dashboard_auth(request)
    items = list(HISTORY)[-500:][::-1]   # newest first, max 500
    # aggregate stats
    agg = {}
    for h in items:
        prov = h.get("provider", "?")
        a = agg.setdefault(prov, {"total": 0, "ok": 0, "err": 0, "ms_sum": 0, "ms_max": 0})
        a["total"] += 1
        if h.get("status") == "ok":
            a["ok"] += 1
        else:
            a["err"] += 1
        a["ms_sum"] += h.get("ms", 0)
        a["ms_max"] = max(a["ms_max"], h.get("ms", 0))
    per_provider = [{"provider": k, **v, "avg_ms": round(v["ms_sum"] / v["total"], 1) if v["total"] else 0}
                    for k, v in sorted(agg.items(), key=lambda x: -x[1]["total"])]
    return {"items": items, "per_provider": per_provider,
            "total": len(items), "ok": sum(1 for h in items if h.get("status") == "ok")}


@app.delete("/api/history")
async def clear_history(request: Request):
    check_dashboard_auth(request)
    HISTORY.clear()
    try:
        os.remove(HISTORY_PATH)
    except Exception:
        pass
    return {"ok": True}


# --- export / import ---
@app.get("/api/export")
async def export_config(request: Request):
    check_dashboard_auth(request)
    cfg = {
        "server": {k: v for k, v in SERVER_CFG.items()},
        "router": ROUTER_CFG,
        "providers": [p.to_dict(include_key=True) for p in PROVIDERS.values()],
        "proxies": [{"name": k, "url": v} for k, v in PROXIES.items()],
        "combos": [{"name": k, "routes": v["routes"], "strategy": v.get("strategy", "random")} for k, v in COMBOS.items()],
    }
    yaml_text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    return Response(content=yaml_text, media_type="application/yaml",
                    headers={"Content-Disposition": 'attachment; filename="vrouter-config.yaml"'})


@app.post("/api/import")
async def import_config(request: Request):
    check_dashboard_auth(request)
    global CONFIG, SERVER_CFG, ROUTER_CFG
    body = await request.body()
    text = body.decode("utf-8", errors="replace")
    try:
        new_cfg = yaml.safe_load(text)
        if not isinstance(new_cfg, dict) or "providers" not in new_cfg:
            raise HTTPException(400, "Invalid config: missing 'providers'")
        # Safety guard: jangan biarkan import kosong wipe config yang ada
        if not new_cfg.get("providers") and PROVIDERS:
            raise HTTPException(400, "Import aborted: config has 0 providers but runtime has "
                                     f"{len(PROVIDERS)}. Refusing to wipe. (Use /api/reload to restore from disk.)")
        # backup current before overwrite
        if os.path.exists(CONFIG_PATH):
            os.rename(CONFIG_PATH, CONFIG_PATH + ".bak")
        with open(CONFIG_PATH, "w") as f:
            f.write(text)
        CONFIG = new_cfg
        SERVER_CFG = CONFIG.get("server", SERVER_CFG)
        ROUTER_CFG = CONFIG.get("router", ROUTER_CFG)
        load_config()
        _load_models_cache()
        return {"ok": True, "providers": len(PROVIDERS), "combos": len(COMBOS), "proxies": len(PROXIES),
                "backup": CONFIG_PATH + ".bak"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid config: {str(e)[:200]}")


# --- 9router sync (Phase 3) ---
def _import_key_list(existing: list, new_keys: list) -> list:
    """Gabung keys: existing dulu, lalu new yang belum ada."""
    out = [k for k in existing if k]
    seen = set(out)
    for k in new_keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _sync_data_to_runtime(data: dict, replace_combos: bool = True) -> dict:
    """Merge hasil sync_9router ke runtime PROVIDERS/PROXIES/COMBOS.
    Returns summary dict."""
    added_providers, updated_providers = 0, 0
    for p in data.get("providers", []):
        name = p["name"]
        if not name or not p.get("base_url"):
            continue
        # Skip providers user explicitly deleted — don't resurrect them
        if name in DELETED_PROVIDERS:
            continue
        if name in PROVIDERS:
            obj = PROVIDERS[name]
            # update non-empty fields only (tinggalkan health stats)
            if p.get("base_url"):
                obj.base_url = p["base_url"]
            if p.get("prefix"):
                obj.prefix = p["prefix"]
            if p.get("type"):
                obj.type = p["type"]
            if p.get("default_model"):
                obj.default_model = p["default_model"]
            if p.get("proxy"):
                obj.proxy = p["proxy"]
            obj.keys = _import_key_list(obj.keys, p.get("keys", []))
            if p.get("models"):
                obj.models = _import_key_list(obj.models, p["models"])
            obj.models_fetched_at = 0.0   # refetch kalau dipakai
            updated_providers += 1
        else:
            PROVIDERS[name] = Provider(
                name=name,
                base_url=p.get("base_url", ""),
                prefix=p.get("prefix", ""),
                type_=p.get("type", "apikey"),
                keys=p.get("keys", []),
                weight=int(p.get("weight", 5)),
                default_model=p.get("default_model", ""),
                proxy=p.get("proxy", ""),
            )
            PROVIDERS[name].models = p.get("models", [])
            added_providers += 1

    added_proxies = 0
    for pr in data.get("proxies", []):
        nm, url = pr.get("name"), pr.get("url")
        if nm and url and nm not in PROXIES:
            PROXIES[nm] = url
            added_proxies += 1

    updated_combos, added_combos = 0, 0
    for cb in data.get("combos", []):
        nm = cb.get("name")
        if not nm:
            continue
        routes = cb.get("routes", [])
        strategy = cb.get("strategy", "random")
        if nm in COMBOS:
            if replace_combos:
                COMBOS[nm]["routes"] = routes
                COMBOS[nm]["strategy"] = strategy
                COMBOS[nm]["rr_idx"] = 0
                updated_combos += 1
        elif routes:
            COMBOS[nm] = {"routes": routes, "strategy": strategy, "rr_idx": 0}
            added_combos += 1

    _save_config()
    return {
        "provider_added": added_providers,
        "provider_updated": updated_providers,
        "proxy_added": added_proxies,
        "combo_added": added_combos,
        "combo_updated": updated_combos,
        "total_providers": len(PROVIDERS),
        "total_keys": sum(len(p.keys) for p in PROVIDERS.values()),
        "total_proxies": len(PROXIES),
        "total_combos": len(COMBOS),
    }


def _sync_fingerprint(data: dict) -> str:
    """Stable hash of 9router data — dipakai buat deteksi perubahan (skip kalau sama)."""
    import hashlib
    try:
        prov = sorted(
            (p["name"], p.get("base_url", ""), p.get("prefix", ""), sorted(p.get("keys", [])))
            for p in data.get("providers", [])
        )
        prox = sorted((p.get("name", ""), p.get("url", "")) for p in data.get("proxies", []))
        combos = sorted(
            (c.get("name", ""), c.get("strategy", "random"),
             sorted((r.get("provider", ""), r.get("model", "")) for r in c.get("routes", [])))
            for c in data.get("combos", [])
        )
        blob = json.dumps({"p": prov, "x": prox, "c": combos}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
    except Exception:
        return ""


async def _after_sync_health_kick():
    """Health check singkat setelah sync (biar provider baru langsung di-ping)."""
    tasks = [_ping_provider(p) for p in PROVIDERS.values() if p.keys]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    global LAST_HEALTH_CHECK
    LAST_HEALTH_CHECK = time.time()


def _reset_provider_state(names: list):
    """Reset health/state Provider yang baru sync (unlock lockout, refresh model cache)."""
    for name in names:
        p = PROVIDERS.get(name)
        if not p:
            continue
        p.locked_until = 0.0
        p.failures = 0
        p.last_error = None
        p.models_fetched_at = 0.0
        # model yang sempat dead di provider ini — kasih kesempatan kedua
        dead_to_clear = [k for k in list(DEAD_MODELS.keys()) if k.startswith(f"{name}/")]
        for k in dead_to_clear:
            DEAD_MODELS.pop(k, None)


async def sync_9router_loop(app: FastAPI):
    """Scheduler: sync dari 9router DB berkala + kick health check kalau berubah."""
    global SYNC_STATE
    # first run after 20s startup
    await asyncio.sleep(20)
    while True:
        interval = int(ROUTER_CFG.get("sync_interval_seconds", 0) or 0)
        if interval <= 0:
            SYNC_STATE["enabled"] = False
            SYNC_STATE["next_run"] = 0
            await asyncio.sleep(60)
            continue
        SYNC_STATE["enabled"] = True
        now = time.time()
        if SYNC_STATE["next_run"] <= 0:
            SYNC_STATE["next_run"] = now + interval
        elif now >= SYNC_STATE["next_run"]:
            SYNC_STATE["running"] = True
            try:
                data = await asyncio.to_thread(sync_9router.sync_from_9router)
                fp = _sync_fingerprint(data)
                prev = SYNC_STATE.get("last_fingerprint")
                if fp and fp != prev:
                    result = _sync_data_to_runtime(data, replace_combos=True)
                    result["warnings"] = data.get("warnings", [])
                    _reset_provider_state([p["name"] for p in data.get("providers", [])])
                    vrouter_db.log_import(source="9router-scheduler", summary=result)
                    _save_config()
                    SYNC_STATE["last_fingerprint"] = fp
                    asyncio.create_task(_after_sync_health_kick())
                    SYNC_STATE["last_result"] = result
                else:
                    SYNC_STATE["last_result"] = {"skipped": True, "reason": "no change"}
            except Exception as e:
                SYNC_STATE["last_result"] = {"error": str(e)}
            finally:
                SYNC_STATE["running"] = False
                SYNC_STATE["last_run"] = time.time()
                SYNC_STATE["next_run"] = time.time() + interval
        await asyncio.sleep(15)


@app.get("/api/import/9router")
async def nine_status(request: Request):
    """Preview sync dari 9router DB (tanpa mengubah runtime)."""
    check_dashboard_or_gateway(request)
    data = sync_9router.sync_from_9router()
    return {"ok": True, "dry": True, **sync_9router.summary(data),
            "nine_db": sync_9router.NINE_ROUTER_DB,
            "exists": os.path.exists(sync_9router.NINE_ROUTER_DB),
            "import_log": vrouter_db.get_import_log(5)}


@app.post("/api/import/9router")
async def api_import_9router(request: Request):
    """Apply sync dari 9router DB ke runtime + config + sqlite + auto-reload state."""
    check_dashboard_or_gateway(request)
    body = await request.json()
    replace_combos = bool(body.get("replace_combos", True))
    data = sync_9router.sync_from_9router()
    result = _sync_data_to_runtime(data, replace_combos=replace_combos)
    result["warnings"] = data.get("warnings", [])
    # Auto-reload: reset state provider yang sync + kasih kesempatan model dead
    _reset_provider_state([p["name"] for p in data.get("providers", [])])
    vrouter_db.log_import(source="9router", summary=result)
    _save_config()
    SYNC_STATE["last_fingerprint"] = _sync_fingerprint(data)
    asyncio.create_task(_after_sync_health_kick())
    return {"ok": True, "applied": True, **result}


@app.get("/api/sync/status")
async def api_sync_status(request: Request):
    """Status scheduler auto-sync + info terakhir."""
    check_dashboard_or_gateway(request)
    return {
        "ok": True,
        "configured_interval": int(ROUTER_CFG.get("sync_interval_seconds", 0) or 0),
        **SYNC_STATE,
    }


@app.post("/api/sync/config")
async def api_sync_config(request: Request):
    """Set interval auto-sync (detik). 0 = mati."""
    check_dashboard_or_gateway(request)
    body = await request.json()
    interval = int(body.get("interval_seconds", 0))
    if interval < 0:
        raise HTTPException(400, "interval_seconds >= 0")
    ROUTER_CFG["sync_interval_seconds"] = interval
    _save_config()
    SYNC_STATE["enabled"] = interval > 0
    SYNC_STATE["next_run"] = time.time() + interval if interval > 0 else 0
    return {"ok": True, "interval_seconds": interval, "enabled": interval > 0}


@app.post("/api/sync/run")
async def api_sync_run(request: Request):
    """Trigger sync sekali sekarang (manual / buat test scheduler)."""
    check_dashboard_or_gateway(request)
    data = sync_9router.sync_from_9router()
    result = _sync_data_to_runtime(data, replace_combos=True)
    result["warnings"] = data.get("warnings", [])
    _reset_provider_state([p["name"] for p in data.get("providers", [])])
    vrouter_db.log_import(source="9router-manual", summary=result)
    _save_config()
    SYNC_STATE["last_fingerprint"] = _sync_fingerprint(data)
    asyncio.create_task(_after_sync_health_kick())
    return {"ok": True, "applied": True, **result}


@app.get("/api/import/log")
async def api_import_log(request: Request):
    check_dashboard_auth(request)
    limit = int(request.query_params.get("limit", 20))
    return {"ok": True, "log": vrouter_db.get_import_log(limit=limit)}


@app.get("/api/db/stats")
async def api_db_stats(request: Request):
    check_dashboard_auth(request)
    return {"ok": True, "stats": vrouter_db.get_db_stats()}


# --- provider models (auto-fetch) ---
@app.post("/api/providers/preview-models")
async def preview_models(request: Request):
    """Fetch /models from an arbitrary provider config (before saving)."""
    check_dashboard_auth(request)
    body = await request.json()
    base_url = body.get("base_url", "").strip().rstrip("/")
    key = body.get("key", "").strip()
    proxy = body.get("proxy", "").strip()
    auth_type = body.get("auth_type", "apikey")
    if not base_url:
        raise HTTPException(400, "base_url required")
    headers = {}
    if auth_type == "apikey" and key:
        headers["Authorization"] = f"Bearer {key}"
    elif auth_type == "oauth" and key:
        headers["Authorization"] = f"Bearer {key}"
    url = f"{base_url}/models"
    proxy_url = resolve_proxy_url(proxy)
    try:
        if proxy_url:
            async with httpx.AsyncClient(timeout=25, proxy=proxy_url) as pc:
                resp = await pc.get(url, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=25) as ac:
                resp = await ac.get(url, headers=headers)
        if resp.status_code < 400:
            data = resp.json().get("data", []) if isinstance(resp.json(), dict) else []
            models = sorted({m.get("id") for m in data if isinstance(m, dict) and m.get("id")})
            return {"ok": True, "models": models, "count": len(models)}
        return {"ok": False, "status": resp.status_code, "error": resp.text[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.post("/api/providers/{name}/models")
async def fetch_provider_models(name: str, request: Request):
    check_dashboard_auth(request)
    p = PROVIDERS.get(name)
    if not p:
        raise HTTPException(404, "Provider not found")
    try:
        body = await request.json()
    except Exception:
        body = {}
    force = bool(body.get("force"))
    ok, result = await asyncio.to_thread(p.fetch_models, force=force)
    if ok:
        return {"ok": True, "models": result, "count": len(result), "provider": name}
    return {"ok": False, "error": result, "provider": name}


@app.post("/api/providers/{name}/models/use")
async def use_provider_model(name: str, request: Request):
    """Set a provider's default_model from its fetched model list."""
    check_dashboard_auth(request)
    p = PROVIDERS.get(name)
    if not p:
        raise HTTPException(404, "Provider not found")
    body = await request.json()
    model = body.get("model", "").strip()
    if not model:
        raise HTTPException(400, "Model required")
    p.default_model = model
    _save_config()
    return {"ok": True, "default_model": p.default_model}


def _save_config():
    cfg = {
        "server": SERVER_CFG,
        "router": ROUTER_CFG,
        "providers": [p.to_dict(include_key=True) for p in PROVIDERS.values()],
        "proxies": [{"name": k, "url": v} for k, v in PROXIES.items()],
        "combos": [{"name": k, "routes": v["routes"], "strategy": v.get("strategy", "random")} for k, v in COMBOS.items()],
        "deleted_providers": sorted(DELETED_PROVIDERS),
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    # Phase 3: mirror ke SQLite
    try:
        vrouter_db.save_snapshot(PROVIDERS, PROXIES, COMBOS)
    except Exception:
        pass



# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# ADMIN LOGIN (console.vrouter.my.id — reads public.db)
# ═══════════════════════════════════════════════════════════════════

@app.post("/admin/login")
async def admin_login(request: Request):
    """Login via email + password (from public.db users table)."""
    body = await request.json()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    db = _admin_db()
    user = db.execute("SELECT id, email, password_hash, role, is_admin FROM users WHERE email=?", (email,)).fetchone()
    db.close()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Verify password
    stored = user["password_hash"]
    if stored:
        import hashlib as _hl
        try:
            salt, h = stored.split(":", 1)
            check = _hl.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
            if check != h:
                raise HTTPException(status_code=401, detail="Invalid credentials")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    else:
        raise HTTPException(status_code=401, detail="No password set")

    is_admin = (user["role"] == "admin") or (user["is_admin"] or 0) == 1
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    # Session token = user id
    session_token = str(user["id"])

    from fastapi.responses import JSONResponse
    resp = JSONResponse({"ok": True, "email": email, "is_admin": True})
    resp.set_cookie("vrouter_email", email, path="/", samesite="lax", max_age=86400 * 7)
    resp.set_cookie("vrouter_session", session_token, path="/", samesite="lax", max_age=86400 * 7)
    return resp


@app.post("/admin/logout")
async def admin_logout():
    """Clear admin cookies."""
    from fastapi.responses import JSONResponse
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("vrouter_email", path="/")
    resp.delete_cookie("vrouter_session", path="/")
    return resp


@app.get("/admin/api/me")
async def admin_me(request: Request):
    """Return current admin user info."""
    email, is_admin = _admin_user(request)
    if not email:
        raise HTTPException(status_code=401, detail="Not logged in")
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return {"email": email, "is_admin": True}


# ═══════════════════════════════════════════════════════════════════
# ADMIN — MODEL MANAGEMENT (console.vrouter.my.id)
# ═══════════════════════════════════════════════════════════════════
# ADMIN — MODEL MANAGEMENT (console.vrouter.my.id)
# ═══════════════════════════════════════════════════════════════════

def _admin_db():
    db = sqlite3.connect(PUBLIC_DB_PATH)
    db.row_factory = sqlite3.Row
    return db

def _admin_user(request):
    email = request.cookies.get("vrouter_email")
    session_id = request.cookies.get("vrouter_session")
    if not email or not session_id:
        return None, False
    db = _admin_db()
    user = db.execute("SELECT role, is_admin FROM users WHERE email=? AND id=?", (email, session_id)).fetchone()
    db.close()
    if not user:
        return None, False
    is_admin = (user["role"] == "admin") or (user["is_admin"] or 0) == 1
    return email, is_admin

def require_admin(request):
    from fastapi.responses import RedirectResponse
    email, is_admin = _admin_user(request)
    if not email:
        raise HTTPException(status_code=401, detail="Not logged in")
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return email

def check_admin(request):
    """Accept admin session OR dashboard session (vr_token)."""
    # Try admin session first
    email, is_admin = _admin_user(request)
    if email and is_admin:
        return email
    # Try dashboard session
    token = request.cookies.get("vr_token")
    if token and token in VALID_SESSIONS:
        return "dashboard"
    raise HTTPException(status_code=401, detail="Admin or dashboard login required")

# Legacy alias
require_admin_orig = require_admin

@app.get("/admin")
async def admin_page(request: Request):
    from fastapi.responses import RedirectResponse
    email, is_admin = _admin_user(request)
    if not email:
        return RedirectResponse(url="/auth?next=/admin", status_code=302)
    if not is_admin:
        return RedirectResponse(url="/", status_code=302)
    resp = templates.TemplateResponse(request, "admin.html", {
        "site_name": SITE_NAME,
        "site_url": SITE_URL,
        "admin_email": email,
    })
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp

@app.get("/admin/api/models")
async def admin_list_models(request: Request):
    check_admin(request)
    backend_models = []
    try:
        resp = await request.app.state.client.get(
            f"{SITE_URL}/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        data = resp.json()
        backend_models = data.get("data", [])
    except Exception as e:
        print(f"[Admin] Backend models fetch error: {e}")
    db = _admin_db()
    config_rows = db.execute("SELECT * FROM models_config").fetchall()
    config = {r["model_id"]: dict(r) for r in config_rows}
    db.close()
    models = []
    seen = set()
    for m in backend_models:
        mid = m.get("id", "")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        cfg = config.get(mid, {})
        if cfg.get("hidden"):
            continue
        models.append({
            "id": mid,
            "name": mid.split("/")[-1].upper().replace("-", " ") if "/" in mid else mid.upper().replace("-", " "),
            "provider": mid.split("/")[0] if "/" in mid else "direct",
            "enabled": cfg.get("enabled", 1),
            "tier": cfg.get("tier", "free"),
            "display_name": cfg.get("display_name", ""),
            "sort_order": cfg.get("sort_order", 0),
            "in_config": mid in config,
        })
    for mid, cfg in config.items():
        if mid not in seen:
            seen.add(mid)
            if cfg.get("hidden"):
                continue
            models.append({
                "id": mid,
                "name": cfg.get("display_name", mid.split("/")[-1].upper().replace("-", " ")),
                "provider": cfg.get("provider", "unknown"),
                "enabled": cfg.get("enabled", 0),
                "tier": cfg.get("tier", "disabled"),
                "display_name": cfg.get("display_name", ""),
                "sort_order": cfg.get("sort_order", 0),
                "in_config": True,
                "removed": True,
            })
    tier_order = {"free": 0, "pro": 1, "disabled": 2}
    models.sort(key=lambda x: (0 if x["enabled"] else 1, tier_order.get(x["tier"], 3), x["sort_order"]))
    return {"models": models, "total": len(models)}

@app.post("/admin/api/models/sync")
async def admin_sync_models(request: Request):
    check_admin(request)
    try:
        resp = await request.app.state.client.get(
            f"{SITE_URL}/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        data = resp.json()
        backend_models = data.get("data", [])
    except Exception as e:
        return {"ok": False, "error": str(e)}
    db = _admin_db()
    added = 0
    skipped = 0
    for m in backend_models:
        mid = m.get("id", "")
        if not mid:
            continue
        exists = db.execute("SELECT id FROM models_config WHERE model_id=?", (mid,)).fetchone()
        if exists:
            skipped += 1
            continue
        provider = mid.split("/")[0] if "/" in mid else "direct"
        display = mid.split("/")[-1].upper().replace("-", " ") if "/" in mid else mid.upper().replace("-", " ")
        db.execute(
            "INSERT INTO models_config (model_id, display_name, provider, enabled, tier, sort_order) VALUES (?, ?, ?, 1, 'free', ?)",
            (mid, display, provider, added)
        )
        added += 1
    db.commit()
    db.close()
    return {"ok": True, "added": added, "skipped": skipped, "total": len(backend_models)}

@app.put("/admin/api/models/update")
async def admin_update_model(request: Request):
    check_admin(request)
    body = await request.json()
    model_id = body.get("model_id")
    db = _admin_db()
    exists = db.execute("SELECT id FROM models_config WHERE model_id=?", (model_id,)).fetchone()
    if exists:
        provider = body.get("provider", "")
        if provider:
            db.execute("UPDATE models_config SET enabled=?, tier=?, display_name=?, sort_order=?, hidden=?, provider=?, updated_at=datetime('now') WHERE model_id=?",
                (body.get("enabled", 1), body.get("tier", "free"), body.get("display_name", ""), body.get("sort_order", 0), body.get("hidden", 0), provider, model_id))
        else:
            db.execute("UPDATE models_config SET enabled=?, tier=?, display_name=?, sort_order=?, hidden=?, updated_at=datetime('now') WHERE model_id=?",
                (body.get("enabled", 1), body.get("tier", "free"), body.get("display_name", ""), body.get("sort_order", 0), body.get("hidden", 0), model_id))
    else:
        provider = body.get("provider", "") or (model_id.split("/")[0] if "/" in model_id else "direct")
        db.execute("INSERT INTO models_config (model_id, display_name, provider, enabled, tier, sort_order) VALUES (?, ?, ?, ?, ?, ?)",
            (model_id, body.get("display_name", model_id.split("/")[-1].upper().replace("-", " ")), provider, body.get("enabled", 1), body.get("tier", "free"), body.get("sort_order", 0)))
    db.commit()
    db.close()
    return {"ok": True}

@app.delete("/admin/api/models/delete")
async def admin_delete_model(request: Request):
    check_admin(request)
    body = await request.json()
    model_id = body.get("model_id")
    if not model_id:
        return {"ok": False, "detail": "model_id required"}
    db = _admin_db()
    exists = db.execute("SELECT id FROM models_config WHERE model_id=?", (model_id,)).fetchone()
    if exists:
        db.execute("UPDATE models_config SET hidden=1, updated_at=datetime('now') WHERE model_id=?", (model_id,))
    else:
        provider = model_id.split("/")[0] if "/" in model_id else "direct"
        display = model_id.split("/")[-1].upper().replace("-", " ") if "/" in model_id else model_id.upper().replace("-", " ")
        db.execute("INSERT INTO models_config (model_id, display_name, provider, enabled, tier, sort_order, hidden) VALUES (?, ?, ?, 0, 'disabled', 0, 1)",
            (model_id, display, provider))
    db.commit()
    db.close()
    return {"ok": True}

@app.post("/admin/api/models/bulk")
async def admin_bulk_update(request: Request):
    check_admin(request)
    body = await request.json()
    updates = body.get("models", [])
    db = _admin_db()
    updated = 0
    for u in updates:
        mid = u.get("model_id")
        if not mid:
            continue
        exists = db.execute("SELECT id FROM models_config WHERE model_id=?", (mid,)).fetchone()
        if exists:
            db.execute("UPDATE models_config SET enabled=?, tier=?, updated_at=datetime('now') WHERE model_id=?", (u.get("enabled", 1), u.get("tier", "free"), mid))
        else:
            provider = mid.split("/")[0] if "/" in mid else "direct"
            db.execute("INSERT INTO models_config (model_id, display_name, provider, enabled, tier, sort_order) VALUES (?, ?, ?, ?, ?, 0)",
                (mid, mid.split("/")[-1].upper().replace("-", " "), provider, u.get("enabled", 1), u.get("tier", "free")))
        updated += 1
    db.commit()
    db.close()
    return {"ok": True, "updated": updated}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_CFG.get("port", 20129), log_level="info")
