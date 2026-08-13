#!/usr/bin/env python3
"""
VRouter Phase 3 — sync_9router.py
=================================
Membaca database 9router (SQLite) dan memetakan ke format config VRouter:

  providerConnections  -> providers  (base_url, prefix, type, keys, default_model)
  providerNodes        -> base_url/prefix fallback untuk openai-compatible nodes
  combos               -> combos     (routes diambil dari model id ber-prefix)
  proxyPools           -> proxies    (name + proxyUrl)
  kv/customModels      -> models list per provider (harvest)

Output: dict {"providers": [...], "proxies": [...], "combos": [...]}
yang bisa langsung di-merge ke runtime VRouter atau disimpan ke config.yaml.

Penting: file ini TIDAK mengubah state 9router — read-only.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

NINE_ROUTER_DB = os.environ.get("NINE_ROUTER_DB", "/home/ubuntu/.9router/db/data.sqlite")

# authType 9router -> type VRouter (Provider.auth_header handle ketiganya)
AUTH_TYPE_MAP = {
    "apikey": "apikey",
    "oauth": "oauth",
    "Authorization": "Authorization",
    "bearer": "apikey",
}


def _json(data: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(data)
    except Exception:
        return None


def _unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_provider_nodes(cur: sqlite3.Cursor) -> Dict[str, Dict[str, Any]]:
    """id -> {prefix, baseUrl, apiType, nodeName}"""
    nodes: Dict[str, Dict[str, Any]] = {}
    try:
        rows = cur.execute("SELECT id, data FROM providerNodes").fetchall()
    except sqlite3.OperationalError:
        return nodes
    for rid, data in rows:
        d = _json(data) or {}
        nodes[rid] = d
    return nodes


def read_custom_models(cur: sqlite3.Cursor) -> Dict[str, List[str]]:
    """prefix/alias -> [model ids] dari scope customModels."""
    models: Dict[str, List[str]] = {}
    try:
        rows = cur.execute("SELECT key, value FROM kv WHERE scope='customModels'").fetchall()
    except sqlite3.OperationalError:
        return models
    for key, value in rows:
        v = _json(value)
        if not v:
            continue
        alias = v.get("providerAlias") or key
        mid = v.get("id") or key
        # key bisa "alias:model" (format lama) atau "alias" (kv per model)
        if ":" in key and alias == key.split(":")[0]:
            # entry value berisi id sendiri -> single model entry
            pass
        models.setdefault(alias, []).append(mid)
    return models


def read_providers(
    cur: sqlite3.Cursor, nodes: Dict[str, Dict[str, Any]], custom_models: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """Grouping providerConnections -> daftar provider VRouter."""
    try:
        rows = cur.execute(
            "SELECT provider, authType, data FROM providerConnections WHERE isActive=1 ORDER BY priority DESC, createdAt"
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for provider, auth_type, data in rows:
        d = _json(data) or {}
        grouped.setdefault(provider, []).append({"authType": auth_type, "data": d})

    providers: List[Dict[str, Any]] = []
    for provider, conns in grouped.items():
        first = conns[0]["data"]
        psd = first.get("providerSpecificData") or {}
        node = nodes.get(provider, {}) or {}

        base_url = psd.get("baseUrl") or node.get("baseUrl") or ""
        # normalize: strip trailing slash + "/chat/completions" -> root /v1
        if base_url:
            if base_url.endswith("/chat/completions"):
                base_url = base_url[: -len("/chat/completions")]
            base_url = base_url.rstrip("/")
        prefix = psd.get("prefix") or node.get("prefix") or ""
        auth_type = first.get("authType") or "apikey"
        vtype = AUTH_TYPE_MAP.get(auth_type, "apikey")
        default_model = first.get("defaultModel") or psd.get("defaultModel") or ""

        keys: List[str] = []
        for c in conns:
            d = c["data"]
            k = d.get("apiKey") or d.get("accessToken") or d.get("token") or ""
            if k and k not in keys and k != "***":
                keys.append(k)

        # harvest models: dari custom_models (by prefix, by provider name)
        models: List[str] = []
        for alias in (prefix, provider):
            models.extend(custom_models.get(alias, []))
        models = _unique(models)

        providers.append({
            "name": provider,
            "base_url": base_url,
            "prefix": prefix,
            "type": vtype,
            "weight": 5,
            "default_model": default_model,
            "proxy": "",
            "keys": keys,
            "models": models,
        })
    return providers


def read_proxies(cur: sqlite3.Cursor) -> List[Dict[str, str]]:
    """proxyPools yang aktif -> [{name, url}]."""
    proxies: List[Dict[str, str]] = []
    try:
        rows = cur.execute("SELECT isActive, data FROM proxyPools ORDER BY createdAt").fetchall()
    except sqlite3.OperationalError:
        return proxies
    for is_active, data in rows:
        if not is_active:
            continue
        d = _json(data) or {}
        name = d.get("name") or ""
        url = d.get("proxyUrl") or ""
        if name and url and "://" in url:
            proxies.append({"name": name, "url": url})
    return proxies


def read_combos(
    cur: sqlite3.Cursor, providers: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """combos 9router (models = list "prefix/model") -> VRouter routes."""
    # prefix -> provider name lookup
    prefix_map: Dict[str, str] = {}
    name_map: Dict[str, str] = {}
    for p in providers:
        if p["prefix"]:
            prefix_map[p["prefix"]] = p["name"]
        name_map[p["name"]] = p["name"]

    combos: List[Dict[str, Any]] = []
    try:
        rows = cur.execute("SELECT name, kind, models FROM combos").fetchall()
    except sqlite3.OperationalError:
        return combos

    for name, kind, models_json in rows:
        mlist = _json(models_json)
        if not isinstance(mlist, list):
            continue
        routes: List[Dict[str, Any]] = []
        for m in mlist:
            if not isinstance(m, str) or "/" not in m:
                continue
            head, _, tail = m.partition("/")
            prov = prefix_map.get(head) or name_map.get(head)
            if prov:
                routes.append({"provider": prov, "model": tail, "weight": 1})
        if routes:
            combos.append({"name": name, "strategy": "random", "routes": routes})
    return combos


def sync_from_9router(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Entry point: baca 9router DB -> {providers, proxies, combos, warnings}."""
    db_path = db_path or NINE_ROUTER_DB
    warnings: List[str] = []
    if not os.path.exists(db_path):
        return {"providers": [], "proxies": [], "combos": [], "warnings": [f"9router DB not found: {db_path}"]}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
    except Exception as e:
        return {"providers": [], "proxies": [], "combos": [], "warnings": [f"9router DB open failed: {e}"]}

    try:
        nodes = read_provider_nodes(cur)
        custom_models = read_custom_models(cur)
        providers = read_providers(cur, nodes, custom_models)
        proxies = read_proxies(cur)
        combos = read_combos(cur, providers)

        # filter provider tanpa base_url (tidak berguna utk routing)
        no_url = [p["name"] for p in providers if not p["base_url"]]
        if no_url:
            warnings.append(f"providers tanpa base_url di-skip: {', '.join(no_url[:5])}")
        providers = [p for p in providers if p["base_url"]]

        return {
            "providers": providers,
            "proxies": proxies,
            "combos": combos,
            "warnings": warnings,
        }
    finally:
        conn.close()


def summary(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "providers": len(data["providers"]),
        "total_keys": sum(len(p["keys"]) for p in data["providers"]),
        "proxies": len(data["proxies"]),
        "combos": len(data["combos"]),
        "warnings": data["warnings"],
        "provider_list": [
            {"name": p["name"], "prefix": p["prefix"], "type": p["type"],
             "keys": len(p["keys"]), "models": len(p["models"]), "base_url": p["base_url"]}
            for p in data["providers"]
        ],
    }


if __name__ == "__main__":
    import sys

    data = sync_from_9router()
    s = summary(data)
    print(json.dumps(s, indent=2))
