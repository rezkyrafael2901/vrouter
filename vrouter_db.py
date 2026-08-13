#!/usr/bin/env python3
"""
VRouter Phase 3 — vrouter_db.py
================================
SQLite persistence layer untuk VRouter.

Tables:
  providers   — satu baris per provider (config + keys + models + runtime state)
  proxies     — name -> url
  combos      — name -> strategy + routes
  import_log  — riwayat import/sync (9router, file import, dll)
  meta        — key-value (schema_version, last_sync, dll)

Semua operasi thread-safe via checkpoint/commit per call; DB default
wal mode supaya baca-tulis sejalan sama FastAPI async.
"""

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

DB_PATH = os.environ.get("VROUTER_DB", "/home/ubuntu/vrouter/vrouter.db")


def conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    c = sqlite3.connect(path, timeout=10)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    return c


def init_db(db_path: Optional[str] = None) -> None:
    c = conn(db_path)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS providers (
        name            TEXT PRIMARY KEY,
        base_url        TEXT,
        prefix          TEXT,
        type            TEXT,
        weight          INTEGER DEFAULT 5,
        default_model   TEXT,
        proxy           TEXT,
        keys_json       TEXT,
        models_json     TEXT,
        failures        INTEGER DEFAULT 0,
        total_requests  INTEGER DEFAULT 0,
        total_errors    INTEGER DEFAULT 0,
        last_error      TEXT,
        last_used       REAL,
        locked_until    REAL DEFAULT 0,
        dead            INTEGER DEFAULT 0,
        updated_at      REAL
    );
    CREATE TABLE IF NOT EXISTS proxies (
        name    TEXT PRIMARY KEY,
        url     TEXT
    );
    CREATE TABLE IF NOT EXISTS combos (
        name        TEXT PRIMARY KEY,
        strategy    TEXT DEFAULT 'random',
        routes_json TEXT
    );
    CREATE TABLE IF NOT EXISTS import_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          REAL,
        source      TEXT,
        summary     TEXT
    );
    CREATE TABLE IF NOT EXISTS meta (
        key     TEXT PRIMARY KEY,
        value   TEXT
    );
    """)
    c.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version','3')")
    c.commit()
    c.close()


def save_provider(p: Any, db_path: Optional[str] = None) -> None:
    """p = Provider object dari main.py."""
    c = conn(db_path)
    c.execute("""
    INSERT OR REPLACE INTO providers
        (name, base_url, prefix, type, weight, default_model, proxy,
         keys_json, models_json, failures, total_requests, total_errors,
         last_error, last_used, locked_until, dead)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        p.name, p.base_url, p.prefix, p.type, getattr(p, "weight", 5),
        p.default_model, getattr(p, "proxy", ""),
        json.dumps(p.keys),
        json.dumps(p.models),
        p.failures, p.total_requests, p.total_errors,
        p.last_error, p.last_used, p.locked_until,
        int(p.failures >= 3),
    ))
    c.commit()
    c.close()


def save_snapshot(
    providers: Dict[str, Any],
    proxies: Dict[str, str],
    combos: Dict[str, Dict[str, Any]],
    db_path: Optional[str] = None,
) -> None:
    """Snapshot semua runtime objects ke SQLite (REPLACE per baris)."""
    c = conn(db_path)
    ts = time.time()
    for name, p in providers.items():
        c.execute(
            "INSERT OR REPLACE INTO providers (name, base_url, prefix, type, weight, default_model, proxy, keys_json, models_json, failures, total_requests, total_errors, last_error, last_used, locked_until, dead) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.name, p.base_url, p.prefix, p.type, getattr(p, "weight", 5),
             p.default_model, getattr(p, "proxy", ""),
             json.dumps(getattr(p, "keys", [])),
             json.dumps(getattr(p, "models", [])),
             getattr(p, "failures", 0), getattr(p, "total_requests", 0),
             getattr(p, "total_errors", 0), getattr(p, "last_error", None),
             getattr(p, "last_used", None), getattr(p, "locked_until", 0.0),
             int(getattr(p, "failures", 0) >= 3)))
    for name, url in proxies.items():
        c.execute("INSERT OR REPLACE INTO proxies (name, url) VALUES (?,?)", (name, url))
    for name, combo in combos.items():
        c.execute("INSERT OR REPLACE INTO combos (name, strategy, routes_json) VALUES (?,?,?)",
                  (name, combo.get("strategy", "random"), json.dumps(combo.get("routes", []))))
    c.execute("INSERT OR REPLACE INTO meta (key,value) VALUES ('last_snapshot',?)", (str(ts),))
    c.commit()
    c.close()


def load_snapshot(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Load snapshot dari SQLite -> {providers: {name: {...}}, proxies: {...}, combos: {...}}."""
    c = conn(db_path)
    providers: Dict[str, Dict[str, Any]] = {}
    for r in c.execute("SELECT * FROM providers").fetchall():
        providers[r["name"]] = {
            "name": r["name"],
            "base_url": r["base_url"],
            "prefix": r["prefix"],
            "type": r["type"],
            "weight": r["weight"],
            "default_model": r["default_model"],
            "proxy": r["proxy"],
            "keys": json.loads(r["keys_json"] or "[]"),
            "models": json.loads(r["models_json"] or "[]"),
            "failures": r["failures"],
            "total_requests": r["total_requests"],
            "total_errors": r["total_errors"],
            "last_error": r["last_error"],
            "last_used": r["last_used"],
            "locked_until": r["locked_until"],
        }
    proxies = {r["name"]: r["url"] for r in c.execute("SELECT name, url FROM proxies").fetchall()}
    combos = {}
    for r in c.execute("SELECT name, strategy, routes_json FROM combos").fetchall():
        combos[r["name"]] = {
            "strategy": r["strategy"],
            "routes": json.loads(r["routes_json"] or "[]"),
            "rr_idx": 0,
        }
    c.close()
    return {"providers": providers, "proxies": proxies, "combos": combos}


def log_import(source: str, summary: Dict[str, Any], db_path: Optional[str] = None) -> None:
    c = conn(db_path)
    c.execute("INSERT INTO import_log (ts, source, summary) VALUES (?,?,?)",
              (time.time(), source, json.dumps(summary)))
    c.commit()
    c.close()


def get_import_log(limit: int = 10, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    c = conn(db_path)
    rows = c.execute("SELECT id, ts, source, summary FROM import_log ORDER BY id DESC LIMIT ?",
                     (limit,)).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "ts": r["ts"],
            "source": r["source"],
            "summary": json.loads(r["summary"] or "{}"),
        })
    c.close()
    return out


def get_meta(key: str, default: str = "", db_path: Optional[str] = None) -> str:
    c = conn(db_path)
    r = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    c.close()
    return r["value"] if r else default


def get_db_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    c = conn(db_path)
    prov = c.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
    keys = c.execute("SELECT COUNT(*) FROM providers WHERE keys_json != '[]' AND keys_json IS NOT NULL").fetchone()[0]
    prox = c.execute("SELECT COUNT(*) FROM proxies").fetchone()[0]
    cmb = c.execute("SELECT COUNT(*) FROM combos").fetchone()[0]
    imp = c.execute("SELECT COUNT(*) FROM import_log").fetchone()[0]
    last = c.execute("SELECT value FROM meta WHERE key='last_snapshot'").fetchone()
    c.close()
    return {
        "providers": prov,
        "providers_with_keys": keys,
        "proxies": prox,
        "combos": cmb,
        "imports": imp,
        "last_snapshot": float(last["value"]) if last else None,
    }


if __name__ == "__main__":
    init_db()
    print(json.dumps(get_db_stats(), indent=2))