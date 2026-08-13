#!/usr/bin/env python3
"""
import_9router.py — CLI sync 9router DB -> VRouter
==================================================
Cara pakai:
  python3 import_9router.py --dry           # preview (read-only)
  python3 import_9router.py --apply         # sync ke runtime + config + sqlite
  python3 import_9router.py --status        # DB stats + import log
  python3 import_9router.py --mirror        # bangun ulang snapshot sqlite dari config.yaml saat ini

Opsional:
  --host http://127.0.0.1:20129   # kalau mau lewat API running instance
  --db /home/ubuntu/.9router/db/data.sqlite

Dengan --host, --apply akan memanggil API VRouter yang sedang jalan
(auth pakai cookie-based, jadi dipakai untuk preview via CLI langsung ke sqlite;
apply via API hanya kalau server hidup).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sync_9router
import vrouter_db


def main():
    ap = argparse.ArgumentParser(description="Sync 9router -> VRouter")
    ap.add_argument("--dryrun", action="store_true", help="preview tanpa apply")
    ap.add_argument("--apply", action="store_true", help="merge ke runtime + save config/sqlite")
    ap.add_argument("--status", action="store_true", help="lihat stats DB + import log")
    ap.add_argument("--mirror", action="store_true", help="tulis snapshot sqlite dari config.yaml saat ini")
    ap.add_argument("--host", default="http://127.0.0.1:20129", help="VRouter base URL")
    ap.add_argument("--cred", default=None, help="isi untuk keperluan masa depan (unused)")
    ap.add_argument("--replace-combos", action="store_true", default=True, help="ganti combos yang sama (default)")
    ap.add_argument("--keep-combos", action="store_true", help="jangan menimpa combo yang sudah ada")
    args = ap.parse_args()

    if args.status:
        print(json.dumps({"db_stats": vrouter_db.get_db_stats(),
                          "import_log": vrouter_db.get_import_log(10)}, indent=2))
        return 0

    if args.mirror:
        # bangun ulang snapshot dari config.yaml
        import yaml
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))
        # cuma provider runtime-to-db memerlukan objek; untuk CLI mirror:
        # baca config langsung -> tulis baris providers/proxies/combos
        c = vrouter_db.conn()
        c.execute("DELETE FROM providers")
        c.execute("DELETE FROM proxies")
        c.execute("DELETE FROM combos")
        for p in cfg.get("providers", []):
            c.execute(
                "INSERT OR REPLACE INTO providers (name, base_url, prefix, type, weight, default_model, proxy, keys_json, models_json, failures) VALUES (?,?,?,?,?,?,?,?,?,0)",
                (p.get("name"), p.get("base_url"), p.get("prefix", ""), p.get("type", "apikey"),
                 p.get("weight", 5), p.get("default_model", ""), p.get("proxy", ""),
                 json.dumps(p.get("keys", [])), json.dumps(p.get("models", []))))
        for pr in cfg.get("proxies", []):
            c.execute("INSERT OR REPLACE INTO proxies (name, url) VALUES (?,?)",
                      (pr["name"], pr.get("url", "")))
        for cb in cfg.get("combos", []):
            c.execute("INSERT OR REPLACE INTO combos (name, strategy, routes_json) VALUES (?,?,?)",
                      (cb.get("name"), cb.get("strategy", "random"), json.dumps(cb.get("routes", []))))
        c.commit()
        c.close()
        print(json.dumps({"ok": True, "mirror": "config.yaml -> vrouter.db",
                          "stats": vrouter_db.get_db_stats()}, indent=2))
        return 0

    data = sync_9router.sync_from_9router()
    s = sync_9router.summary(data)
    if args.dryrun or not args.apply:
        print(json.dumps({"dryrun": True, **s}, indent=2))
        return 0

    # apply via API (server hidup)
    import urllib.request

    api_key = os.environ.get("VROUTER_API_KEY", "hermes-router-2026")
    req = urllib.request.Request(
        args.host.rstrip("/") + "/api/import/9router",
        data=json.dumps({"replace_combos": not args.keep_combos}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            print(json.dumps({"applied": True, **body}, indent=2))
            return 0
    except Exception as e:
        print(json.dumps({"applied": False, "error": str(e)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())