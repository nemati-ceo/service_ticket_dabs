"""servicenow_source.py — pull incidents live from the ServiceNow REST gateway.

Alternative stage-01 input source (config: source.type == "servicenow"). The NM
gateway authenticates with an `apikey` header — NOT `Authorization: Bearer` — so
sending the key any other way returns "not allowed". Returns a pandas DataFrame
shaped like the Delta input table (`tables.input`).
"""

import os

import pandas as pd


def _api_key(cfg):
    """Read the ServiceNow API key from the Databricks secret scope (env fallback)."""
    sec = cfg.get("secrets") or {}
    scope = sec.get("scope")
    key_name = sec.get("snow_api_key", "snow-api-key")
    try:
        return dbutils.secrets.get(scope, key_name)   # dbutils: notebook global
    except Exception:
        return os.environ.get("SNOW_API_KEY", "")


def fetch_incidents(cfg, numbers=None):
    """Fetch incidents from ServiceNow -> pandas DataFrame.

    `numbers` overrides the configured `servicenow.incident_numbers` list.
    """
    import requests

    sn = cfg.get("servicenow") or {}
    base_url = sn["base_url"].rstrip("/")
    numbers = numbers or sn.get("incident_numbers") or []
    timeout = sn.get("timeout", 30)
    headers = {"apikey": _api_key(cfg)}   # <-- the key fix: header name is `apikey`

    rows = []
    for number in numbers:
        url = f"{base_url}/v1/incidents/{number}"
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        # Gateway wraps a single record under "result"; fall back to the raw body.
        rows.append(payload.get("result", payload) if isinstance(payload, dict) else payload)

    df = pd.json_normalize(rows)
    print(f"[servicenow] fetched {len(df)} incident(s) from {base_url}")
    return df
