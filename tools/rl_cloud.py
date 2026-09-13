"""Private in-process credentials and bounded provider requests for RL operations."""
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'run_artifacts/rl_v3_20260908'


def credentials():
    for line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip().strip('\"\'')


def runpod(path, method='GET', payload=None, v1=False):
    credentials()
    base = 'https://rest.runpod.io/v1' if v1 else 'https://api.runpod.io/v2'
    response = requests.request(method, base + path,
        headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY'],
                 'User-Agent': 'Mozilla/5.0'}, json=payload, timeout=40)
    if not response.ok:
        raise RuntimeError(f'RunPod {method} {path.split("?")[0]} HTTP {response.status_code}')
    return response.json() if response.content else None


def graphql(query, variables=None):
    credentials()
    response = requests.post('https://api.runpod.io/graphql',
        headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY'],
                 'User-Agent': 'Mozilla/5.0'},
        json={'query': query, 'variables': variables or {}}, timeout=40)
    return response.json()


def save(name, value):
    """Caller must supply only redacted/allowlisted data."""
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
