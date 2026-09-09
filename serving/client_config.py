"""Local configuration only: importing this module never contacts RunPod."""
import os
from pathlib import Path


def load_local_env(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    path = root / '.env'
    allowed = {'RUNPOD_API_KEY', 'RUNPOD_ENDPOINT_ID', 'CLAIM_ENDPOINT_PAUSED'}
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if '=' not in line or line.lstrip().startswith('#'):
                continue
            key, value = line.split('=', 1)
            key, value = key.strip(), value.strip().strip('\"\'')
            if key in allowed and value:
                os.environ.setdefault(key, value)
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
