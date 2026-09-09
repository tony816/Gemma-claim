"""Start the local UI, or check configuration without a model/API call."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'serving'))
from client_config import load_local_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='local configuration only; no network')
    parser.add_argument('--port', type=int, default=None,
                        help='fixed port; otherwise find a free port starting at 7860')
    args = parser.parse_args()
    load_local_env(ROOT)
    from claim_client import DEFAULT_ENDPOINT, TUNED_MODEL
    if args.check:
        print(json.dumps({
            'endpoint': os.environ.get('RUNPOD_ENDPOINT_ID', DEFAULT_ENDPOINT),
            'model': TUNED_MODEL,
            'api_key_configured': bool(os.environ.get('RUNPOD_API_KEY', '').strip()),
            'requests_paused': os.environ.get('CLAIM_ENDPOINT_PAUSED') == '1',
            'remote_status_checked': False,
        }, indent=2))
        return
    sys.path.insert(0, str(ROOT / 'space'))
    from test_ui import demo
    demo.queue(default_concurrency_limit=1).launch(
        server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=True,
    )


if __name__ == '__main__':
    main()
