"""Pull the base model weights up front.

62.5 GB dominates the wall clock, and it is needed regardless of when the
dataset becomes reachable, so it runs before the dataset stage.
"""
from __future__ import annotations

import os

from common import log

MODEL_ID = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
REVISION = os.environ.get("BASE_MODEL_REVISION", "main")


def main() -> None:
    from huggingface_hub import snapshot_download

    log(f"Downloading {MODEL_ID}@{REVISION} weights.")
    path = snapshot_download(
        MODEL_ID, revision=REVISION, max_workers=8,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt"],
    )
    log(f"Base model cached at {path}")

    # Record the exact commit so the report can pin the revision precisely.
    from huggingface_hub import HfApi

    info = HfApi().model_info(MODEL_ID, revision=REVISION)
    from common import OUT, write_json

    write_json(OUT / "base_model_revision.json", {
        "model_id": MODEL_ID,
        "requested_revision": REVISION,
        "resolved_sha": info.sha,
        "local_path": str(path),
    })
    log(f"Resolved revision sha={info.sha}")


if __name__ == "__main__":
    main()
