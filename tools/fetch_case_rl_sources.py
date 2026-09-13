"""Download frozen allowed train/validation inputs only; no models or GPU calls."""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_case_rl import digest, save_json

REPO = "Mepeng22/gemma-claim-v2-approved-20260902"
REVISION = "e978f526d0b15f6c981a4b82fd25404cef68a8d7"
HASHES = {"train": "69c780015eb37da06257cb33795963d77db5d8b60fd0602620ab18b792b561d3",
          "validation": "69c28ae4b5205a9a13ed5cec2081231936e991a123100f8264c6b54e110d2db0"}


def main():
    from huggingface_hub import HfApi, hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    token = None
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            if key.strip() == "HF_TOKEN":
                token = value.strip().strip("\"'")
    dest = ROOT / "data/rl_source_v2"
    seeds = json.loads((ROOT / "rl_materials/visual_seeds.json").read_text(encoding="utf-8"))
    ids = {row["record_id"] for row in seeds}
    metadata = {item.rfilename: item for item in HfApi(token=token).dataset_info(
        REPO, revision=REVISION, files_metadata=True).siblings}

    def matches_provider(path, info):
        if not path.is_file():
            return False
        if info.lfs:
            return digest(path) == info.lfs.sha256
        raw = path.read_bytes()
        return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == info.blob_id

    def fetch(name):
        info, local = metadata[name], dest / name
        if matches_provider(local, info):
            return local
        path = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION,
                                   token=token, local_dir=dest))
        if not matches_provider(path, info):
            raise ValueError(f"Provider checksum mismatch: {name}")
        return path

    files = set()
    for split, expected in HASHES.items():
        path = fetch(f"hf_multimodal/{split}.jsonl")
        if digest(path) != expected:
            raise ValueError(f"Frozen {split} hash mismatch")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if args.all_images or row["id"] in ids:
                for name in row["images"]:
                    if not name.startswith("images/") or not (dest / name).resolve().is_relative_to((dest / "images").resolve()):
                        raise ValueError("Non-drawing source path")
                    files.add(name)
    images = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as pool:
        for i, path in enumerate(pool.map(fetch, sorted(files)), 1):
            images.append({"path": path.relative_to(dest).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size})
            if i % 100 == 0 or i == len(files):
                print(f"Drawing download {i}/{len(files)}", flush=True)
    save_json(dest / "download_revision.json", {"repo": REPO, "revision": REVISION})
    save_json(dest / "image_integrity.json", {"scope": "train_validation" if args.all_images else "visual_seeds",
              "test_opened": False, "images": images})
    print(f"Verified {len(images)} drawing files; no test records or model weights requested.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
