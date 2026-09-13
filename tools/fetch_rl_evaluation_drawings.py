"""Freeze new drawing-only evaluation inputs; does not read patent claims.

The author sees only the manifest and these images. Family/title bibliographic
metadata stays in the separate audit directory and is not a policy prompt.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
IDS = ["US20240001357A1", "US20250196124A1", "US20240131511A1",
       "US20250091047A1", "US20240066518A1", "US20250161939A1",
       "US20250012704A1", "US11998906B2"]
OUT = ROOT / "data/case_rl/new_claim_test"


def one(pid):
    from PIL import Image
    meta = json.loads((ROOT / f"data/case_rl/families/{pid}.json").read_text(encoding="utf-8"))
    if meta["status"] != "metadata_extracted":
        raise ValueError(f"Missing family metadata: {pid}")
    urls = list(dict.fromkeys(meta["drawing_urls"]))
    paths, evidence = [], []
    for i, url in enumerate(urls[:8]):
        if not url.startswith("https://patentimages.storage.googleapis.com/"):
            raise ValueError("Unapproved drawing source")
        path = OUT / "images" / pid / f"{i:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as response:
                blob = response.read()
            path.write_bytes(blob)
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            size = list(im.size)
        rel = path.relative_to(ROOT).as_posix()
        paths.append(rel)
        evidence.append({"path": rel, "source_url": url,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": size})
    if len(paths) < 1:
        raise ValueError(f"No drawings: {pid}")
    task = {"record_id": f"NEW-{pid}-claimset", "source_patent_id": pid,
            "source_split": "new_test", "images": paths, "output_language": "ko",
            "jurisdiction": "KR", "max_dependent_claims": 3,
            "selection": "first_up_to_8_distinct_gallery_URLs_in_source_order",
            "source_gallery_count": len(urls), "images_inspected_by_author": False,
            "original_claims_or_description_in_input": False}
    return task, {"record_id": task["record_id"], "images": evidence,
                  "family_metadata_sha256": hashlib.sha256((ROOT / f"data/case_rl/families/{pid}.json").read_bytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, IDS))
    for name, values in (("manifest.jsonl", [r[0] for r in rows]), ("provenance.jsonl", [r[1] for r in rows])):
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf-8")
    print(json.dumps({"records": len(rows), "images": sum(len(r[0]["images"]) for r in rows),
                      "test_labels_authored": False, "family_disjointness_approved": False}))


if __name__ == "__main__":
    main()
