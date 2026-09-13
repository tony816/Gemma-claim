"""Read patent-family metadata for leakage audits, never model input/targets.

Only family/priority/publication metadata is retained. Original claims and
specification text in the HTTP response are discarded, never sent to an LLM.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/case_rl/families"


def read_token():
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            if key.strip() == "HF_TOKEN":
                return value.strip().strip("\"'")
    return None


def protected_ids() -> list[str]:
    from huggingface_hub import HfApi
    info = HfApi(token=read_token()).dataset_info(
        "Mepeng22/gemma-claim-v2-approved-20260902",
        revision="e978f526d0b15f6c981a4b82fd25404cef68a8d7")
    ids = sorted({p.rfilename.split("/")[2] for p in info.siblings
                  if p.rfilename.startswith("images/") and len(p.rfilename.split("/")) == 4})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protected_ids.json").write_text(json.dumps({"patent_ids": ids,
        "origin": "pinned_Hub_image_path_metadata_only", "test_content_read": False}, indent=2), encoding="utf-8")
    return ids


def metadata_from_html(patent_id: str, text: str) -> dict:
    fragments = []
    family = re.search(r'<section itemprop="family"[^>]*>(.*?)</section>', text, re.S)
    family_id = re.search(r'<h2>\s*ID=(\d+)\s*</h2>', family.group() if family else text)
    if family_id:
        fragments.append(family_id.group())
    # The broad family section can also contain patent-citation tables.
    # Those are prior-art references, NOT family members. Select named rows.
    for prop in ("applications", "countryStatus", "docdbFamily", "pubs", "priorityApps",
                 "appsClaimingPriority", "parentApps", "childApps", "beforeApplications", "afterApplications"):
        fragments.extend(re.findall(r'<tr itemprop="' + prop + r'"[^>]*>.*?</tr>', text, re.S))
    joined = "\n".join(fragments)
    publications = sorted({patent_id, *re.findall(r'/patent/([A-Z]{2}(?:RE)?\d+(?:[A-Z]\d?)?)(?:/|\")', joined)})
    priorities = sorted(set(re.findall(r'<[^>]+itemprop="priorityDate"[^>]*>\s*(\d{4}-\d{2}-\d{2})', joined)))
    galleries = re.findall(r'<meta itemprop="full" content="([^"]+)"', text)
    title = re.search(r'<meta name="DC.title" content="([^"]+)"', text)
    application_numbers = sorted(set(html.unescape(s).strip() for s in re.findall(
        r'<[^>]+itemprop="applicationNumber"[^>]*>(.*?)</[^>]+>', joined, re.S)))
    return {"patent_id": patent_id, "metadata_parser_version": 3, "source_url": f"https://patents.google.com/patent/{patent_id}/en",
            "status": "metadata_extracted" if family_id else "family_id_missing",
            "family_id": family_id.group(1) if family_id else None,
            "related_publications": publications, "priority_dates": priorities,
            "application_numbers": application_numbers,
            "title": html.unescape(title.group(1)) if title else None,
            "drawing_urls": galleries, "metadata_html": joined,
            "metadata_sha256": hashlib.sha256(joined.encode()).hexdigest(),
            "source_response_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "limits": "Google/DOCDB metadata, not a certified exhaustive legal family search; unresolved related applications require review.",
            "original_claims_or_description_retained": False}


def fetch(patent_id: str, refresh=False) -> dict:
    if not re.fullmatch(r"[A-Z]{2}(?:RE)?\d+(?:[A-Z]\d?)?", patent_id):
        raise ValueError("Invalid publication ID")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{patent_id}.json"
    if path.exists() and not refresh:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("status") == "metadata_extracted" and previous.get("metadata_parser_version") == 3:
            return previous
    error = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"https://patents.google.com/patent/{patent_id}/en",
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as response:
                value = metadata_from_html(patent_id, response.read().decode("utf-8"))
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            return value
        except Exception as exc:
            error = type(exc).__name__
            if attempt < 2:
                time.sleep(1 + attempt)
    value = {"patent_id": patent_id, "status": "fetch_failed", "error_type": error}
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protected", action="store_true")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    ids = sorted(set((protected_ids() if args.protected else []) + args.ids))
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(12, args.workers))) as pool:
        for index, row in enumerate(pool.map(fetch, ids), 1):
            results.append({key: row.get(key) for key in ("patent_id", "status", "family_id")})
            if index % 50 == 0 or index == len(ids):
                print(f"Family metadata {index}/{len(ids)}; extracted={sum(r['status']=='metadata_extracted' for r in results)}", flush=True)
    (OUT / "latest_fetch_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
