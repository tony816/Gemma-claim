#!/usr/bin/env python3
"""Project an approved dataset release into the package layout the pipeline reads.

The release ships 66 fields per row. Three of them are model input; the rest are
provenance and oracle material, and one of them --
`canonical_source_claim_transcription` -- is byte-identical to the target.
Carrying a row through unfiltered would put the answer in the prompt. So this
converts by *whitelist*: a field not named here cannot reach the model, and the
emitted records are asserted to carry exactly {id, images, messages}.

Reads the release read-only and writes outside it. Verifies every consumed file
against the release's own manifest before using it, and refuses to emit on any
contract violation rather than dropping the offending record.

    python tools/convert_release.py --release <dir> --out <dir> [--zip]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation", "test")

# The only fields that may be read out of a release row.
INPUT_FIELDS = ("id", "split", "prompt", "visual_inputs_compact", "target_claim_clean")
# The only keys an emitted record may carry.
EMITTED_KEYS = {"id", "images", "messages"}


def die(code: str, msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"\n### HARD_FAILURE {code}\n{msg}\n", flush=True)
    sys.exit(2)


def log(msg: str) -> None:
    print(f"[convert] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                die("RELEASE_JSONL_MALFORMED", f"{path}:{lineno}: {exc}")
    return rows


def load_manifest(release: Path) -> dict[str, dict]:
    path = release / "manifests" / "file_manifest.jsonl"
    if not path.is_file():
        die("RELEASE_MANIFEST_MISSING", f"{path} does not exist.")
    return {r["path"]: r for r in read_jsonl(path)}


def verify_against_manifest(release: Path, rel: str, manifest: dict[str, dict]) -> None:
    """Hash a release file and compare it with its manifest entry. Fail closed."""
    entry = manifest.get(rel)
    if entry is None:
        die("RELEASE_FILE_UNMANIFESTED", f"{rel} is not in file_manifest.jsonl.")
    src = release / rel
    if not src.is_file():
        die("RELEASE_FILE_MISSING", f"{src} does not exist.")
    got = sha256(src)
    if got != entry["sha256"]:
        die(
            "RELEASE_FILE_HASH_MISMATCH",
            f"{rel}\n  manifest {entry['sha256']}\n  actual   {got}",
        )


def check_receipt(release: Path) -> dict:
    receipt_path = release / "run_receipt.json"
    if not receipt_path.is_file():
        die("RELEASE_RECEIPT_MISSING", f"{receipt_path} does not exist.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("full_fine_tuning_ready") is not True:
        die(
            "RELEASE_NOT_FINETUNE_READY",
            "run_receipt.json has full_fine_tuning_ready="
            f"{receipt.get('full_fine_tuning_ready')!r}; refusing to convert.",
        )
    if receipt.get("final_release_validation") != "PASS":
        die(
            "RELEASE_VALIDATION_NOT_PASS",
            f"final_release_validation={receipt.get('final_release_validation')!r}",
        )
    canonical = release / "canonical" / "all_pairs.jsonl"
    got = sha256(canonical)
    want = receipt.get("all_pairs_sha256")
    if got != want:
        die(
            "RELEASE_CANONICAL_HASH_MISMATCH",
            f"canonical/all_pairs.jsonl\n  receipt {want}\n  actual  {got}",
        )
    log(f"receipt PASS, canonical sha256 {got} verified")
    return receipt


def check_leakage_report(release: Path) -> dict:
    path = release / "validation" / "final_release_validation.json"
    if not path.is_file():
        die("RELEASE_VALIDATION_MISSING", f"{path} does not exist.")
    v = json.loads(path.read_text(encoding="utf-8"))
    zero_required = (
        "cross_split_family_leakage_count",
        "cross_split_target_exact_leakage_count",
        "cross_split_image_exact_leakage_count",
        "cross_split_pdf_exact_leakage_count",
        "korean_target_integrity_failure_count",
        "duplicate_publication_count",
        "duplicate_family_count",
        "row_gate_failure_count",
        "asset_failure_count",
    )
    nonzero = {k: v.get(k) for k in zero_required if v.get(k) not in (0, None)}
    if nonzero:
        die("RELEASE_LEAKAGE_REPORTED", f"Non-zero leakage counters: {nonzero}")
    if v.get("status") != "PASS":
        die("RELEASE_VALIDATION_NOT_PASS", f"status={v.get('status')!r}")
    log("release validation reports zero leakage on every axis")
    return v


def project(row: dict, split: str, image_map: dict[str, str]) -> dict:
    """Whitelist one release row into a model-input record."""
    rid = row.get("id")
    if not rid:
        die("RECORD_ID_MISSING", f"A {split} row has no id: keys={sorted(row)[:10]}")

    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        die("RECORD_PROMPT_EMPTY", f"{rid} has no usable prompt.")

    target = row.get("target_claim_clean")
    if not isinstance(target, str) or not target.strip():
        die("EMPTY_ASSISTANT_TARGET", f"{rid} has an empty target_claim_clean.")

    # The prompt is a fixed instruction. If the answer ever appears inside it the
    # task is trivially solvable and every metric downstream is meaningless.
    if target.strip() in prompt:
        die("TARGET_LEAKED_INTO_PROMPT", f"{rid}: target text appears in the prompt.")

    srcs = row.get("visual_inputs_compact")
    if not isinstance(srcs, list) or not srcs:
        die("RECORD_NO_IMAGES", f"{rid} has no visual_inputs_compact.")

    images = []
    for p in srcs:  # order is part of the contract; never sort, never dedupe
        if p not in image_map:
            die("IMAGE_NOT_MAPPED", f"{rid} references unmapped image {p!r}.")
        images.append(image_map[p])

    rec = {
        "id": rid,
        "images": images,
        # Image placeholders are deliberately absent: common.normalise_record
        # materialises one per image at the head of the user turn, in array
        # order. That is the path the previous run exercised; preflight then
        # asserts placeholder count == image count.
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
    }
    if set(rec) != EMITTED_KEYS:
        die("EMITTED_KEYS_UNEXPECTED", f"{rid} emitted keys {sorted(rec)}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=os.environ.get("RELEASE_DIR", ""),
                    help="approved release directory (read-only)")
    ap.add_argument("--out", required=True,
                    help="output package root (must not be inside --release)")
    ap.add_argument("--zip", action="store_true",
                    help="also emit a ZIP and print its SHA-256")
    ap.add_argument("--skip-image-hashes", action="store_true",
                    help="skip per-image manifest verification (faster, weaker)")
    args = ap.parse_args()

    if not args.release:
        die("RELEASE_DIR_UNSET", "Pass --release or set RELEASE_DIR.")
    release = Path(args.release).resolve()
    out = Path(args.out).resolve()
    if not release.is_dir():
        die("RELEASE_DIR_MISSING", f"{release} is not a directory.")
    if out == release or release in out.parents:
        die("OUTPUT_INSIDE_RELEASE",
            f"Refusing to write inside the frozen release.\n"
            f"  release {release}\n  out     {out}")

    receipt = check_receipt(release)
    validation = check_leakage_report(release)
    manifest = load_manifest(release)

    # ---- read the split ledgers -------------------------------------------
    raw: dict[str, list[dict]] = {}
    for split in SPLITS:
        rel = f"splits/{split}.jsonl"
        verify_against_manifest(release, rel, manifest)
        raw[split] = read_jsonl(release / rel)
        log(f"{split}: {len(raw[split])} rows, manifest-verified")

    counts = {s: len(raw[s]) for s in SPLITS}
    if counts != receipt["split_counts"]:
        die("SPLIT_COUNTS_DISAGREE",
            f"ledger {counts} != receipt {receipt['split_counts']}")

    # the split field must agree with the file the row came from
    mismatched = [r["id"] for s in SPLITS for r in raw[s] if r.get("split") != s]
    if mismatched:
        die("SPLIT_FIELD_MISMATCH", f"{len(mismatched)} rows: {mismatched[:10]}")

    # ids unique within a split and disjoint across splits
    ids = {s: [r["id"] for r in raw[s]] for s in SPLITS}
    dupes = [i for s in SPLITS for i, c in Counter(ids[s]).items() if c > 1]
    if dupes:
        die("DUPLICATE_RECORD_ID", f"{dupes[:10]}")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(ids[a]) & set(ids[b])
        if overlap:
            die("TEST_LEAKAGE", f"{a}|{b} share ids: {sorted(overlap)[:10]}")

    # ---- collect and verify every referenced image ------------------------
    referenced: list[str] = []
    seen: set[str] = set()
    for s in SPLITS:
        for r in raw[s]:
            for p in (r.get("visual_inputs_compact") or []):
                if p not in seen:
                    seen.add(p)
                    referenced.append(p)
    log(f"{len(referenced)} unique images referenced")

    image_map: dict[str, str] = {}
    for i, rel in enumerate(referenced, 1):
        if not rel.startswith("artifacts/images/"):
            die("IMAGE_PATH_UNEXPECTED", f"{rel!r} is not under artifacts/images/.")
        if args.skip_image_hashes:
            if not (release / rel).is_file():
                die("MISSING_IMAGE", f"{release / rel} does not exist.")
        else:
            verify_against_manifest(release, rel, manifest)
        image_map[rel] = "images/" + rel[len("artifacts/images/"):]
        if i % 500 == 0:
            log(f"  verified {i}/{len(referenced)}")
    log(f"all {len(referenced)} images accounted for")

    # ---- project ----------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "hf_multimodal").mkdir(parents=True)

    emitted = {s: [project(r, s, image_map) for r in raw[s]] for s in SPLITS}

    # ---- copy images ------------------------------------------------------
    copied = 0
    for src_rel, dst_rel in image_map.items():
        dst = out / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(release / src_rel, dst)
        copied += 1
        if copied % 500 == 0:
            log(f"  copied {copied}/{len(image_map)}")
    log(f"copied {copied} images into {out / 'images'}")

    # ---- write ledgers ----------------------------------------------------
    ledger_hashes = {}
    for s in SPLITS:
        path = out / "hf_multimodal" / f"{s}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in emitted[s]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ledger_hashes[s] = sha256(path)
        log(f"wrote {path.name}: {len(emitted[s])} records, "
            f"sha256 {ledger_hashes[s][:16]}...")

    # ---- post-write assertions --------------------------------------------
    for s in SPLITS:
        for rec in read_jsonl(out / "hf_multimodal" / f"{s}.jsonl"):
            if set(rec) != EMITTED_KEYS:
                die("EMITTED_KEYS_UNEXPECTED", f"{rec.get('id')}: {sorted(rec)}")
            for p in rec["images"]:
                if not (out / p).is_file():
                    die("MISSING_IMAGE", f"{rec['id']} -> {out / p}")
    log("post-write assertions passed")

    # ---- manifest ---------------------------------------------------------
    img_per_rec = Counter(len(r["images"]) for s in SPLITS for r in emitted[s])
    tgt = [r["messages"][-1]["content"] for s in SPLITS for r in emitted[s]]
    conv = {
        "source_release_id": receipt.get("release_id"),
        "source_release_dir": str(release),
        "source_all_pairs_sha256": receipt.get("all_pairs_sha256"),
        "source_validation_status": validation.get("status"),
        "input_fields_whitelisted": list(INPUT_FIELDS),
        "emitted_keys": sorted(EMITTED_KEYS),
        "counts": counts,
        "unique_images": len(image_map),
        "images_per_record": dict(sorted(img_per_rec.items())),
        "target_chars": {"min": min(map(len, tgt)), "max": max(map(len, tgt))},
        "ledger_sha256": ledger_hashes,
    }
    (out / "conversion_manifest.json").write_text(
        json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {out / 'conversion_manifest.json'}")

    # ---- optional zip -----------------------------------------------------
    if args.zip:
        zpath = out.parent / f"{out.name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    arc = str(Path(out.name) / p.relative_to(out)).replace("\\", "/")
                    zf.write(p, arc)
        log(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
        print(f"\nZIP_SHA256 {sha256(zpath)}\nZIP_PATH   {zpath}")

    print("\nCONVERSION_PASS")


if __name__ == "__main__":
    main()
