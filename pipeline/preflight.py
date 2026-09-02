"""Data-contract preflight. Nothing trains until this passes.

Enforces the frozen-dataset contract: split sizes, image existence and
ordering, non-empty assistant targets, no oracle/leakage material reachable
from a training record, and family-disjoint splits.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    FORBIDDEN_SUBDIRS,
    OUT,
    SPLITS,
    Record,
    _count_images,
    die,
    find_package_root,
    load_all,
    log,
    write_json,
)

EXPECTED_COUNTS = {"train": 554, "validation": 65, "test": 75}

# A converted record may carry these keys and no others. The approved release
# ships 66 fields per row, and several of them are the answer or a route to it:
# `canonical_source_claim_transcription` is byte-identical to the target, and
# the oracle / evidence / boundary fields describe how the target was derived.
# tools/convert_release.py projects a row down to three keys; this re-checks
# the projection at the point of use, so a package assembled any other way
# cannot smuggle oracle material into model input.
ALLOWED_RECORD_KEYS = {"id", "images", "messages"}
FORBIDDEN_KEY_MARKERS = (
    "oracle", "canonical", "evidence", "boundary", "transcription",
    "source_", "semantic", "family", "publication", "target_claim",
)


def run_packaged_validators(pkg: Path) -> dict:
    """Run the validators shipped inside the package, if present.

    --verify-source-pdfs is deliberately not passed: the source PDFs are not
    present on the training host and the flag would fail for the wrong reason.
    """
    results = {}
    for name in ("validate_dataset.py", "smoke_test.py"):
        script = pkg / "scripts" / name
        if not script.is_file():
            results[name] = {"present": False}
            continue
        p = subprocess.run(
            [sys.executable, str(script)], cwd=str(pkg),
            capture_output=True, text=True, timeout=1800,
        )
        tail = (p.stdout + p.stderr)[-4000:]
        results[name] = {"present": True, "returncode": p.returncode, "output_tail": tail}
        log(f"--- {name} exit={p.returncode} ---\n{tail}")
    return results


def check_images(recs: list[Record], pkg: Path, report: dict) -> None:
    missing, order_mismatch = [], []
    for rec in recs:
        for p in rec.images:
            if not p.is_file():
                missing.append({"record": rec.record_id, "path": str(p)})
        placeholders = sum(_count_images(m["content"]) for m in rec.messages)
        # Every referenced image must have exactly one placeholder, in order.
        if placeholders != len(rec.images):
            order_mismatch.append(
                {"record": rec.record_id, "placeholders": placeholders,
                 "images": len(rec.images)}
            )
    report["missing_images"] = missing
    report["image_placeholder_mismatch"] = order_mismatch
    report["records_with_injected_placeholders"] = sum(
        1 for r in recs if r.placeholders_injected
    )
    report["total_images_referenced"] = sum(len(r.images) for r in recs)
    if missing:
        die("MISSING_IMAGE", f"{len(missing)} referenced image(s) absent: {missing[:10]}")
    if order_mismatch:
        die(
            "IMAGE_ORDER_CONTRACT_VIOLATION",
            "Image placeholder count != images[] length for: "
            f"{order_mismatch[:10]}. The array order is part of the data contract "
            "and images may not be dropped or reordered.",
        )


def check_targets(recs: list[Record], report: dict) -> None:
    empties = [r.record_id for r in recs if not r.target.strip()]
    report["empty_targets"] = empties
    if empties:
        die("EMPTY_ASSISTANT_TARGET", f"Empty assistant target(s): {empties[:20]}")


def check_leakage(recs: list[Record], report: dict) -> None:
    """No training input may reference oracle / canonical / evaluation material."""
    pat = re.compile("|".join(re.escape(s) for s in FORBIDDEN_SUBDIRS))
    hits = []
    for rec in recs:
        for p in rec.images:
            if pat.search(str(p)):
                hits.append({"record": rec.record_id, "path": str(p)})
        for m in rec.messages:
            for part in m["content"]:
                if part.get("type") == "text" and pat.search(part.get("text", "")):
                    hits.append({"record": rec.record_id, "where": "message_text"})
    report["forbidden_references"] = hits
    if hits:
        die(
            "SOURCE_ORACLE_MATERIAL_IN_MODEL_INPUT",
            f"Training inputs reference forbidden material: {hits[:10]}",
        )


def check_record_keys(recs: list[Record], report: dict) -> None:
    """No record may carry a field outside the model-input whitelist.

    check_leakage above looks for oracle *paths and text*. This looks for oracle
    *fields*, which is the shape the approved release actually delivers: the
    answer travels as a sibling key of the input, not as a path.
    """
    offenders = []
    for rec in recs:
        extra = set(rec.raw_keys) - ALLOWED_RECORD_KEYS
        if extra:
            marked = sorted(
                k for k in extra
                if any(m in k.lower() for m in FORBIDDEN_KEY_MARKERS)
            )
            offenders.append({
                "record": rec.record_id,
                "unexpected_keys": sorted(extra)[:20],
                "oracle_shaped": marked[:20],
            })
    report["record_key_violations"] = offenders[:20]
    report["record_key_violation_count"] = len(offenders)
    if offenders:
        die(
            "ORACLE_FIELD_IN_MODEL_INPUT",
            f"{len(offenders)} record(s) carry fields outside "
            f"{sorted(ALLOWED_RECORD_KEYS)}. Convert the release with "
            f"tools/convert_release.py instead of loading it directly.\n"
            f"First offenders: {offenders[:3]}",
        )
    log(f"record key whitelist clean: every record carries exactly "
        f"{sorted(ALLOWED_RECORD_KEYS)}")


def family_of(rec: Record) -> str:
    """Best-effort family key, used only to assert split disjointness."""
    rid = rec.record_id
    for sep in ("__", "::", "#"):
        if sep in rid:
            return rid.split(sep)[0]
    m = re.match(r"^([A-Za-z]{2}\d{4,})", rid)
    return m.group(1) if m else rid


def check_splits(data: dict[str, list[Record]], report: dict) -> None:
    counts = {s: len(data[s]) for s in SPLITS}
    report["record_counts"] = counts
    report["expected_counts"] = EXPECTED_COUNTS
    if counts != EXPECTED_COUNTS:
        die(
            "DATASET_SPLIT_COUNTS_UNEXPECTED",
            f"Got {counts}, expected {EXPECTED_COUNTS}. The split is frozen and "
            "must not be re-partitioned.",
        )

    fams = {s: {family_of(r) for r in data[s]} for s in SPLITS}
    overlaps = {
        f"{a}|{b}": sorted(fams[a] & fams[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        if fams[a] & fams[b]
    }
    report["family_overlaps"] = overlaps
    ids = {s: {r.record_id for r in data[s]} for s in SPLITS}
    id_overlaps = {
        f"{a}|{b}": sorted(ids[a] & ids[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        if ids[a] & ids[b]
    }
    report["record_id_overlaps"] = id_overlaps
    if id_overlaps:
        die("TEST_LEAKAGE", f"Records appear in more than one split: {id_overlaps}")
    if overlaps:
        # Family keys are heuristic, so this is reported loudly but is not fatal
        # on its own; exact record overlap above is the hard gate.
        log(f"WARNING: heuristic family key overlaps across splits: {overlaps}")


def main() -> None:
    pkg = find_package_root()
    data = load_all(pkg)
    report: dict = {
        "package_root": str(pkg),
        "dataset_version": os.environ.get("DATASET_VERSION", "finetune_multilingual_approved_20260902"),
        "observed_schema": {},
    }

    for split in SPLITS:
        recs = data[split]
        if recs:
            r0 = recs[0]
            report["observed_schema"][split] = {
                "raw_top_level_keys": r0.raw_keys,
                "n_messages": len(r0.messages),
                "roles": [m["role"] for m in r0.messages],
                "n_images": len(r0.images),
                "images_per_record": Counter(len(r.images) for r in recs),
                "target_chars": {
                    "min": min(len(r.target) for r in recs),
                    "max": max(len(r.target) for r in recs),
                },
            }
    report["observed_schema"] = {
        k: {kk: (dict(vv) if isinstance(vv, Counter) else vv) for kk, vv in v.items()}
        for k, v in report["observed_schema"].items()
    }
    log(f"Observed schema: {report['observed_schema']}")

    check_splits(data, report)
    all_recs = [r for s in SPLITS for r in data[s]]
    check_images(all_recs, pkg, report)
    check_targets(all_recs, report)
    check_leakage(all_recs, report)
    check_record_keys(all_recs, report)

    report["packaged_validators"] = run_packaged_validators(pkg)
    pv = report["packaged_validators"]
    failed = [
        n for n, v in pv.items()
        if v.get("present") and v.get("returncode") not in (0, None)
    ]
    report["packaged_validators_failed"] = failed

    report["DATASET_VALIDATOR_PASS"] = not failed
    report["MULTI_IMAGE_CONTRACT_PASS"] = (
        not report["missing_images"] and not report["image_placeholder_mismatch"]
    )
    write_json(OUT / "dataset_preflight.json", report)

    if failed:
        die(
            "PACKAGED_VALIDATOR_FAILED",
            f"The dataset's own validators failed: {failed}. See "
            f"{OUT / 'dataset_preflight.json'} for their output.",
        )
    log("DATASET_VALIDATOR_PASS=true MULTI_IMAGE_CONTRACT_PASS=true")


if __name__ == "__main__":
    main()
