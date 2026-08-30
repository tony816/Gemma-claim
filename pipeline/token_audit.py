"""Measure every assistant target with the real tokenizer.

A configuration that truncates even one target is not allowed, so the audit
runs before training and records the max lengths the training config must
accommodate.
"""
from __future__ import annotations

import os

from common import OUT, SPLITS, die, find_package_root, load_all, log, write_json

MODEL_ID = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
REVISION = os.environ.get("BASE_MODEL_REVISION", "main")


def main() -> None:
    from transformers import AutoProcessor

    from dataset import encode_record

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    pkg = find_package_root()
    data = load_all(pkg)

    model_max = int(
        os.environ.get("MODEL_MAX_POSITION", "262144")
    )

    per_record, per_split = [], {}
    for split in SPLITS:
        lens = []
        for rec in data[split]:
            enc = encode_record(processor, rec, with_labels=True)
            row = {
                "split": split,
                "record_id": enc["_record_id"],
                "n_images": enc["_n_images"],
                "prompt_tokens": enc["_prompt_len"],
                "target_tokens": enc["_target_len"],
                "total_tokens": enc["_total_len"],
            }
            per_record.append(row)
            lens.append(row)
        per_split[split] = {
            "n": len(lens),
            "max_total_tokens": max((r["total_tokens"] for r in lens), default=0),
            "max_target_tokens": max((r["target_tokens"] for r in lens), default=0),
            "max_prompt_tokens": max((r["prompt_tokens"] for r in lens), default=0),
            "mean_total_tokens": (
                round(sum(r["total_tokens"] for r in lens) / len(lens), 1) if lens else 0
            ),
        }
        log(f"{split}: {per_split[split]}")

    max_total = max(r["total_tokens"] for r in per_record)
    max_target = max(r["target_tokens"] for r in per_record)
    empty = [r["record_id"] for r in per_record if r["target_tokens"] <= 0]

    # No truncation is applied anywhere in this pipeline; this asserts the
    # model context is large enough that none would ever be required.
    over = [r["record_id"] for r in per_record if r["total_tokens"] > model_max]

    report = {
        "base_model": MODEL_ID,
        "revision": REVISION,
        "model_max_position_embeddings": model_max,
        "max_total_tokens": max_total,
        "max_target_tokens": max_target,
        "truncation_applied_anywhere": False,
        "records_over_context": over,
        "empty_targets": empty,
        "per_split": per_split,
        "per_record": per_record,
        "recommended_max_new_tokens": int(max_target * 1.25) + 64,
        "ZERO_TARGET_TRUNCATION_PASS": not over and not empty,
    }
    write_json(OUT / "token_length_audit.json", report)

    if empty:
        die("EMPTY_ASSISTANT_TARGET", f"Zero-length targets: {empty}")
    if over:
        die(
            "TARGET_TRUNCATION_RISK",
            f"{len(over)} record(s) exceed the {model_max}-token context: {over[:10]}",
        )
    log(
        f"ZERO_TARGET_TRUNCATION_PASS=true max_total={max_total} "
        f"max_target={max_target} rec_max_new_tokens={report['recommended_max_new_tokens']}"
    )


if __name__ == "__main__":
    main()
