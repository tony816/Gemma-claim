"""Audit the private work pack; a successful integrity audit is not GPU approval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_materials.contracts import eligible_cards, validate_output
from tools.prepare_case_rl import digest, read_jsonl, save_json


def main():
    out, source = ROOT / "data/case_rl", ROOT / "data/rl_source_v2"
    errors, checks = [], {}
    provenance = json.loads((source / "image_integrity.json").read_text(encoding="utf-8"))
    images = {row["path"]: row for row in provenance["images"]}
    for name, row in images.items():
        path = (source / name).resolve()
        if not path.is_relative_to((source / "images").resolve()) or not path.is_file():
            errors.append(f"missing_or_forbidden_image:{name}")
        elif digest(path) != row["sha256"]:
            errors.append(f"image_hash_mismatch:{name}")
    checks["local_drawing_hashes_checked"] = len(images)
    checks["image_scope"] = provenance["scope"]
    ids = {}
    for split in ("train", "validation"):
        tasks = read_jsonl(out / f"tasks.{split}.jsonl")
        ids[split] = {task["patent_id"] for task in tasks}
        for task in tasks:
            if task["source_split"] != split or task["training_eligible"] is not False:
                errors.append(f"task_status_or_split:{task['task_id']}")
            if task["case_cards"] or task["claimset_target"] is not None:
                errors.append(f"unreviewed_context:{task['task_id']}")
            if any(message["role"] != "user" for message in task["prompt"]):
                errors.append(f"answer_in_prompt:{task['task_id']}")
            for image in task["images"]:
                if image not in images:
                    errors.append(f"drawing_not_verified:{image}")
        checks[f"{split}_tasks"] = len(tasks)
    if ids["train"] & ids["validation"]:
        errors.append("exact_patent_id_overlap")
    frozen = json.loads((out / "source_split_integrity.json").read_text(encoding="utf-8"))
    for row in frozen["sources"]:
        if digest(Path(row["path"])) != row["sha256"]:
            errors.append(f"frozen_hash:{row['split']}")
    cards = read_jsonl(out / "principle_cards.draft.jsonl")
    db = sqlite3.connect(out / "evidence.sqlite")
    for card in cards:
        if digest(Path(card["pdf_path"])) != card["source_sha256"]:
            errors.append(f"case_source_hash:{card['card_id']}")
        for evidence in card["evidence"]:
            fetched = db.execute("SELECT text FROM evidence WHERE path=? AND unit=?",
                                 (card["pdf_path"], evidence["pdf_page"])).fetchone()
            if not fetched or fetched[0] != evidence["page_text"] or hashlib.sha256(fetched[0].encode()).hexdigest() != evidence["page_text_sha256"]:
                errors.append(f"case_page_mismatch:{card['card_id']}")
    db.close()
    if eligible_cards(cards, "KR", {"unrelated-test-patent"}):
        errors.append("draft_card_became_model_context")
    checks["source_verified_draft_cards"] = len(cards)
    drafts = read_jsonl(out / "claimsets.visual_drafts.jsonl")
    for draft in drafts:
        for error in validate_output(draft["claimset_target"], {}):
            errors.append(f"draft:{draft['source_record_id']}:{error}")
        if draft["source_split"] != "train" or draft["training_eligible"] is not False:
            errors.append("draft_status_or_split")
        for image in draft["images"]:
            if image["path"] not in images or image["sha256"] != images[image["path"]]["sha256"]:
                errors.append("draft_image_hash")
        for observation in draft["drawing_observations"]:
            if not 0 <= observation["image_index"] < len(draft["images"]):
                errors.append("draft_image_index")
        for indices in draft["claim_evidence"]:
            if not indices or any(not 0 <= i < len(draft["drawing_observations"]) for i in indices):
                errors.append("draft_claim_evidence_index")
    checks["structurally_checked_visual_drafts"] = len(drafts)
    pairs = read_jsonl(out / "preferences.controlled_candidates.jsonl")
    for pair in pairs:
        if pair["training_eligible"] is not False or pair["chosen"] == pair["rejected"]:
            errors.append(f"invalid_candidate:{pair['pair_id']}")
        rejected_errors = validate_output(pair["rejected"], {})
        if pair["pair_id"].endswith("unsupplied-citation") and "annotation_unapproved_or_unsupplied_card" not in rejected_errors:
            errors.append("fabricated_citation_not_detected")
        # The unsupported-material example intentionally has valid JSON and
        # dependencies. It proves why structural checks cannot supply reward.
        if pair["pair_id"].endswith("unsupported-material") and rejected_errors:
            errors.append("semantic_negative_unintentionally_has_structural_shortcut")
    checks["controlled_preference_candidates"] = len(pairs)
    report = {"integrity_passed": not errors, "checks": checks, "errors": errors,
              "training_ready": False, "gpu_allowed": False,
              "note": "Source integrity and structural checks only; reviewed gold, semantic reward calibration and trainer rehearsal remain incomplete."}
    save_json(out / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
