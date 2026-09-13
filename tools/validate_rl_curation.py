"""Content provenance and schema audit for authored RL materials.

This reports objective errors. Independent semantic review is separate and
must bind each reviewed record's canonical hash; a schema pass is not approval.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_materials.contracts import validate_output
from tools.prepare_case_rl import digest, read_jsonl, save_json


def canonical_hash(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def normalized(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def load_rows(path):
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return json.loads(raw) if raw.startswith("[") else [json.loads(s) for s in raw.splitlines() if s.strip()]


def case_audit(lane):
    path = ROOT / f"rl_materials/curation/{lane}_cases.jsonl"
    if not path.is_file():
        return {"lane": lane, "passed": False, "errors": ["curation_file_missing"], "records": 0}
    from pypdf import PdfReader
    rows, errors, record_reports = load_rows(path), [], []
    db = sqlite3.connect(ROOT / "data/case_rl/evidence.sqlite")
    ids = []
    for row in rows:
        rid = row.get("case_id", "missing-id")
        ids.append(rid)
        issues = []
        required = ("case_id", "jurisdiction", "case_name", "docket", "court", "decision_date",
                    "pdf_path", "source_sha256", "case_patent_ids", "lineage_notes", "history_scope",
                    "evidence_spans", "principles", "advisory_episodes", "author", "review_status")
        issues += [f"missing:{field}" for field in required if field not in row]
        if issues:
            errors.extend(f"{rid}:{x}" for x in issues)
            record_reports.append({"record_id": rid, "sha256": canonical_hash(row), "errors": issues})
            continue
        if row["jurisdiction"] != lane.upper():
            issues.append("jurisdiction")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["decision_date"]):
            issues.append("decision_date_format")
        if not isinstance(row["case_patent_ids"], list) or not row["lineage_notes"]:
            issues.append("lineage_schema")
        history = row["history_scope"]
        if not isinstance(history, dict) or history.get("mode") not in {"historical_as_of_decision", "current_sources_checked"} or not history.get("limits"):
            issues.append("history_scope")
        elif history["mode"] == "historical_as_of_decision" and history.get("as_of_date") != row["decision_date"]:
            issues.append("historical_date_mismatch")
        pdf = Path(row["pdf_path"])
        page_map = {}
        if not pdf.is_file():
            issues.append("pdf_missing")
        elif digest(pdf) != row["source_sha256"]:
            issues.append("pdf_hash_mismatch")
        else:
            page_map = dict(db.execute("SELECT unit,text FROM evidence WHERE path=?", (str(pdf.resolve()),)).fetchall())
            if not page_map:
                try:
                    page_map = {i: p.extract_text() or "" for i, p in enumerate(PdfReader(pdf).pages, 1)}
                except Exception as exc:
                    issues.append(f"pdf_extract:{type(exc).__name__}")
        eids = set()
        for span in row["evidence_spans"]:
            if not all(field in span for field in ("id", "pdf_page", "quote", "speaker", "section")):
                issues.append("evidence_span_schema")
                continue
            if span["id"] in eids:
                issues.append("duplicate_evidence_id")
            eids.add(span["id"])
            page = span["pdf_page"]
            if type(page) is not int or page not in page_map:
                issues.append(f"page_missing:{span['id']}")
            elif not normalized(span["quote"]) or normalized(span["quote"]) not in normalized(page_map[page]):
                issues.append(f"quote_mismatch:{span['id']}:page{page}")
            if span["speaker"] not in {"court_holding", "court_reasoning", "party_submission", "quoted_authority", "dissent", "concurrence"}:
                issues.append(f"speaker_unrecognized:{span['id']}")
        if not row["principles"]:
            issues.append("no_principles")
        for principle in row["principles"]:
            if not all(field in principle for field in ("id", "statement", "evidence_ids", "scope", "non_rules")):
                issues.append("principle_schema")
            elif not principle["evidence_ids"] or not set(principle["evidence_ids"]).issubset(eids):
                issues.append("principle_evidence")
        episodes = row["advisory_episodes"]
        if len(episodes) < 2:
            issues.append("fewer_than_two_episodes")
        for ep in episodes:
            if not all(field in ep for field in ("episode_id", "question", "scenario_facts", "scenario_origin", "temporal_scope", "reference_answer", "required_points", "forbidden_claims", "evidence_ids", "inferior_answer", "preference_reason")):
                issues.append("episode_schema")
                continue
            if not set(ep["evidence_ids"]).issubset(eids) or not ep["evidence_ids"]:
                issues.append(f"episode_evidence:{ep['episode_id']}")
            if ep["reference_answer"] == ep["inferior_answer"]:
                issues.append("identical_preference")
            if not ep["required_points"] or not ep["forbidden_claims"] or not ep["preference_reason"]:
                issues.append("empty_reward_criteria")
        record_reports.append({"record_id": rid, "sha256": canonical_hash(row), "episodes": len(episodes),
                               "source_pages": sorted({s.get("pdf_page", 0) for s in row["evidence_spans"]}), "errors": issues})
        errors.extend(f"{rid}:{x}" for x in issues)
    db.close()
    if len(ids) != len(set(ids)):
        errors.append("duplicate_case_ids")
    return {"lane": lane, "passed": not errors, "records": len(rows),
            "episodes": sum(len(r.get("advisory_episodes", [])) for r in rows),
            "errors": errors, "record_hashes": record_reports,
            "semantic_review_passed": False, "note": "Exact-source/schema checks only; independent interpretation and relevance review still required."}


def as_output(claims, parents, abstentions=None):
    return {"claims": [{"number": i + 1, "kind": "independent" if i == 0 else "dependent",
                         "depends_on": parents[i], "text": text} for i, text in enumerate(claims)],
            "annotations": [], "abstentions": abstentions or []}


def drawing_audit(lane="drawings"):
    path = ROOT / ("rl_materials/curation/claimsets_test.jsonl" if lane == "new_test" else "rl_materials/curation/claimsets.jsonl")
    if not path.is_file():
        return {"lane": "drawings", "passed": False, "errors": ["curation_file_missing"], "records": 0}
    rows = load_rows(path)
    source = {}
    if lane == "new_test":
        source = {r["record_id"]: r for r in read_jsonl(ROOT / "data/case_rl/new_claim_test/manifest.jsonl")}
    else:
        for split in ("train", "validation"):
            source.update({r["source_record_id"]: r for r in read_jsonl(ROOT / f"data/case_rl/tasks.{split}.jsonl")})
    errors, reports = [], []
    for row in rows:
        rid, issues = row.get("record_id", "missing-id"), []
        required = ("record_id", "source_split", "images", "observations", "claims", "parents", "claim_evidence", "abstentions", "reward_required_points", "reward_forbidden_claims", "alternatives", "author", "review_status")
        issues += [f"missing:{key}" for key in required if key not in row]
        if not issues:
            task = source.get(rid)
            if not task or task["source_split"] != row["source_split"] or task["images"] != row["images"]:
                issues.append("source_split_or_image_order")
            if len(row["claims"]) != len(row["parents"]) or len(row["claim_evidence"]) != len(row["claims"]):
                issues.append("claim_parallel_array_lengths")
            else:
                issues.extend(validate_output(as_output(row["claims"], row["parents"], row["abstentions"]), {}))
            for observation in row["observations"]:
                if type(observation.get("image_index")) is not int or not 0 <= observation["image_index"] < len(row["images"]) or not observation.get("visible"):
                    issues.append("observation_image_index")
            for evidence in row["claim_evidence"]:
                if not evidence or any(type(i) is not int or not 0 <= i < len(row["observations"]) for i in evidence):
                    issues.append("claim_observation_reference")
            minimum = 2 if row["source_split"] == "train" else 1
            if len(row["alternatives"]) < minimum:
                issues.append("too_few_substantive_alternatives")
            for alt in row["alternatives"]:
                if not all(k in alt for k in ("claims", "parents", "defect", "preference_reason")):
                    issues.append("alternative_schema")
                elif alt["claims"] == row["claims"] and alt["parents"] == row["parents"]:
                    issues.append("identical_alternative")
                elif len(alt["claims"]) != len(alt["parents"]):
                    issues.append("alternative_parallel_array_lengths")
        errors.extend(f"{rid}:{x}" for x in issues)
        reports.append({"record_id": rid, "sha256": canonical_hash(row), "errors": issues})
    if len({r.get("record_id") for r in rows}) != len(rows):
        errors.append("duplicate_record_id")
    return {"lane": lane, "passed": not errors, "records": len(rows),
            "source_splits": dict(Counter(r.get("source_split") for r in rows)),
            "records_with_3plus_images": sum(len(r.get("images", [])) >= 3 for r in rows),
            "errors": errors, "record_hashes": reports, "semantic_review_passed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=["kr", "us", "drawings", "new_test"], required=True)
    args = parser.parse_args()
    report = drawing_audit(args.lane) if args.lane in {"drawings", "new_test"} else case_audit(args.lane)
    target = ROOT / f".superloopy/evidence/{args.lane}-source-schema-audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    save_json(target, report)
    print(json.dumps({k: v for k, v in report.items() if k != "record_hashes"}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
