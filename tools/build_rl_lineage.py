"""Build fail-closed patent/case lineage graph from retained bibliography.

Edges come from family, priority, continuation and publication metadata; never
from citations or title similarity. This is a data-leakage check, not a legal
certification of the complete worldwide family.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_rl_curation import load_rows, canonical_hash


def publication_key(value):
    value = re.sub(r"[^A-Z0-9]", "", value.upper())
    match = re.fullmatch(r"([A-Z]{2}(?:RE)?\d+)(?:[A-Z]\d?)?", value)
    return "pub:" + (match.group(1) if match else value)


def application_key(value):
    # Keep country/series/year. Do not join an application to a publication
    # merely because their number strings happen to be equal.
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    # DOCDB US application aliases add a four-digit year and kind to the
    # eight-digit series/serial number; preserve the actual series/serial.
    docdb_us = re.fullmatch(r"US(?:19|20)\d{2}(\d{8})[AP]", compact)
    if docdb_us:
        compact = "US" + docdb_us.group(1)
    return "app:" + compact


class Graph:
    def __init__(self):
        self.parents = {}

    def find(self, key):
        self.parents.setdefault(key, key)
        while key != self.parents[key]:
            self.parents[key] = self.parents[self.parents[key]]
            key = self.parents[key]
        return key

    def join(self, values):
        values = list(values)
        if not values:
            return
        head = self.find(values[0])
        for key in values[1:]:
            other = self.find(key)
            if head != other:
                self.parents[other] = head

    def same(self, a, b):
        return self.find(a) == self.find(b)


def case_lookup_ids(row, overrides):
    override = overrides.get(row["case_id"])
    if override:
        return [override["publication_id"]]
    ids = []
    for entry in row["case_patent_ids"]:
        if isinstance(entry, str):
            ids.append(re.sub(r"[^A-Z0-9]", "", entry.upper()))
        else:
            digits = re.sub(r"\D", "", entry["id"])
            utility = "실용신안" in entry.get("provenance", {}).get("quote", "")
            ids.append("KR" + (("20" if utility else "10") + digits.zfill(7) if len(digits) <= 7 else digits))
    return ids


def build():
    directory = ROOT / "data/case_rl/families"
    overrides_path = ROOT / "rl_materials/curation/family_overrides.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    graph, known, sources, errors = Graph(), set(), [], []
    metadata = []
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(row, dict) and row.get("status") == "metadata_extracted" and row.get("metadata_parser_version") == 3:
            metadata.append(row)
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for row in overrides["records"]:
        source = ROOT / row["source_path"]
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != row["source_sha256"]:
            errors.append("official_source_integrity:" + row["patent_id"])
            continue
        metadata.append(row)
    for row in metadata:
        pid = row["patent_id"]
        values = [publication_key(pid)] + [publication_key(x) for x in row.get("related_publications", [])]
        values += [application_key(x) for x in row.get("application_numbers", [])]
        if row.get("family_id"):
            values.append("docdb:" + str(row["family_id"]))
        graph.join(values)
        known.add(pid)
    protected = json.loads((directory / "protected_ids.json").read_text())["patent_ids"]
    unresolved = [pid for pid in protected if pid not in known]
    errors += ["protected_metadata_missing:" + pid for pid in unresolved]
    cases = [r for lane in ("kr", "us") for r in load_rows(ROOT / f"rl_materials/curation/{lane}_cases.jsonl")]
    case_info = []
    for row in cases:
        ids = case_lookup_ids(row, overrides["case_identifier_resolutions"])
        missing = [pid for pid in ids if pid not in known]
        if missing or not ids:
            errors.append("case_metadata_missing:" + row["case_id"])
        # One litigation containing multiple patents keeps their linked lineages
        # in one split. These are episode-grouping edges, not DOCDB assertions.
        graph.join(["case:" + row["case_id"]] + [publication_key(x) for x in ids])
        case_info.append({"case_id": row["case_id"], "lookup_ids": ids, "missing_metadata": missing,
                          "record_sha256": canonical_hash(row)})
    # Resolve after all edges have been joined, including cases asserting several patents.
    for row in case_info:
        row["lineage_key"] = graph.find("case:" + row["case_id"])
        row["protected_overlap"] = [pid for pid in protected if graph.same("case:" + row["case_id"], publication_key(pid))]
    new_tasks = load_rows(ROOT / "data/case_rl/new_claim_test/manifest.jsonl")
    new_info = []
    for task in new_tasks:
        pid = task["source_patent_id"]
        overlap = [p for p in protected if graph.same(publication_key(pid), publication_key(p))]
        case_overlap = [r["case_id"] for r in cases if graph.same(publication_key(pid), "case:" + r["case_id"])]
        if pid not in known or overlap or case_overlap:
            errors.append("new_test_overlap_or_unknown:" + pid)
        new_info.append({"record_id": task["record_id"], "patent_id": pid, "lineage_key": graph.find(publication_key(pid)),
                         "protected_overlap": overlap, "case_overlap": case_overlap})
    if len({r["lineage_key"] for r in new_info}) != len(new_info):
        errors.append("new_test_internal_family_overlap")
    frozen_train = {r['patent_id'] for r in load_rows(ROOT/'data/case_rl/tasks.train.jsonl')}
    frozen_validation = {r['patent_id'] for r in load_rows(ROOT/'data/case_rl/tasks.validation.jsonl')}
    frozen_test = set(protected)-frozen_train-frozen_validation
    selected_drawings=[]
    for row in load_rows(ROOT/'rl_materials/curation/claimsets.jsonl'):
        pid=row['record_id'].split('_claim')[0]
        test_overlap=sorted(p for p in frozen_test if graph.same(publication_key(pid),publication_key(p)))
        train_overlap=sorted(p for p in frozen_train if graph.same(publication_key(pid),publication_key(p))) if row['source_split']=='validation' else []
        if test_overlap or train_overlap:
            errors.append('drawing_split_family_overlap:'+row['record_id'])
        selected_drawings.append({'record_id':row['record_id'],'patent_id':pid,'source_split':row['source_split'],
            'record_sha256':canonical_hash(row),'lineage_key':graph.find(publication_key(pid)),
            'old_test_family_overlap':test_overlap,'validation_to_train_overlap':train_overlap})
    for info in case_info:
        info['old_test_overlap']=sorted(set(info['protected_overlap'])&frozen_test)
    report = {"passed": not errors, "errors": errors, "protected_patents": len(protected),
              "protected_metadata_unresolved": unresolved, "case_lineages": case_info, "new_test": new_info,
              "selected_drawings": selected_drawings, "old_test_patent_count":len(frozen_test),
              "metadata_sources": sources, "overrides_sha256": hashlib.sha256(overrides_path.read_bytes()).hexdigest(),
              "semantic_review_passed": False,
              "limits": "Recorded family/priority/application metadata, plus visually transcribed official bibliographic exceptions; independent review required. Not an exhaustive certified family search."}
    path = ROOT / "data/case_rl/lineage_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"metadata_sources", "case_lineages", "new_test", "selected_drawings"}}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = build()
    raise SystemExit(0 if result["passed"] else 1)
