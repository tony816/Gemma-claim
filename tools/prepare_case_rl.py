"""Build a local, resumable evidence/annotation work pack without any GPU call.

Raw source text is audit-only. Only reviewed principle cards may be retrieved.
This tool never manufactures dependent-claim targets or a passing reward score.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def document_info(path: Path, source: Path) -> dict:
    rel = path.relative_to(source).as_posix()
    name = path.stem
    jurisdiction = "US" if "미국 특허 판례 원본 PDF" in path.parts else "KR"
    flags = []
    if "Rule_36" in path.parts:
        flags.append("rule36_no_reasoning_candidate")
    if re.search(r"Errata|Order|Rehearing", name, re.I):
        flags.append("procedural_or_correction_review")
    if "99_확인필요" in path.parts or name.startswith("0000-"):
        flags.append("metadata_unresolved")
    if re.search(r"상표|디자인|의장", name):
        flags.append("non_patent_title_candidate")
    if "기능항 참고" in path.parts:
        flags.append("reference_copy_candidate")
    parts = name.split("__")
    court = parts[1] if jurisdiction == "US" and len(parts) > 2 else "Patent Court of Korea (unverified metadata)"
    docket = parts[2] if jurisdiction == "US" and len(parts) > 3 else None
    if jurisdiction == "KR":
        match = re.search(r"\d{4}[가-힣]+\d+", name)
        docket = match.group() if match else None
    return {"document_id": hashlib.sha256(rel.encode()).hexdigest()[:20],
            "relative_path": rel, "path": str(path.resolve()), "bytes": path.stat().st_size,
            "jurisdiction": jurisdiction, "court_from_filename": court,
            "docket_from_filename": docket,
            "date_from_filename": parts[0] if jurisdiction == "US" else None,
            "flags": flags, "review_status": "unreviewed",
            "sha256": None, "hash_status": "not_read_for_inventory"}


def inventory(source: Path, out: Path) -> list[dict]:
    pdf_root = source / "사건명_PDF"
    docs = [document_info(path, source) for path in sorted(pdf_root.rglob("*.pdf"))]
    with (source / "판례_목록.csv").open(encoding="utf-8-sig", newline="") as stream:
        posts = {str(int(row["게시판번호"])): row for row in csv.DictReader(stream)}
    for row in docs:
        if row["jurisdiction"] != "KR":
            continue
        prefix = Path(row["path"]).name.split("_", 1)[0]
        post = posts.get(str(int(prefix))) if prefix.isdigit() else None
        if post:
            row["board_metadata"] = {"number": post["게시판번호"], "title": post["제목"],
                                     "posted_date": post["작성일"], "url": post["상세페이지"]}
            if re.search(r"상표|디자인|의장|\((?:상|디|의)\)", post["제목"]):
                row["flags"].append("non_patent_board_title_candidate")
        else:
            row["flags"].append("board_mapping_unresolved")
    save_jsonl(out / "inventory.jsonl", docs)
    return docs


def build_index(source: Path, out: Path, docs: list[dict]) -> dict:
    from pypdf import PdfReader

    db = sqlite3.connect(out / "evidence.sqlite")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS sources (
        path TEXT PRIMARY KEY, sha256 TEXT, kind TEXT, status TEXT, units INTEGER);
      CREATE TABLE IF NOT EXISTS evidence (
        path TEXT, unit INTEGER, text TEXT, PRIMARY KEY(path,unit));
    """)
    # This is an audit/search index, not a training or inference retrieval store.
    md_count = 0
    for path in sorted((source / "병합본_MD").glob("*.md")):
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        key = str(path.resolve())
        cached = db.execute("SELECT sha256,status FROM sources WHERE path=?", (key,)).fetchone()
        if cached != (sha, "extracted"):
            text = raw.decode("utf-8-sig")
            db.execute("DELETE FROM evidence WHERE path=?", (key,))
            db.execute("INSERT INTO evidence VALUES(?,?,?)", (key, 0, text))
            db.execute("INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?)", (key, sha, "merged_md", "extracted", 1))
        md_count += 1
    db.commit()
    print(f"Indexed {md_count} existing Korean Markdown files (audit-only).", flush=True)
    keywords = ("WILLIAMSON", "PHILLIPS", "NAUTILUS", "MARKMAN", "TEVA", "BIOMEDINO",
                "ARISTOCRAT", "WMS_GAMING", "WAG_Acquisition")
    selected = [row for row in docs if "기능항 참고" in Path(row["path"]).parts
                or (row["jurisdiction"] == "US" and any(k.casefold() in Path(row["path"]).name.casefold() for k in keywords))]
    outcomes = []
    for i, row in enumerate(selected, 1):
        path = Path(row["path"])
        key, sha = str(path.resolve()), digest(path)
        row.update(sha256=sha, hash_status="content_verified")
        old = db.execute("SELECT sha256,status,units FROM sources WHERE path=?", (key,)).fetchone()
        if old and old[0] == sha and old[1] == "extracted":
            count = old[2]
            empty = db.execute("SELECT count(*) FROM evidence WHERE path=? AND length(trim(text))=0", (key,)).fetchone()[0]
            status = "extracted"
        else:
            db.execute("DELETE FROM evidence WHERE path=?", (key,))
            try:
                reader = PdfReader(path)
                pages = [(page.extract_text() or "") for page in reader.pages]
                count, empty = len(pages), sum(not page.strip() for page in pages)
                db.executemany("INSERT INTO evidence VALUES(?,?,?)", [(key, n, text) for n, text in enumerate(pages, 1)])
                status = "extracted"
            except Exception as exc:
                count, empty, status = 0, 0, f"error:{type(exc).__name__}"
            db.execute("INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?)", (key, sha, "pdf", status, count))
            db.commit()
        outcomes.append({"document_id": row["document_id"], "path": key, "sha256": sha,
                         "status": status, "pages": count, "empty_pages": empty,
                         "selection": "filename_topic_seed_not_relevance_approval"})
        if i % 5 == 0 or i == len(selected):
            print(f"PDF extraction {i}/{len(selected)}", flush=True)
    save_jsonl(out / "extraction.jsonl", outcomes)
    save_jsonl(out / "inventory.jsonl", docs)
    db.close()
    return {"markdown_indexed": md_count, "selected_pdfs": len(selected),
            "pdfs_extracted": sum(x["status"] == "extracted" for x in outcomes),
            "pdf_pages": sum(x["pages"] for x in outcomes),
            "empty_pdf_pages": sum(x["empty_pages"] for x in outcomes)}


def create_cards(out: Path, docs: list[dict]) -> list[dict]:
    specifications = [
        ("KR-2005HEO7354-STRUCTURE", "0146_2005허7354.pdf", "2005허7354", "특허법원", "2006-11-23", [6, 7, 8, 9, 10],
         "기능적 표현을 구조와 연결해 검토하고, 구조가 뒷받침하지 않는 완전 배출 등의 결과를 임의로 추가하지 않는다.",
         "도면만으로 명세서 기재요건 충족 여부를 단정하지 않는다. 일부 표현에 대한 판단을 모든 기능적 표현의 금지로 일반화하지 않는다."),
        ("US-WILLIAMSON-2015-STRUCTURE", "2015-06-16__CAFC__13-1130__RICHARD_WILLIAMSON", "2013-1130", "US Court of Appeals for the Federal Circuit", "2015-06-16", [16, 17, 18, 21, 22],
         "Examine functional language in context and distinguish a function label from identified supporting structure.",
         "US-specific historical means-plus-function reasoning; not a categorical ban on module or functional language, and not automatically applicable to KR claims."),
        ("US-PHILLIPS-2005-CONTEXT", "2005-07-12__CAFC__2003-1269__PHILLIPS_v_AWH__Opinion", "03-1269, -1286", "US Court of Appeals for the Federal Circuit", "2005-07-12", [35, 36, 37],
         "Review claim language in context; do not assume that adding every embodiment detail improves claim drafting.",
         "Claim-construction reasoning is not a drawing-only test of legal scope; do not infer a universal rule that an omitted drawing feature is necessary or unnecessary.")
    ]
    db = sqlite3.connect(out / "evidence.sqlite")
    cards = []
    for card_id, needle, docket, court, date, pages, principle, limits in specifications:
        matches = [row for row in docs if needle in Path(row["path"]).name and
                   (card_id.startswith("US-") or "기능항 참고" in Path(row["path"]).parts)]
        if len(matches) != 1:
            raise ValueError(f"Expected one source for {card_id}, got {len(matches)}")
        row = matches[0]
        excerpts = []
        for page in pages:
            fetched = db.execute("SELECT text FROM evidence WHERE path=? AND unit=?", (row["path"], page)).fetchone()
            if not fetched or not fetched[0].strip():
                raise ValueError(f"Missing evidence {card_id} page {page}")
            excerpts.append({"pdf_page": page, "page_text_sha256": hashlib.sha256(fetched[0].encode()).hexdigest(),
                             "page_text": fetched[0]})
        cards.append({"card_id": card_id, "jurisdiction": row["jurisdiction"], "docket": docket,
                      "court": court, "decision_date": date, "pdf_path": row["path"],
                      "source_sha256": row["sha256"], "pages": pages,
                      "drafting_principle": principle, "limits": limits,
                      "evidence": excerpts, "source_text_verified": True,
                      "review_status": "draft", "legal_history_status": "unverified",
                      "case_family_status": "unresolved", "reviewer": None,
                      "generator_eligible": False})
    db.close()
    save_jsonl(out / "principle_cards.draft.jsonl", cards)
    return cards


def build_tasks(source_data: Path, out: Path) -> dict:
    expected = {"train": "69c780015eb37da06257cb33795963d77db5d8b60fd0602620ab18b792b561d3",
                "validation": "69c28ae4b5205a9a13ed5cec2081231936e991a123100f8264c6b54e110d2db0"}
    counts, references = {}, []
    patent_sets = {}
    for split in ("train", "validation"):
        path = source_data / "hf_multimodal" / f"{split}.jsonl"
        if digest(path) != expected[split]:
            raise ValueError(f"Frozen {split} source hash mismatch")
        rows, tasks, patents = read_jsonl(path), [], set()
        for row in rows:
            record_id = row["id"]
            patent = record_id.rsplit("_claim", 1)[0]
            patents.add(patent)
            lang = "ko" if any("가" <= c <= "힣" for c in row["messages"][0]["content"]) else "en"
            images = row["images"]
            for image in images:
                resolved = (source_data / image).resolve()
                if not resolved.is_relative_to((source_data / "images").resolve()) or not image.startswith("images/"):
                    raise ValueError(f"Non-drawing input path for {record_id}")
            instruction = (
                "제공된 도면의 구조에 근거해 독립 장치항 1개와 종속항 최대 3개를 한국어로 작성하세요. "
                "종속항은 앞선 항을 인용하고 도면에서 확인 가능한 한정을 추가하세요. "
                "도면으로 확인할 수 없는 치수·재료·성능은 만들지 마세요. "
                "claims, annotations, abstentions 키의 JSON을 출력하세요. 제공된 판례 카드만 주석에 사용하세요. "
                "카드가 없으면 annotations는 빈 배열로 두세요."
                if lang == "ko" else
                "Draft one independent apparatus claim and up to three dependent claims grounded in the drawings. "
                "Each dependent claim must refer to an earlier claim and add a visible, supported limitation. "
                "Do not invent dimensions, materials or performance. Return JSON with claims, annotations and "
                "abstentions. Cite only supplied case cards; use an empty annotations array if none are supplied."
            )
            instruction += (
                '\n각 claims 항목은 number(1부터 연속된 정수), kind("independent" 또는 "dependent"), '
                'depends_on(독립항은 [], 종속항은 앞선 항 번호 하나의 배열), text를 포함합니다. '
                '각 annotations 항목은 claim_number, card_id, mode("provided_during_drafting"), '
                'application을 포함합니다. abstentions는 근거 부족으로 보류한 사항의 문자열 배열입니다. '
                '청구항 번호와 인용항 번호는 사용하되 청구항 본문에 도면 부호는 쓰지 마세요.'
                if lang == 'ko' else
                '\nEach claims item contains number (consecutive integers starting at 1), '
                'kind ("independent" or "dependent"), depends_on ([] for the independent claim; '
                'one earlier claim number for each dependent claim), and text. Each annotations '
                'item contains claim_number, card_id, mode ("provided_during_drafting"), and '
                'application. abstentions is an array of strings describing unsupported matters. '
                'Use claim numbers and dependency references, but no drawing reference numerals in claim text.'
            )
            tasks.append({"task_id": f"claimset-{record_id}", "source_record_id": record_id,
                          "source_split": split, "patent_id": patent, "language": lang,
                          "jurisdiction": next((j for j in ("KR", "US") if patent.startswith(j)), "unassigned"),
                          "images": images, "image_root": str(source_data.resolve()),
                          "prompt": [{"role": "user", "content": instruction}],
                          "case_cards": [], "max_dependent_claims": 3,
                          "status": "needs_drawing_annotation_and_claimset_target",
                          "drawing_evidence": [], "claimset_target": None,
                          "case_family_overlap_review": "unresolved",
                          "training_eligible": False})
        patent_sets[split] = patents
        save_jsonl(out / f"tasks.{split}.jsonl", tasks)
        references.append({"split": split, "path": str(path.resolve()), "sha256": digest(path), "records": len(rows)})
        counts[split] = len(tasks)
    if patent_sets["train"] & patent_sets["validation"]:
        raise ValueError("Source patent ID leakage between train and validation")
    save_json(out / "source_split_integrity.json", {"sources": references,
              "exact_patent_id_overlap": [], "cross_publication_family_review": "unresolved",
              "test_content_opened": False, "frozen_sources_modified": False})
    return counts


def build_visual_seeds(source_data: Path, out: Path) -> dict:
    from rl_materials.contracts import validate_output

    authored = json.loads((ROOT / "rl_materials/visual_seeds.json").read_text(encoding="utf-8"))
    tasks = {row["source_record_id"]: row for row in read_jsonl(out / "tasks.train.jsonl")}
    seeds, preferences = [], []
    for seed in authored:
        task = tasks[seed["record_id"]]
        image_provenance = []
        for name in task["images"]:
            path = source_data / name
            if not path.is_file():
                raise ValueError(f"Visual seed image missing: {name}")
            image_provenance.append({"path": name, "sha256": digest(path)})
        claims = [{"number": i, "kind": "independent" if i == 1 else "dependent",
                   "depends_on": seed.get("parents", [[], [1], [1], [1]])[i - 1], "text": text}
                  for i, text in enumerate(seed["claims"], 1)]
        target = {"claims": claims, "annotations": [], "abstentions": seed["abstentions"]}
        errors = validate_output(target, {})
        if errors:
            raise ValueError(f"Visual seed {seed['record_id']}: {errors}")
        value = {"source_record_id": seed["record_id"], "source_split": "train",
                 "images": image_provenance, "drawing_observations": seed["observations"],
                 "claim_evidence": seed["claim_evidence"], "claimset_target": target,
                 "author": "Codex assistant", "author_visual_review": True,
                 "review_status": "draft_needs_independent_review", "training_eligible": False,
                 "author_context": "Allowed original independent-target excerpts were seen before visual drafting; this is not a blind evaluation.",
                 "suggested_posthoc_card": seed["suggested_posthoc_card"],
                 "citation_note": "Candidate post-hoc connection only; no case supplied to the drawing generator and no causal attribution claimed."}
        seeds.append(value)
        # Clearly labeled candidate examples, never count these as independent gold.
        rejected = copy.deepcopy(target)
        rejected["claims"][-1]["text"] = (
            "제1항에 있어서, 상기 장치의 모든 구성요소는 순도 99.999%의 티타늄으로 형성되는 장치."
        )
        rejected["claims"][-1]["depends_on"] = [1]
        preferences.append({"pair_id": f"{seed['record_id']}-unsupported-material",
                            "source_record_id": seed["record_id"], "source_split": "train",
                            "chosen": target, "rejected": rejected,
                            "reason": "The drawings provide no evidence for the introduced composition and purity.",
                            "label_origin": "assistant_authored_controlled_corruption",
                            "review_status": "candidate_needs_independent_review", "training_eligible": False})
        rejected = copy.deepcopy(target)
        rejected["annotations"] = [{"claim_number": 1, "card_id": "UNSUPPLIED-CASE",
                                    "mode": "provided_during_drafting", "application": "구조 확인"}]
        preferences.append({"pair_id": f"{seed['record_id']}-unsupplied-citation",
                            "source_record_id": seed["record_id"], "source_split": "train",
                            "chosen": target, "rejected": rejected,
                            "reason": "No case was supplied; the rejection invents an available drafting source.",
                            "label_origin": "assistant_authored_controlled_corruption",
                            "review_status": "candidate_needs_independent_review", "training_eligible": False})
    save_jsonl(out / "claimsets.visual_drafts.jsonl", seeds)
    save_jsonl(out / "preferences.controlled_candidates.jsonl", preferences)
    lines = ["# 청구항 세트 검토 자료", "",
             "기존 v2 Gemma를 이어서 학습하기 위한 검토 초안입니다. GPU 학습은 실행하지 않았습니다.", "",
             "4건의 학습 도면에서 독립항 4개와 종속항 12개를 작성했습니다. 작성자가 도면을 확인했으나 독립 검토를 거친 정답 데이터는 아닙니다.",
             "기존 독립항 정답 일부를 먼저 본 상태에서 작성했으므로 블라인드 평가로 사용하지 않습니다.", "",
             "판례 주석은 사건번호·법원·선고일·근거 페이지를 저장한 카드로 연결합니다. 가중치 안에서 실제로 어떤 판례를 떠올렸는지를 증명하는 기능은 아닙니다.",
             "아래 초안에는 판례를 제공하지 않았습니다. 판례 연결 후보는 작성 후 검토용이며, 실제 작성 근거로 표시하지 않습니다.", "",
             "## 검토할 항목", "",
             "- 각 한정이 연결된 도면에서 확인되는가", "- 종속항이 인용항의 한정을 유지하며 새로운 한정을 추가하는가",
             "- 도면에 없는 재료·수치·성능을 단정하지 않는가", "- 서로 다른 실시형태의 양립할 수 없는 특징을 합치지 않았는가",
             "- 판례 원칙의 적용 범위와 근거 페이지가 실제 주석 내용을 뒷받침하는가", ""]
    for seed in seeds:
        lines += [f"## {seed['source_record_id']}", "", "출처 분할: train / 상태: 독립 검토 전", ""]
        for claim in seed["claimset_target"]["claims"]:
            lines += [f"**청구항 {claim['number']}**", "", claim["text"], ""]
            indices = seed["claim_evidence"][claim["number"] - 1]
            observations = [seed["drawing_observations"][i] for i in indices]
            lines += ["도면 근거: " + "; ".join(f"이미지 {v['image_index'] + 1}, 도면 {v['figure']}: {v['visible']}" for v in observations), ""]
        lines += ["보류 사항: " + " ".join(seed["claimset_target"]["abstentions"]), "",
                  "작성 후 판례 검토 후보: `" + seed["suggested_posthoc_card"] + "` (적용 검토 전)", ""]
        for n, image in enumerate(seed["images"], 1):
            path = (source_data / image["path"]).resolve().as_posix()
            lines += [f"![검토 도면 {n}](<{path}>)", ""]
    (out / "REVIEW_PACKET.md").write_text("\n".join(lines), encoding="utf-8")
    return {"visual_draft_claimsets": len(seeds),
            "visual_draft_dependent_claims": sum(len(x["claimset_target"]["claims"]) - 1 for x in seeds),
            "controlled_preference_candidates": len(preferences),
            "independently_reviewed_gold": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, default=ROOT / "data/rl_source_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "data/case_rl")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if not (source / "사건명_PDF").is_dir():
        raise ValueError("Expected the parent patent-case folder")
    if not out.is_relative_to((ROOT / "data").resolve()) or out.is_relative_to(source):
        raise ValueError("Output must be under this repository's data directory, outside sources")
    if out == args.source_data.resolve() or args.source_data.resolve().is_relative_to(out) or out.is_relative_to(args.source_data.resolve()):
        raise ValueError("Output and frozen-source directories must be disjoint")
    out.mkdir(parents=True, exist_ok=True)
    docs = inventory(source, out)
    indexed = build_index(source, out, docs)
    cards = create_cards(out, docs)
    tasks = build_tasks(args.source_data, out)
    seeds = build_visual_seeds(args.source_data, out)
    report = {"status": "ANNOTATION_WORK_PACK_BUILT_NOT_TRAINING_READY",
              "gpu_allowed": False, "gpu_or_inference_calls": 0,
              "resume_adapter": "Mepeng22/gemma-4-31b-claim-lora-v2",
              "resume_revision": "cc04579366bbb88663029d53b2aafcc84159589e",
              "inventory_pdfs": len(docs), "jurisdictions": dict(Counter(x["jurisdiction"] for x in docs)),
              "flags": dict(Counter(flag for x in docs for flag in x["flags"])),
              "index": indexed, "source_verified_draft_cards": len(cards), "approved_cards": 0,
              "annotation_tasks": tasks, "reviewed_claimsets": 0, "reviewed_preference_pairs": 0,
              "visual_seed_pack": seeds,
              "calibrated_reward": False, "raw_sources_used_as_model_context": False,
              "blockers": ["Reviewed drawing-grounded independent/dependent claim-set targets missing",
                           "Case applicability, subsequent history and target-patent overlap unresolved",
                           "Reviewed preference and held-out reward calibration examples missing",
                           "Multimodal continuation trainer and real-processor rehearsal not yet implemented"],
              "scope_note": "Inventory covers the synced PDF tree; text extraction is complete only for the recorded seed selection."}
    save_json(out / "readiness.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
