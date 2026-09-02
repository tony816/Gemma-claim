"""Identical-conditions evaluation of the base and the fine-tuned model.

The base numbers come from the same weights, same collator and same decoding
parameters as the tuned numbers -- the adapter is simply disabled -- so the
comparison is not confounded by a second load path.

Test is gated: it is only read after training has finished and a best
checkpoint has been recorded, and it is scored exactly once.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch

# claim_prompt lives in serving/ and is the one definition of the language
# rules. tools/baseline.py imports it the same way; the two have to agree or
# the baseline and the fine-tune are not being scored on the same footing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))

from common import (OUT, SEED, SPLITS, die, find_package_root, load_all, load_vlm,
                    log, set_all_seeds, write_json)
from dataset import Collator, RecordDataset, encode_prompt_only

MODEL_ID = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
REVISION = os.environ.get("BASE_MODEL_REVISION", "main")
ADAPTER_DIR = OUT / "final_model_or_adapter"

# Independent physical-apparatus claim shape. Dependent claims back-reference a
# numbered claim; independent ones open with an article and a body noun.
DEPENDENT_RE = re.compile(r"\bof\s+claim\s+\d+|\baccording\s+to\s+claim\s+\d+", re.I)
APPARATUS_RE = re.compile(
    r"\b(device|apparatus|system|cartridge|assembly|module|instrument|unit)\b", re.I
)
TRANSITION_RE = re.compile(r"\b(comprising|including|consisting of|configured to)\b", re.I)


def repetition_score(text: str, n: int = 10) -> float:
    """Fraction of n-grams that are duplicates; ~0 is healthy, ->1 is a loop."""
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    c = Counter(grams)
    return round(1.0 - len(c) / len(grams), 4)


def claim_form_checks(text: str) -> dict:
    t = text.strip()
    return {
        "empty": not t,
        "starts_with_article": bool(re.match(r"^(a|an)\s", t, re.I)),
        "mentions_apparatus_noun": bool(APPARATUS_RE.search(t)),
        "has_transition_phrase": bool(TRANSITION_RE.search(t)),
        "looks_dependent": bool(DEPENDENT_RE.search(t)),
        "ends_with_period": t.endswith("."),
        "repetition_10gram": repetition_score(t),
        "words": len(t.split()),
    }


def independent_claim_ok(chk: dict) -> bool:
    return (
        not chk["empty"]
        and not chk["looks_dependent"]
        and chk["mentions_apparatus_noun"]
        and chk["has_transition_phrase"]
        and chk["ends_with_period"]
        and chk["repetition_10gram"] < 0.30
    )


def text_metrics(preds: list[str], refs: list[str]) -> dict:
    out: dict = {}
    try:
        from rouge_score import rouge_scorer

        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        vals = [sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
        out["rougeL_f"] = round(sum(vals) / len(vals), 4) if vals else None
    except Exception as exc:
        out["rougeL_f"] = None
        out["rougeL_error"] = str(exc)
    try:
        import sacrebleu

        out["chrf"] = round(sacrebleu.corpus_chrf(preds, [refs]).score, 4)
    except Exception as exc:
        out["chrf"] = None
        out["chrf_error"] = str(exc)
    return out


@torch.no_grad()
def split_loss(model, processor, records, tag: str) -> dict:
    ds = RecordDataset(processor, records)
    coll = Collator(pad_token_id=processor.tokenizer.pad_token_id or 0)
    total_loss, total_tok = 0.0, 0
    for i in range(len(ds)):
        batch = coll([ds[i]])
        batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(**batch)
        ntok = int((batch["labels"] != -100).sum())
        total_loss += float(out.loss) * ntok
        total_tok += ntok
    mean = total_loss / max(total_tok, 1)
    log(f"{tag}: loss={mean:.4f} scored_tokens={total_tok}")
    return {
        "loss": round(mean, 6),
        "perplexity": round(math.exp(mean), 4) if mean < 20 else None,
        "scored_target_tokens": total_tok,
    }


@torch.no_grad()
def generate_split(model, processor, records, max_new_tokens: int, tag: str) -> list[dict]:
    rows = []
    for rec in records:
        enc = encode_prompt_only(processor, rec)
        enc = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in enc.items()}
        plen = int(enc["input_ids"].shape[-1])
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1, temperature=None, top_p=None, top_k=None,
            pad_token_id=processor.tokenizer.pad_token_id or 0,
        )
        gen = processor.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
        rows.append({
            "record_id": rec.record_id,
            "split": rec.split,
            "n_images": len(rec.images),
            "reference": rec.target,
            "prediction": gen,
            "checks": claim_form_checks(gen),
        })
        log(f"  [{tag}] {rec.record_id}: {len(gen.split())} words")
    return rows


def qualitative_summary(rows: list[dict]) -> dict:
    chks = [r["checks"] for r in rows]
    n = max(len(chks), 1)
    return {
        "n": len(chks),
        "empty_responses": sum(c["empty"] for c in chks),
        "looks_dependent": sum(c["looks_dependent"] for c in chks),
        "missing_apparatus_noun": sum(not c["mentions_apparatus_noun"] for c in chks),
        "missing_transition_phrase": sum(not c["has_transition_phrase"] for c in chks),
        "unterminated_no_final_period": sum(not c["ends_with_period"] for c in chks),
        "excessive_repetition_gt_0_3": sum(c["repetition_10gram"] > 0.3 for c in chks),
        "well_formed_independent_claims": sum(independent_claim_ok(c) for c in chks),
        "well_formed_rate": round(sum(independent_claim_ok(c) for c in chks) / n, 4),
        "mean_words": round(sum(c["words"] for c in chks) / n, 1),
    }


def by_language(rows: list[dict]) -> dict:
    """Metrics split by the language of the reference.

    The release is 612 Korean of 694, so a single aggregate is the Korean
    number with the rest rounded away. The dataset handoff requires the two
    reported separately, and tools/baseline.py already does -- this is the same
    breakdown, in the same shape, so the baseline and the fine-tune can be put
    side by side.
    """
    from claim_prompt import detect_lang

    langs = [detect_lang(r["reference"]) for r in rows]
    out: dict = {}
    for lang in sorted(set(langs)):
        idx = [i for i, x in enumerate(langs) if x == lang]
        out[lang] = {
            "n": len(idx),
            **text_metrics([rows[i]["prediction"] for i in idx],
                           [rows[i]["reference"] for i in idx]),
            **qualitative_summary([rows[i] for i in idx]),
        }
    return out


def language_drift(rows: list[dict]) -> int:
    """Records answered in a different language from their reference.

    Mode collapse showed up last run as a length collapse; on a Korean-majority
    corpus it can just as easily show up as the model reverting to English.
    """
    from claim_prompt import detect_lang

    return sum(1 for r in rows
               if detect_lang(r["prediction"]) != detect_lang(r["reference"]))


def main() -> None:
    from peft import PeftModel
    from transformers import AutoProcessor

    set_all_seeds(SEED)
    splits = os.environ.get("EVAL_SPLITS", "validation,test").split(",")

    if "test" in splits:
        cfg = OUT / "training_config.yaml"
        if not cfg.is_file():
            die("TEST_EVAL_BEFORE_FREEZE",
                "training_config.yaml is absent: the configuration and checkpoint "
                "are not frozen, so the test split must not be read yet.")

    audit = json.loads((OUT / "token_length_audit.json").read_text())
    max_new = int(os.environ.get("MAX_NEW_TOKENS", audit["recommended_max_new_tokens"]))
    log(f"max_new_tokens={max_new} (from token audit, no target is truncated)")

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    pkg = find_package_root()
    data = load_all(pkg)

    log("Loading base model (bf16).")
    model = load_vlm(
        MODEL_ID, REVISION, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager",
    )
    if not ADAPTER_DIR.is_dir():
        die("ADAPTER_MISSING", f"No adapter at {ADAPTER_DIR}; run training first.")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()
    model.config.use_cache = True

    metrics: dict = {"base_model": MODEL_ID, "revision": REVISION,
                     "decoding": {"do_sample": False, "num_beams": 1,
                                  "max_new_tokens": max_new},
                     "base": {}, "tuned": {}}

    for split in splits:
        recs = data[split]
        log(f"===== {split} (n={len(recs)}) =====")

        # Base: same object with the adapter switched off.
        with model.disable_adapter():
            base_loss = split_loss(model, processor, recs, f"base/{split}")
            base_rows = generate_split(model, processor, recs, max_new, f"base/{split}")
        tuned_loss = split_loss(model, processor, recs, f"tuned/{split}")
        tuned_rows = generate_split(model, processor, recs, max_new, f"tuned/{split}")

        base_tm = text_metrics([r["prediction"] for r in base_rows],
                               [r["reference"] for r in base_rows])
        tuned_tm = text_metrics([r["prediction"] for r in tuned_rows],
                                [r["reference"] for r in tuned_rows])

        metrics["base"][split] = {**base_loss, **base_tm,
                                  "qualitative": qualitative_summary(base_rows),
                                  "by_language": by_language(base_rows),
                                  "language_drift": language_drift(base_rows)}
        metrics["tuned"][split] = {**tuned_loss, **tuned_tm,
                                   "qualitative": qualitative_summary(tuned_rows),
                                   "by_language": by_language(tuned_rows),
                                   "language_drift": language_drift(tuned_rows)}

        with (OUT / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as fh:
            for b, t in zip(base_rows, tuned_rows):
                fh.write(json.dumps({
                    "record_id": t["record_id"], "split": split,
                    "n_images": t["n_images"], "reference": t["reference"],
                    "base_prediction": b["prediction"], "base_checks": b["checks"],
                    "tuned_prediction": t["prediction"], "tuned_checks": t["checks"],
                }, ensure_ascii=False) + "\n")
        log(f"{split} base : {metrics['base'][split]}")
        log(f"{split} tuned: {metrics['tuned'][split]}")

    metrics["VALIDATION_COMPLETE"] = "validation" in splits
    metrics["TEST_EVALUATION_COMPLETE"] = "test" in splits
    prev = {}
    if (OUT / "metrics.json").is_file():
        try:
            prev = json.loads((OUT / "metrics.json").read_text())
        except Exception:
            prev = {}
    for k in ("base", "tuned"):
        merged = {**prev.get(k, {}), **metrics[k]}
        metrics[k] = merged
    write_json(OUT / "metrics.json", metrics)
    log("EVALUATION_DONE")


if __name__ == "__main__":
    main()
