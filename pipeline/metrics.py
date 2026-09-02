"""Scoring for generated claims — no torch, no model, no GPU.

Split out of evaluate.py so tools/baseline.py can import it. baseline.py is the
gate that decides whether a GPU is worth paying for, and it runs from wherever
the serving endpoint is reachable — which may be a machine that cannot import
torch at all. Nothing here needs it: these are string checks, ROUGE and chrF.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

# claim_prompt lives in serving/ and is the one definition of the language
# rules; tools/baseline.py imports it the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))

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
