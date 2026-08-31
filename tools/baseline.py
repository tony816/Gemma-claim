#!/usr/bin/env python3
"""Score the prompted base model before deciding to fine-tune.

The previous run trained first and measured afterwards. It spent about $21 and
produced a model that, on free-running generation, wrote a claim about the same
apparatus as the reference in 0 of 12 test records — while the base model read
the drawings correctly. Nobody had scored the base model, so there was nothing
to compare against and no moment where stopping was the obvious call.

This scores the base model through the deployed endpoint, using the same
metrics pipeline/evaluate.py applies to a fine-tune. Run it before the GPU.
Whatever number comes out is the bar the fine-tune has to clear to be worth
paying for.

    export RUNPOD_API_KEY=...
    python tools/baseline.py --split validation

Costs one cold start plus a few seconds per record — cents, not dollars. The
test split is deliberately not the default: it is spent once, after the config
and checkpoint are frozen.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "serving"))

DEFAULT_ENDPOINT = "fdiltabt78bogm"
SERVED_MODEL = "gemma4-31b"


def encode_image(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def ask(endpoint: str, key: str, system: str, rec, max_tokens: int, temperature: float) -> str:
    """One record through worker-vllm's OpenAI passthrough.

    The passthrough shape is the only one that honours top-level model and
    max_tokens; the bare messages/sampling_params shorthand silently ignores
    both, which invalidated an earlier round of endpoint tests.
    """
    from common import build_chat_messages

    parts: list[dict] = []
    for img in rec.images:                       # source order is part of the contract
        parts.append({"type": "image_url", "image_url": {"url": encode_image(Path(img))}})
    text = " ".join(
        p["text"] for m in build_chat_messages(rec, for_prompt=True)
        for p in m["content"] if p.get("type") == "text"
    ).strip()
    parts.append({"type": "text", "text": text or "Draft an independent apparatus claim."})

    payload = {"input": {
        "openai_route": "/v1/chat/completions",
        "openai_input": {
            "model": SERVED_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": parts}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }}}

    base = f"https://api.runpod.ai/v2/{endpoint}"
    req = urllib.request.Request(
        f"{base}/run", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        job = json.loads(resp.read())

    delay = 4.0
    while True:
        time.sleep(delay)
        s = urllib.request.Request(f"{base}/status/{job['id']}",
                                   headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(s, timeout=60) as resp:
            st = json.loads(resp.read())
        if st.get("status") == "COMPLETED":
            out = st["output"]
            if isinstance(out, list) and out:
                out = out[0]
            return out["choices"][0]["message"]["content"]
        if st.get("status") in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise SystemExit(f"job {st.get('status')}: {json.dumps(st)[:800]}")
        delay = min(delay * 1.3, 15)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation",
                    help="validation (default). Use test only once, at the very end.")
    ap.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT))
    ap.add_argument("--limit", type=int, default=0, help="score only the first N records")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("-o", "--out", default="baseline_predictions.jsonl")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise SystemExit("set RUNPOD_API_KEY")
    if args.split == "test":
        print("⚠ scoring the TEST split. It is spent once — this should be the "
              "final measurement, not an exploratory one.\n", file=sys.stderr)

    from claim_prompt import SYSTEM_PROMPT, sanitise
    from common import find_package_root, load_all
    from evaluate import claim_form_checks, qualitative_summary, text_metrics

    recs = load_all(find_package_root())[args.split]
    if args.limit:
        recs = recs[:args.limit]
    print(f"scoring {len(recs)} {args.split} records through {args.endpoint}\n"
          f"(first call waits out a cold start, ~5 min)\n")

    rows, preds, refs = [], [], []
    for i, rec in enumerate(recs, 1):
        raw = ask(args.endpoint, key, SYSTEM_PROMPT, rec, args.max_tokens, args.temperature)
        cleaned, removed = sanitise(raw)
        rows.append({"record_id": rec.record_id, "n_images": len(rec.images),
                     "reference": rec.target, "raw": raw, "prediction": cleaned,
                     "sanitiser_removed": removed, "checks": claim_form_checks(cleaned)})
        preds.append(cleaned)
        refs.append(rec.target)
        print(f"  [{i}/{len(recs)}] {rec.record_id}: {len(cleaned.split())} words"
              + (f"  (cleaned: {', '.join(removed)})" if removed else ""))

    metrics = {"split": args.split, "n": len(rows), "model": "base + claim prompt",
               **text_metrics(preds, refs), **qualitative_summary(rows)}

    # chrF is the number that fell while loss fell in the last run — the one
    # signal that the model had stopped conditioning on its input. A baseline
    # without it is missing the measurement this tool exists to provide.
    if metrics.get("chrf") is None:
        print(f"\n⚠ chrF unavailable ({metrics.get('chrf_error')}). "
              "Install it — this is the metric that exposes mode collapse:\n"
              "    pip install sacrebleu rouge-score\n", file=sys.stderr)

    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    Path(args.out).with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    print("\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\npredictions → {args.out}")
    print(
        "\nThis is the bar. A fine-tune is worth its GPU time only if it beats these\n"
        "numbers on free-running generation — not on training loss, which is\n"
        "teacher-forced and fell all the way through the last run while chrF, the\n"
        "number that tracks whether the model is reading its input at all, fell too.\n"
        "Read a few predictions above before trusting any of it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
