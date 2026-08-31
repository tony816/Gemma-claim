#!/usr/bin/env python3
"""Draft an independent patent claim from drawings, via a RunPod Serverless
vLLM endpoint running Gemma 4 31B.

Why the base model and not the fine-tuned adapter: see
run_artifacts/POST_HOC_GENERATION_AUDIT.md. The adapter learned the output
format completely and lost image grounding entirely, producing well-formed
claims about the wrong apparatus in 10 of 12 test records. The base model reads
the drawings correctly; what it lacks is the output format, and that is a
prompting problem rather than a training one. This client supplies the format
through the system prompt and a narrow output sanitiser.

Usage:
    export RUNPOD_API_KEY=...          # never hard-code it
    python claim_client.py fig1.png fig2.png --context "inserted into a slot"

The endpoint id defaults to the one this project provisioned; override with
--endpoint or RUNPOD_ENDPOINT_ID.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "e5yibbozs40ji4"
SERVED_MODEL = "gemma4-31b"

# The base model, unprompted, answers with a markdown preamble, a bolded
# "Claim 1:" heading, and a trailing "Drafting Notes" section. Each instruction
# below suppresses one of those observed behaviours.
SYSTEM_PROMPT = (
    "You are a patent attorney drafting claims from technical drawings.\n"
    "Study every drawing you are given before answering. Ground the claim in "
    "the structure actually shown: the components, their arrangement, and how "
    "they connect.\n"
    "\n"
    "Output ONLY the text of a single independent apparatus claim. Specifically:\n"
    "- No preamble, no introduction, no restatement of the request.\n"
    "- No markdown: no headings, no bold, no bullet points, no horizontal rules.\n"
    "- No 'Claim 1:' label.\n"
    "- No drafting notes, rationale, or commentary after the claim.\n"
    "- No reference numerals anywhere, with or without parentheses.\n"
    "\n"
    "Begin with an article ('A' or 'An'). Name the apparatus, then write "
    "'comprising:' followed by the elements. Use one sentence. End with a "
    "period."
)

# Anything the sanitiser strips is a prompt-compliance failure worth seeing, so
# it reports what it removed rather than silently cleaning up.
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)
_LABEL = re.compile(r"(?im)^\s*(?:\*\*)?\s*claim\s*\d+\s*[:.]?\s*(?:\*\*)?[ \t]*")
# A claim opens with an article and reaches a transition phrase. Searching for
# that pair is what separates the claim from the base model's preamble, which
# opens with "Based on the provided drawings...".
_CLAIM_START = re.compile(r"(?ms)^[ \t]*(?:A|An)\s+.*?\b(?:comprising|consisting|including)\b")
_ARTICLE_LINE = re.compile(r"(?m)^[ \t]*(?:A|An)\s+\S")
# Only cut on a marker that FOLLOWS the claim - the base model also emits a
# horizontal rule before it.
_NUMERAL = re.compile(
    r"\s+\d{1,4}\b(?!\.\d)(?=\s*(?:[;,.]|and\b|the\b|to\b|such\b|wherein\b|is\b|are\b"
    r"|that\b|which\b|for\b|in\b|of\b|connected\b|disposed\b|extending\b|having\b))"
)
_NOTES = re.compile(
    r"\n\s*(?:\*{3,}|-{3,}|_{3,}|#{1,6}\s|Drafting\s+Notes|Notes\s*&|Rationale\b|Notes\s*:)",
    re.IGNORECASE,
)


def sanitise(text: str) -> tuple[str, list[str]]:
    """Trim the wrapper the base model adds around the claim itself.

    Order matters: the preamble is located and dropped before any trailing
    section is cut, because the model puts a horizontal rule on both sides of
    the claim and cutting on the first one would discard the claim.
    """
    removed: list[str] = []
    out = text.strip()

    if _FENCE.search(out):
        out = _FENCE.sub("", out).strip()
        removed.append("code fence")

    stripped = _LABEL.sub("", out)
    if stripped != out:
        out, _ = stripped.strip(), removed.append("claim label")

    # Locate where the claim actually begins.
    m = _CLAIM_START.search(out) or _ARTICLE_LINE.search(out)
    if m:
        if m.start() > 0:
            removed.append("preamble")
        out = out[m.start():]
    elif out:
        removed.append("no claim opening found - output returned as-is")

    # The notes cut runs while the horizontal rules are still intact: stripping
    # emphasis first would turn "***" into "*" and leave it behind.
    cut = _NOTES.search(out)
    if cut:
        out = out[: cut.start()]
        removed.append("trailing notes section")

    if "**" in out or "__" in out:
        out = out.replace("**", "").replace("__", "")
        removed.append("emphasis markers")

    # The model complies with "no reference numerals in parentheses" by writing
    # them bare ("a housing 10"), so strip a bare integer only where a numeral
    # can appear: directly before punctuation or a structural word. A real
    # quantity is followed by its unit ("100 pL") and is left alone.
    stripped = _NUMERAL.sub("", out)
    if stripped != out:
        out = stripped
        removed.append("reference numerals")

    return out.strip(), removed


def encode_image(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None or not mime.startswith("image/"):
        mime = "image/png"
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"


def build_content(image_paths: list[str], context: str | None) -> list[dict]:
    """Image order is preserved: the drawings are a figure sequence and the
    later figures are read relative to the earlier ones."""
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": encode_image(p)}} for p in image_paths
    ]
    ask = "Draft an independent apparatus claim for the invention shown in these drawings."
    if context:
        ask += f"\n\nAdditional context: {context}"
    content.append({"type": "text", "text": ask})
    return content


def post(url: str, payload: dict, api_key: str) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def run(endpoint: str, api_key: str, content: list[dict], max_tokens: int, temperature: float) -> dict:
    # worker-vllm accepts three input shapes. This is the OpenAI passthrough:
    # only under `openai_route`/`openai_input` are `model` and `max_tokens`
    # actually honoured. The bare {"messages": ..., "sampling_params": ...}
    # shorthand silently ignores both.
    payload = {
        "input": {
            "openai_route": "/v1/chat/completions",
            "openai_input": {
                "model": SERVED_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }
    }

    base = f"https://api.runpod.ai/v2/{endpoint}"
    job = post(f"{base}/run", payload, api_key)
    job_id = job["id"]
    print(f"job {job_id} submitted; a cold start takes a few minutes", file=sys.stderr)

    delay, waited = 5, 0
    while True:
        time.sleep(delay)
        waited += delay
        req = urllib.request.Request(
            f"{base}/status/{job_id}", headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = json.loads(resp.read())

        state = status.get("status")
        if state == "COMPLETED":
            return status
        if state in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise SystemExit(f"job {state}: {json.dumps(status)[:2000]}")
        print(f"  {state} ({waited}s)", file=sys.stderr)
        delay = min(delay * 1.5, 20)


def extract_text(result: dict) -> str:
    out = result.get("output")
    # RAW_OPENAI_OUTPUT=true returns the OpenAI response, sometimes wrapped in a
    # single-element list by the async handler.
    if isinstance(out, list) and out:
        out = out[0]
    if isinstance(out, dict) and "choices" in out:
        return out["choices"][0]["message"]["content"]
    raise SystemExit(f"unrecognised output shape: {json.dumps(result)[:2000]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+", help="drawing files, in figure order")
    ap.add_argument("--context", help="extra description of the invention")
    ap.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT))
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--raw", action="store_true", help="print the model output unsanitised")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("set RUNPOD_API_KEY in the environment")

    for p in args.images:
        if not os.path.exists(p):
            raise SystemExit(f"no such image: {p}")

    content = build_content(args.images, args.context)
    result = run(args.endpoint, api_key, content, args.max_tokens, args.temperature)
    text = extract_text(result)

    if args.raw:
        print(text)
        return

    claim, removed = sanitise(text)
    if removed:
        print(f"[sanitiser removed: {', '.join(removed)}]", file=sys.stderr)
    print(claim)


if __name__ == "__main__":
    main()
