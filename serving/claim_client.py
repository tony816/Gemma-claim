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
import sys
import time
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "e5yibbozs40ji4"
SERVED_MODEL = "gemma4-31b"

# The base model, unprompted, answers with a markdown preamble, a bolded
# "Claim 1:" heading, and a trailing "Drafting Notes" section. Each instruction
# below suppresses one of those observed behaviours.
from claim_prompt import SYSTEM_PROMPT, sanitise, system_prompt  # noqa: F401  (re-exported)


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


def run(endpoint: str, api_key: str, content: list[dict], max_tokens: int,
        temperature: float, lang: str = "en") -> dict:
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
                    {"role": "system", "content": system_prompt(lang)},
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
    ap.add_argument("--lang", choices=("en", "ko"), default="en",
                    help="claim language (default: en)")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("set RUNPOD_API_KEY in the environment")

    for p in args.images:
        if not os.path.exists(p):
            raise SystemExit(f"no such image: {p}")

    content = build_content(args.images, args.context)
    result = run(args.endpoint, api_key, content, args.max_tokens, args.temperature,
                 args.lang)
    text = extract_text(result)

    if args.raw:
        print(text)
        return

    claim, removed = sanitise(text, args.lang)
    if removed:
        print(f"[sanitiser removed: {', '.join(removed)}]", file=sys.stderr)
    print(claim)


if __name__ == "__main__":
    main()
