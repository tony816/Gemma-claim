#!/usr/bin/env python3
"""Draft an independent patent claim from drawings, via a RunPod Serverless
vLLM endpoint running Gemma 4 31B.

Defaults to the cumulative v2 + RL adapter, claim-v3, without a system turn.
The response model ID is verified before
output is displayed. See serving/README.md for deployment and interpretation.

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

from client_config import load_local_env

load_local_env()

DEFAULT_ENDPOINT = "fdiltabt78bogm"
SERVED_MODEL = "gemma4-31b"
TUNED_MODEL = "claim-v3"
TRAINING_PROMPTS = {
    "ko": "제공된 발명 도면을 근거로 바이오·분자진단 분야의 독립된 물리적 장치·기기·시스템·카트리지 또는 조립체 청구항을 한국어로 작성하세요.",
    "en": "Draft an independent physical apparatus, device, system, cartridge, or assembly patent claim in the bio/molecular-diagnostics field based on the provided invention drawings.",
}
RL_PROMPTS = {
    'ko': '제공된 도면에 근거해 독립항 1개와 근거가 있는 종속항 2개를 한국어로 작성하세요. '
          '도면에 없는 기술 구성·효과를 추가하지 마세요. JSON 객체만 출력하세요. '
          '형식: {"claims":[{"number":1,"kind":"independent","depends_on":[],"text":"..."},'
          '{"number":2,"kind":"dependent","depends_on":[1],"text":"제1항에 있어서, ..."},'
          '{"number":3,"kind":"dependent","depends_on":[1],"text":"제1항에 있어서, ..."}],'
          '"annotations":[],"abstentions":[]}. 각 종속항은 앞선 항 하나만 인용하고 범위를 좁히세요. '
          '판례 카드가 제공되지 않았으므로 annotations는 빈 배열로 두세요. 불확실하거나 근거가 없는 사항은 abstentions에 적으세요.',
    'en': 'Using only the supplied drawings, draft one independent claim and two supported dependent claims. '
          'Do not invent technical features or effects. Return only a JSON object with claims, annotations, abstentions. '
          'Each claim has number, kind (independent/dependent), depends_on (an array), text. '
          'Number from 1; the independent claim has no parent; each dependent narrows one earlier claim and explicitly references it. '
          'No case cards were supplied, so annotations must be []. List uncertainties in abstentions.'
}

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


def build_content(image_paths: list[str], context: str | None, lang: str = "en") -> list[dict]:
    """Image order is preserved: the drawings are a figure sequence and the
    later figures are read relative to the earlier ones."""
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": encode_image(p)}} for p in image_paths
    ]
    ask = (RL_PROMPTS if TUNED_MODEL == 'claim-v3' else TRAINING_PROMPTS)[lang]
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


def iter_job(endpoint: str, api_key: str, messages: list[dict], max_tokens: int,
             temperature: float, model: str = TUNED_MODEL, timeout: float = 760):
    if os.environ.get('CLAIM_ENDPOINT_PAUSED') == '1':
        raise RuntimeError('Requests are paused by CLAIM_ENDPOINT_PAUSED=1')
    # worker-vllm accepts three input shapes. This is the OpenAI passthrough:
    # only under `openai_route`/`openai_input` are `model` and `max_tokens`
    # actually honoured. The bare {"messages": ..., "sampling_params": ...}
    # shorthand silently ignores both.
    payload = {
        "input": {
            "openai_route": "/v1/chat/completions",
            "openai_input": {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        },
        "policy": {"executionTimeout": int(timeout * 1000),
                   "ttl": int((timeout + 60) * 1000)},
    }

    base = f"https://api.runpod.ai/v2/{endpoint}"
    job = post(f"{base}/run", payload, api_key)
    job_id = job["id"]
    deadline = time.monotonic() + timeout
    delay = 2
    completed = False
    try:
        yield job
        while time.monotonic() < deadline:
            time.sleep(min(delay, max(0, deadline - time.monotonic())))
            req = urllib.request.Request(
                f"{base}/status/{job_id}", headers={"Authorization": f"Bearer {api_key}"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = json.loads(resp.read())

            state = status.get("status")
            if state == "COMPLETED":
                completed = True
                yield status
                return
            if state in ("FAILED", "CANCELLED", "TIMED_OUT"):
                completed = True
                raise RuntimeError(f"RunPod job {job_id}: {state}")
            yield status
            delay = min(delay * 1.5, 10)
        raise TimeoutError(f"RunPod job {job_id} exceeded {timeout:g}s")
    finally:
        if not completed:
            try:
                post(f"{base}/cancel/{job_id}", {}, api_key)
            except Exception:
                pass


def run(endpoint: str, api_key: str, content: list[dict], max_tokens: int,
        temperature: float, lang: str = "en", model: str = TUNED_MODEL,
        use_system: bool | None = None) -> dict:
    if use_system is None:
        use_system = model != TUNED_MODEL
    messages = ([{"role": "system", "content": system_prompt(lang)}] if use_system else [])
    messages.append({"role": "user", "content": content})
    for status in iter_job(endpoint, api_key, messages, max_tokens, temperature, model):
        if status.get("status") == "COMPLETED":
            return status
        print(f"job {status['id']}: {status.get('status')}", file=sys.stderr)
    raise RuntimeError("RunPod returned no completed result")


def extract_text(result: dict) -> str:
    out = result.get("output")
    # RAW_OPENAI_OUTPUT=true returns the OpenAI response, sometimes wrapped in a
    # single-element list by the async handler.
    if isinstance(out, list) and out:
        out = out[0]
    if isinstance(out, dict) and "choices" in out:
        return out["choices"][0]["message"]["content"]
    raise RuntimeError("RunPod returned no chat completion; check the selected model and worker logs")


def response_metadata(result: dict, expected_model: str) -> dict:
    out = result.get("output")
    if isinstance(out, list) and out:
        out = out[0]
    if not isinstance(out, dict) or not out.get("choices"):
        raise RuntimeError("No chat completion returned; the selected adapter may not be loaded")
    actual = out.get("model")
    if actual != expected_model:
        raise RuntimeError(f"Requested {expected_model}, but response model was {actual!r}")
    return {"model": actual, "finish_reason": out["choices"][0].get("finish_reason"),
            "usage": out.get("usage", {}), "job_id": result.get("id"),
            "execution_ms": result.get("executionTime"), "delay_ms": result.get("delayTime")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+", help="drawing files, in figure order")
    ap.add_argument("--context", help="extra description of the invention")
    ap.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT))
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--model", choices=(SERVED_MODEL, TUNED_MODEL), default=TUNED_MODEL)
    ap.add_argument("--system-prompt", action=argparse.BooleanOptionalAction, default=None)
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

    content = build_content(args.images, args.context, args.lang)
    result = run(args.endpoint, api_key, content, args.max_tokens, args.temperature,
                 args.lang, args.model, args.system_prompt)
    print(json.dumps(response_metadata(result, args.model)), file=sys.stderr)
    text = extract_text(result)

    if args.raw or args.model == 'claim-v3':
        print(text)
        return

    claim, removed = sanitise(text, args.lang)
    if removed:
        print(f"[sanitiser removed: {', '.join(removed)}]", file=sys.stderr)
    print(claim)


if __name__ == "__main__":
    main()
