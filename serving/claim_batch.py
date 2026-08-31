#!/usr/bin/env python3
"""
Gemma 4 31B 배치 호출기  ·  RunPod / DeepInfra

여러 건의 프롬프트를 한 번에 던지고, 결과와 비용을 정리해 줍니다.
대화형이 아니라 한 번에 몰아서 처리할 때 쓰세요. 대화는 gemma_chat.py.

준비 (최초 1회):
    pip install openai

    # RunPod 사용 시 (기본값)
    export RUNPOD_API_KEY='런팟_키'
    export RUNPOD_ENDPOINT_ID='fdiltabt78bogm'

    # DeepInfra 사용 시
    export DEEPINFRA_TOKEN='발급받은_키'

사용:
    python gemma_batch.py prompts.txt              # 한 줄에 프롬프트 하나
    python gemma_batch.py prompts.txt --provider deepinfra
    python gemma_batch.py prompts.txt -o 결과.jsonl
    python gemma_batch.py prompts.txt --system "당신은 특허 전문가입니다."
    python gemma_batch.py prompts.txt --dry-run    # 호출 없이 비용만 추정

RunPod은 시간 과금이라 --dry-run 의 토큰 단가 표는 DeepInfra 기준입니다.
RunPod에서는 실행 후 GPU 가동 시간 기준 비용이 따로 표시됩니다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# DeepInfra 공시가 (USD per 1M tokens). 요금제가 바뀌면 여기만 고치세요.
PRICING = {
    "flex":     {"in": 0.104, "out": 0.304},   # 0.8x — 급하지 않은 배치용. 기본값.
    "standard": {"in": 0.130, "out": 0.380},
    "priority": {"in": 0.195, "out": 0.570},   # 1.5x
}

RUNPOD_USD_PER_HOUR = 1.22   # AMPERE_48(A40) 단일 풀. 시간 과금이라 토큰 단가와 무관.
DEFAULT_ENDPOINT = "fdiltabt78bogm"
USD_TO_KRW = 1380            # 대략치. 감만 잡는 용도.


def resolve(provider: str) -> tuple[str, str, str]:
    """(base_url, model, api_key) 반환. 준비가 안 됐으면 안내하고 종료."""
    if provider == "runpod":
        eid = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
        if not eid:
            sys.exit("RUNPOD_ENDPOINT_ID 환경변수가 없습니다. 예: export RUNPOD_ENDPOINT_ID='fdiltabt78bogm'")
        key = os.environ.get("RUNPOD_API_KEY", "").strip()
        if not key:
            sys.exit("RUNPOD_API_KEY 환경변수가 없습니다.")
        return f"https://api.runpod.ai/v2/{eid}/openai/v1", "gemma4-31b", key

    key = os.environ.get("DEEPINFRA_TOKEN", "").strip()
    if not key:
        sys.exit(
            "DEEPINFRA_TOKEN 환경변수가 없습니다.\n"
            "  deepinfra.com 대시보드에서 키를 발급받아 아래처럼 설정하세요.\n"
            "    export DEEPINFRA_TOKEN='...'      (macOS/Linux)\n"
            "    set DEEPINFRA_TOKEN=...           (Windows)"
        )
    return "https://api.deepinfra.com/v1/openai", "google/gemma-4-31B-it", key



@dataclass
class Result:
    index: int
    prompt: str
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    error: str | None = None
    attempts: int = 1


@dataclass
class Summary:
    results: list[Result] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def ok(self) -> list[Result]:
        return [r for r in self.results if r.error is None]

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if r.error is not None]

    def cost(self, tier: str) -> float:
        p = PRICING[tier]
        tin = sum(r.prompt_tokens for r in self.ok)
        tout = sum(r.completion_tokens for r in self.ok)
        return tin / 1e6 * p["in"] + tout / 1e6 * p["out"]


def build_client(token: str, base_url: str, timeout: float):
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai 패키지가 없습니다.  pip install openai  을 먼저 실행하세요.")
    return OpenAI(api_key=token, base_url=base_url, max_retries=0, timeout=timeout)


def one_call(client, model: str, index: int, prompt: str, system: str | None,
             max_tokens: int, temperature: float, retries: int) -> Result:
    """프롬프트 한 건. 실패하면 지수 백오프로 재시도."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_err = None
    for attempt in range(1, retries + 2):
        started = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            usage = resp.usage
            return Result(
                index=index,
                prompt=prompt,
                text=resp.choices[0].message.content or "",
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                seconds=round(time.monotonic() - started, 2),
                attempts=attempt,
            )
        except Exception as exc:  # noqa: BLE001 - 어떤 예외든 재시도 대상
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt <= retries:
                time.sleep(min(2 ** attempt, 20))

    return Result(index=index, prompt=prompt, error=last_err, attempts=retries + 1)


def run_batch(client, model: str, prompts: list[str], system: str | None, workers: int,
              max_tokens: int, temperature: float, retries: int) -> Summary:
    summary = Summary()
    started = time.monotonic()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(one_call, client, model, i, p, system, max_tokens, temperature, retries): i
            for i, p in enumerate(prompts)
        }
        for fut in as_completed(futures):
            summary.results.append(fut.result())
            done += 1
            print(f"\r  진행 {done}/{len(prompts)}", end="", flush=True)

    print()
    summary.results.sort(key=lambda r: r.index)
    summary.wall_seconds = round(time.monotonic() - started, 2)
    return summary


def estimate(prompts: list[str], max_tokens: int) -> None:
    """호출 없이 대략의 비용만. 토큰 수는 글자수/2.5로 거칠게 잡습니다."""
    tin = sum(max(1, int(len(p) / 2.5)) for p in prompts)
    tout = len(prompts) * max_tokens
    print(f"\n프롬프트 {len(prompts)}건 · 입력 약 {tin:,} tok · 출력 최대 {tout:,} tok")
    print(f"{'요금제':<10}{'예상 비용':>14}{'원화':>12}")
    print("-" * 36)
    for tier, p in PRICING.items():
        usd = tin / 1e6 * p["in"] + tout / 1e6 * p["out"]
        print(f"{tier:<10}{'$' + format(usd, '.4f'):>14}{format(usd * USD_TO_KRW, ',.0f') + '원':>12}")
    print("\n출력은 상한 기준이라 실제로는 이보다 적게 나옵니다.")


def report(summary: Summary, tier: str, runpod_usd: float | None = None) -> None:
    ok, failed = summary.ok, summary.failed
    tin = sum(r.prompt_tokens for r in ok)
    tout = sum(r.completion_tokens for r in ok)
    usd = summary.cost(tier)
    retried = [r for r in ok if r.attempts > 1]

    print("\n" + "=" * 46)
    print(f"  성공 {len(ok)}건 / 실패 {len(failed)}건")
    if retried:
        print(f"  재시도 후 성공: {len(retried)}건")
    print(f"  전체 소요      : {summary.wall_seconds}초")
    if ok:
        slowest = max(r.seconds for r in ok)
        print(f"  가장 느린 1건  : {slowest}초")
        print(f"  토큰           : 입력 {tin:,} · 출력 {tout:,}")
        if runpod_usd is not None:
            print(f"  GPU 시간 비용  : ${runpod_usd:.3f}  (약 {runpod_usd * USD_TO_KRW:,.0f}원)")
            print(f"  {'':14} + 워커 유휴 대기 10분이 추가로 과금됩니다")
        else:
            print(f"  비용({tier})  : ${usd:.4f}  (약 {usd * USD_TO_KRW:,.0f}원)")
    print("=" * 46)

    for r in failed:
        print(f"  [실패] #{r.index + 1} {r.prompt[:40]}... → {r.error}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gemma 4 31B 배치 호출 (RunPod / DeepInfra)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("prompts_file", help="프롬프트 파일 (한 줄에 하나, 빈 줄과 # 주석은 무시)")
    ap.add_argument("--provider", choices=["runpod", "deepinfra"], default="runpod",
                    help="어디로 보낼지 (기본: runpod)")
    ap.add_argument("-o", "--out", default="results.jsonl", help="결과 저장 경로")
    ap.add_argument("--system", default=None, help="모든 요청에 붙일 시스템 프롬프트")
    ap.add_argument("--tier", choices=list(PRICING), default="flex", help="요금제 (비용 계산용)")
    ap.add_argument("--workers", type=int, default=8, help="동시 요청 수")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--retries", type=int, default=2, help="건당 재시도 횟수")
    ap.add_argument("--dry-run", action="store_true", help="호출 없이 비용만 추정")
    args = ap.parse_args()

    path = Path(args.prompts_file)
    if not path.exists():
        sys.exit(f"파일이 없습니다: {path}")

    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        sys.exit("프롬프트가 하나도 없습니다.")

    if args.dry_run:
        estimate(prompts, args.max_tokens)
        return

    base_url, model, token = resolve(args.provider)
    is_runpod = args.provider == "runpod"
    client = build_client(token, base_url, 900.0 if is_runpod else 180.0)

    print(f"\n{model} · {args.provider} · {len(prompts)}건 · 동시 {args.workers}")
    if is_runpod:
        print(f"{'':2}첫 요청은 모델을 올리느라 5~6분 걸립니다.")
    print()
    summary = run_batch(
        client, model, prompts, args.system, args.workers,
        args.max_tokens, args.temperature, args.retries,
    )

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        for r in summary.results:
            fh.write(json.dumps({
                "index": r.index,
                "prompt": r.prompt,
                "response": r.text,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "seconds": r.seconds,
                "attempts": r.attempts,
                "error": r.error,
            }, ensure_ascii=False) + "\n")

    if is_runpod:
        gpu_usd = summary.wall_seconds / 3600 * RUNPOD_USD_PER_HOUR
        report(summary, args.tier, runpod_usd=gpu_usd)
    else:
        report(summary, args.tier)
    print(f"\n결과 저장: {out.resolve()}")


if __name__ == "__main__":
    main()
