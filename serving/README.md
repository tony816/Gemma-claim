# Serving

Draft independent patent claims from drawings, on a RunPod Serverless endpoint.

## What is deployed, and why it is the base model

The fine-tuned adapter is **not** served. `Mepeng22/gemma-4-31b-claim-lora`
learned the output format completely and lost image grounding entirely: across
the 12 test records it produced a claim about the same apparatus as the
reference **0 times**, collapsed onto a handful of memorised openings, and gave
three different claims of the same patent substantially the same answer. Full
evidence in [`../run_artifacts/POST_HOC_GENERATION_AUDIT.md`](../run_artifacts/POST_HOC_GENERATION_AUDIT.md).

The base model reads the drawings correctly — it cites figure numerals and picks
up context visible only in the images. What it lacks is the output format: it
answers with a markdown preamble, a bolded `Claim 1:` heading, and a trailing
`Drafting Notes` section. That is a prompting problem, not a training one, so
`claim_client.py` fixes it with a system prompt plus a narrow sanitiser.

## Endpoint

| | |
|---|---|
| id | `fdiltabt78bogm` |
| name | `gemma4-31b-claim` |
| model | `cyankiwi/gemma-4-31B-it-qat-AWQ-INT4` (public, ~18 GB) |
| GPU pool | `AMPERE_48` (A40, 48 GB) |
| workers | min 0, max 1 — **no charge when idle** |
| idle timeout | 300 s |
| context | 16384 tokens |
| KV cache | 19.0 GiB (53,218 tokens) |

Scaling to zero is the whole point: you are billed only while a worker is up.
A cold start pays for the model download and load; back-to-back requests within
the idle window reuse the warm worker.

Measured on an A40: cold start **~290 s** (19-30 s to download 19.5 GiB, 34-48 s
to load, 70 s of torch.compile, 65 s of multimodal warmup), then generation in
**5-7 s**. The 300 s idle timeout is set so a run of test requests reuses one
warm worker instead of paying that cold start each time; the trailing idle
window costs about $0.10 at the A40 serverless rate. Lower it if you are making
one request at a time and would rather wait than pay.

## Verified

Both paths were exercised end to end on this endpoint.

**Format** — text-only request, no sanitiser intervention needed:

> A diagnostic cartridge comprising a body having an interior surface and an
> exterior surface, wherein a border between the interior surface and the
> exterior surface is configured to form a capillary gap when the body is
> inserted into a slot of an analytical instrument, the capillary gap being
> narrower than any adjacent space within the slot such that liquid is retained
> at an edge of the slot.

**Image grounding** — two synthetic drawings, sent in order. Figure 1 showed a
housing numbered 10 containing components 20 and 22 joined by a channel, with
ports 12 and 14; figure 2 showed plates 10 and 16 separated by a hatched gap 30.
The model returned every numeral mapped to the right structure: "a housing 10 …
a first component 20 and a second component 22 … connected", "an input interface
12", "an output interface 14", "a coupling member 30 … a gap is formed between
the housing 10 and the second housing 16". Prompt tokens rose from 242 to 725,
confirming both images were encoded.

That run also exposed the one prompt-compliance gap: told not to use reference
numerals *in parentheses*, the model wrote them bare. The instruction now
forbids them outright and the sanitiser strips a bare integer where a numeral
can appear — before punctuation or a structural word — while leaving quantities
like "0.1 ml" and "100 pL" alone, since those are followed by a unit.

## Use

```bash
export RUNPOD_API_KEY=...          # from the RunPod console; never commit it
python serving/claim_client.py fig1.png fig2.png fig3.png
```

Pass the drawings **in figure order** — later figures are read relative to
earlier ones, and the order is preserved end to end. Add `--context "..."` to
supply anything the drawings do not show. `--raw` prints the model's output
without the sanitiser, which is what you want when checking prompt compliance.

## Request shape — the part that is easy to get wrong

worker-vllm accepts three input shapes. Only the OpenAI passthrough honours
top-level `model` and `max_tokens`:

```json
{"input": {"openai_route": "/v1/chat/completions",
           "openai_input": {"model": "gemma4-31b", "messages": [...],
                            "max_tokens": 500, "temperature": 0}}}
```

The shorthand `{"messages": ..., "sampling_params": ...}` silently ignores both,
which is what made an earlier round of endpoint tests meaningless — a request
naming a different model came back served by the default one, and a 96-token cap
produced 687 tokens.

Images go in `openai_input.messages[].content` as `image_url` parts with
`data:image/png;base64,...` URLs.

## If workers go UNHEALTHY

Check `BASE_PATH` first. worker-vllm downloads the model into `BASE_PATH`, which
defaults to `/runpod-volume`. If that path is a network volume that has since
been deleted, every container exits within about 16 seconds, before vLLM writes
a single log line — the failure looks like a model or GPU problem and is neither.
This endpoint sets `HF_HOME=/tmp/hf` and attaches no network volume, so it has
nothing to lose.

RunPod's API does not surface container stdout for these workers, only system
lines. Repeated `start container` entries a few seconds apart, with no container
output, is the crash-loop signature.

## 대화형 클라이언트

`tony816/Gemma` 의 `gemma_chat.py` / `gemma_batch.py` 를 이 프로젝트에 맞춰
가져온 것이 `claim_chat.py` / `claim_batch.py` 입니다. 스트리밍, 대화 기억,
저장·불러오기, 비용 누적 표시는 원본 그대로입니다.

```bash
pip install openai
export RUNPOD_API_KEY='런팟_키'
python serving/claim_chat.py
```

`RUNPOD_ENDPOINT_ID` 는 생략하면 이 프로젝트의 엔드포인트가 기본값입니다.

### 원본에서 바뀐 것

| | 원본 | 여기 |
|---|---|---|
| 엔드포인트 | `e5yibbozs40ji4` (삭제됨) | `fdiltabt78bogm` |
| GPU 단가 | $1.75/hr (두 풀 중 비싼 쪽) | $1.22/hr (AMPERE_48 단일 풀) |
| 유휴 타임아웃 | 600 s | 300 s |
| 히스토리 | 30턴 | 12턴 (컨텍스트가 32K → 16K) |
| 시스템 프롬프트 | 없음 | 청구항 작성용이 기본 |
| `/image` | 1장 | 여러 장, figure 순서 유지 |
| temperature | 0.7 고정 | 청구항 모드 0.2 / 일반 0.7 |

시스템 프롬프트와 sanitiser는 `claim_client.py` 에서 그대로 가져다 씁니다.
같은 문구를 두 군데 두지 않으려는 것이고, 그쪽을 고치면 대화창에도 반영됩니다.

```bash
python serving/claim_chat.py --plain                 # 청구항 프롬프트 없이
python serving/claim_chat.py --system "..."          # 다른 프롬프트로
python serving/claim_chat.py --temperature 0.5
```

대화 중:

```
/image 도면1.png 도면2.png 도면3.png
/image 도면1.png 이 도면의 구성요소를 청구항 1의 한정과 대응시켜줘
```

도면은 **적은 순서 그대로** 전달됩니다. 답변이 형식을 어기면(마크다운, 도면
부호 등) 원문 아래에 다듬은 결과를 함께 보여줍니다. 히스토리에는 모델이 실제로
낸 말을 남기므로 다음 턴의 맥락은 어긋나지 않습니다.

### 배치

`claim_batch.py` 는 프롬프트를 여러 건 동시에 던져 JSONL로 받습니다.
**텍스트 전용**이라 도면은 다루지 못합니다. 도면이 필요하면 `claim_chat.py`
또는 `claim_client.py` 를 쓰세요.

```bash
python serving/claim_batch.py prompts.txt --dry-run   # 호출 없이 비용만
python serving/claim_batch.py prompts.txt -o 결과.jsonl
```

### 한 가지 미검증 사항

이 클라이언트는 `https://api.runpod.ai/v2/{id}/openai/v1` (OpenAI 호환 경로)로
붙습니다. 이 경로는 worker-vllm의 표준 인터페이스이고, 핸들러 쪽 OpenAI
패스스루는 `claim_client.py` 로 실측 확인했습니다. 다만 RunPod 게이트웨이의
`/openai/v1` 경로 자체는 API 키가 없어 직접 호출해 보지 못했습니다.

첫 실행이 곧 검증입니다. 만약 404나 인증 오류가 나면 `claim_client.py` 가
검증된 경로(`/run` + `/status`)를 쓰므로 그쪽으로 대체하면 됩니다.
