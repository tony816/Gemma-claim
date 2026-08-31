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
