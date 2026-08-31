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
| idle timeout | 60 s |
| context | 16384 tokens |

Scaling to zero is the whole point: you are billed only while a worker is up.
A cold start pays for the model download and load; back-to-back requests within
the idle window reuse the warm worker.

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
