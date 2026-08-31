# Working in this repository

A previous run fine-tuned `google/gemma-4-31B-it` on 91 patent-claim records,
spent about $21, and produced a model that was worse at the task than the base
model it started from. Everything below is what that cost bought. Read it before
starting a GPU.

## The rule that matters most

**Never use a GPU pod as a development environment.**

Every pipeline bug in the last run was discovered for the first time on a
$1.50-2.00/hr machine, one restart cycle at a time, and the pod stayed alive
between attempts. That is where $10.33 of the $21 went — a pod nobody
terminated. The three most expensive bugs were all catchable on a laptop for
nothing:

| Bug | How it was found | How to find it free |
|---|---|---|
| Chat template prefix violation | restart cycle on the GPU | `AutoProcessor.from_pretrained` — tokenizer only, no weights |
| LoRA targets re-matched the vision tower | restart cycle on the GPU | `init_empty_weights` + `from_config` — meta device, no VRAM, no download |
| `TrainingArguments` signature change | restart cycle on the GPU | `inspect.signature` — pure Python |

So: **run `python tools/rehearse.py` and get a clean result before any GPU is
started.** It exercises the real pipeline against the real processor and a
weightless model skeleton. It reports SKIP loudly for anything it could not
check; a skip is not a pass.

## Before training at all

The last run's model was useless, and the run reported success. Loss fell
3.497 -> 1.821 on validation, which is teacher-forced and says nothing about
free generation. In free generation the model produced a claim about the same
apparatus as the reference in **0 of 12** test records.

The warning was already in the metrics and was under-weighted: **chrF fell**
(28.85 -> 21.04 on test) while loss fell. Loss down + chrF down means the model
learned the unconditional distribution of the target text, not the mapping from
input to output. Full evidence in `run_artifacts/POST_HOC_GENERATION_AUDIT.md`.

Two gates follow from that:

1. **Score the prompted base model first.** `python tools/baseline.py` runs the
   base model over the eval split through the serving endpoint and scores it
   with the same metrics training uses. If prompting already does the job,
   there is nothing to buy with a GPU. That was true last time and nobody
   checked.
2. **Look at generations during training, not after.** Sample 3 free-running
   generations at step 0 and again early in the run. Mode collapse is obvious
   by eye long before the run ends. Never judge a run by loss alone.

Dataset size is the usual cause. 91 records against 410 LoRA modules over 60
layers was far too strong an adaptation for the evidence available. If the next
dataset is a similar size, expect the same outcome and prefer prompting.

## Money

- Pods bill per hour whether or not anything is running. `tools/pod_guard.sh`
  self-terminates a pod at a deadline; put it in the boot path of every pod.
- Serverless bills only while a container is up. An endpoint at `min workers 0`
  costs nothing idle, including disk — measured: zero charges across the hours
  after workers scaled down.
- Network volumes bill monthly regardless of use. Delete them when done, and
  see the `BASE_PATH` trap below before you do.
- `mcp__Runpod__get-billing` with `bucketSize: hour` is the ground truth.
  Do not estimate balances by arithmetic; that was got wrong twice.

## Traps that cost hours

**`BASE_PATH` pointing at a deleted volume.** worker-vllm downloads the model
into `BASE_PATH`, default `/runpod-volume`. When the network volume backing that
path was deleted, every container exited within ~16 seconds, before vLLM wrote a
single log line. It looked like a LoRA problem, then a GPU problem, then a CUDA
problem. It was none of those. If workers go UNHEALTHY with no container output,
check `BASE_PATH` and attached volumes first.

**Gemma 4's generation prompt opens an empty thought channel.**
`add_generation_prompt=True` emits `<|channel>thought\n<channel|>`, which is
absent from the full rendering. Build the training sequence from the real
generation prompt plus the template-rendered body, and assert the prompt is a
prefix of the full sequence. `pipeline/dataset.py:render_texts` does this.

**PEFT matches bare module suffixes everywhere.** `q_proj` also exists in the
vision tower, wrapped in `Gemma4ClippableLinear`, which PEFT cannot adapt. Pass
fully-qualified names. `pipeline/train.py:discover_lora_targets` does this.

**worker-vllm accepts three input shapes and only one honours top-level
parameters.** Use `{"openai_route": ..., "openai_input": {...}}`. The shorthand
`{"messages": ..., "sampling_params": ...}` silently ignores `model` and
`max_tokens` — a round of endpoint tests proved nothing because of this, and
produced a confident wrong conclusion.

**Gradio's API moves.** 6.x dropped `Chatbot(type=)` and
`Textbox(show_copy_button=)`; 5.x needs the former. `space/app.py` probes the
installed signature rather than assuming.

## Constraints on the data

The dataset release is frozen. Do not modify, delete, re-split, or overwrite its
files, targets, images, or splits. Never put `metadata/source_oracle_pages/`,
`canonical/`, `excluded/`, `evaluation/`, original claim pages, oracle judgment
material, or manifest contents into a prompt, into training data, or into any
retrieval context. Model input is `hf_multimodal/{train,validation,test}.jsonl`
plus the images they reference. Preserve the order of the `images` array. Train
loss on assistant response tokens only. Use the test split once, after the
config and checkpoint are frozen.

## Secrets

Never write a Hugging Face token, RunPod API key, or Google Drive credential
into a file, a log, a commit, or a tool call that echoes it back. A RunPod key is
account-wide: it can create and delete pods, not just call an endpoint. An
endpoint's `env` is stored and returned in plaintext by the API — keep keys out
of it where the model is public.

## Layout

```
pipeline/     training pipeline (fetch, preflight, train, evaluate, push, report)
tools/        rehearse.py, baseline.py, pod_guard.sh  <- run these before a GPU
serving/      clients for the deployed endpoint; claim_prompt.py is the one
              definition of the prompt and output sanitiser
space/        the Hugging Face Space (a copy of claim_prompt.py, kept in sync
              by space/deploy.sh)
tests/        offline pipeline tests with a fake processor
run_artifacts/ what the completed run actually produced, including the audit
```

Develop on the branch named in the task. Do not open a pull request unless
asked.
