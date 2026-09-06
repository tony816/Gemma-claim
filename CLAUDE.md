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

All of it runs on the laptop. The `WinError 4551` torch failure recorded in
older notes was OneDrive; the venv now lives at `Projects\Gemma-claim\.venv`
outside it and imports fine. Two local quirks: export `PYTHONIOENCODING=utf-8`
or the console's cp949 kills `rehearse.py` on the em dash in its own banner, and
`WORKSPACE` defaults to `/workspace`, which on Windows is `C:\workspace` — set
`OUT_DIR` rather than discovering that later.

**A gate that has never executed is not yet a gate.** On 2026-09-06
`rehearse.py` ran for the first time and failed the assistant-only-loss gate.
The pipeline was correct; the gate was wrong — it enumerated a `(1, seq)` labels
tensor along the batch axis and concluded token 0 was supervised. A gate that
can raise a false alarm can also return a false pass, so when a never-run check
fires, verify the check before believing it. The same applies to
`tests/make_synthetic.py`, which was still emitting the previous release's
91/11/12 shape and a `metadata` key that the oracle-field whitelist rejects.

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
2. **Look at generations during training, not after.** `pipeline/train.py` has
   a `GenerationSampler` callback that does this: three free-running
   generations at step 0, at `SAMPLE_STEPS`, and at every epoch-end evaluation,
   each logged with its chrF and its reference. `rehearse.py` checks the wiring
   is still there. Step 0 arrives minutes into a run, when killing it is cheap.

**Read those samples per record, never as a mean.** On 2026-09-06 the epoch-1
mean chrF was 9.57 against a base of 8.42 — an improvement on paper. Underneath,
one record had gone 4.63 -> 16.87 and two had collapsed below base into
enumeration templates (`제1 유로 … 제15 유로`, a coined word repeated six times).
The mean was the good record carrying the other two.

**Write the stop criteria down before the run reaches them**, in the form "at
epoch N, if X or Y then stop". Deciding what counts as bad while watching a
$3.59/hr pod produce ambiguous samples is not a decision, it is a rationalisation.

**Repetition metrics have a blind spot worth knowing.** N-gram duplication does
not fire on `제1 유로에 연결된 제2 유로; 상기 제2 유로에 연결된 제3 유로; …`
because the numeral makes every n-gram unique. That output is degenerate and the
metric scores it 0.00. Normalise digits before counting, or read the text.

Dataset size is the usual cause. 91 records against 410 LoRA modules over 60
layers was far too strong an adaptation for the evidence available. If the next
dataset is a similar size, expect the same outcome and prefer prompting.

## Measuring, after the 694-record run (2026-09-06)

**`baseline.py` and `evaluate.py` do not measure the same thing, and their
numbers must not be put in one table.** `baseline.py` sends the engineered
system prompt from `serving/claim_prompt.py`; that is "what prompting alone
achieves", and on this dataset it is chrF 9.59 on the 59 Korean validation
records, with 58 of 59 well-formed. `evaluate.py` uses the dataset's own
messages with no system turn, the same input training saw, and switches the
adapter off for its base numbers — that is a clean A/B of the adapter, and its
base model rambles to 250-350 words because nothing constrains it.

**A metric written for English will silently report catastrophe on Korean.**
`claim_form_checks` keyed on `device|apparatus|comprising`, none of which occur
in a Korean claim, so it reported `well_formed_rate` 0.09 and "59 of 59 missing
an apparatus noun". With Korean rules the same predictions score 0.98. 612 of
694 targets are Korean; check every metric against the language it will meet.

**The training prompt is not the serving prompt.** Training records carry no
system turn at all — one fixed sentence (Korean for Korean targets, English for
English) plus the drawings. Sending the serving system prompt to a fine-tuned
model is therefore an instruction it has never seen in training, and it can be
ignored or can push the model off the distribution its LoRA deltas were fit on.
`tools/prompt_probe.py` measures both conditions on the tuned model; run it
while a pod holding the adapter is still alive, before deciding how to serve.

**Anything in the target becomes model behaviour.** 97 of 694 targets carry
drawing reference numerals (93 Korean, 15.2%), and the fine-tuned model writes
them into claims, where they are never acceptable. The release is frozen; the
fix is a dataset-side requirement for the next release, recorded in
`patent-dataset-factory/governance/DECISION_LOG.md`. Meanwhile
`serving/claim_prompt.py` strips them — that rule had never fired before,
because the base model does not write numerals in Korean and the fine-tuned one
does.

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
- **Evaluation costs more than training.** Measured on 2026-09-06: 417 optimiser
  steps over 554 records took ~50 minutes, while scoring the 65 validation
  records for base and tuned — 130 free-running generations — took longer than
  the training did. `MAX_NEW_TOKENS` drives it, and the base model runs to the
  limit because nothing stops it. Set `EVAL_SPLITS=validation` first and spend
  the test split only on a run that survives it; `run_all.sh` takes both from the
  environment now.
- Set `MAX_NEW_TOKENS` from the eval splits, not from `token_audit`'s
  recommendation. That number is derived from the longest *train* target, which
  is never generated: it came out 2401 while the longest validation reference is
  513 tokens and the longest test reference 894, so 1024 covers every reference
  the evaluator can be asked to reproduce.

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
into a file, a log, a commit, or a tool call that echoes it back.

Passing them to a pod is the awkward case: RunPod stores and returns pod `env`
in plaintext, so they cannot go there. Generate the env file locally and pipe it
over stdin — `python make_pod_env.py | ssh POD 'umask 077 && cat > pod.env'` —
which keeps the value out of every command line. Then do not `cat` or `tail`
that file to check your work; on 2026-09-06 a `tail -3 pod.env` printed the
Hugging Face token into the transcript and it had to be rotated. A RunPod key is
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
