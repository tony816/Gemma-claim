# Handoff — continuing the claim generator on the 694-record release

Written 2026-09-02 at the end of a local session, for whoever picks this up in a
cloud session. Read `CLAUDE.md` first; it is the record of what the previous run
cost and why. This file is only the delta since then.

## What the user decided

1. **A new dataset is ready** and replaces `v1.1.2-independent-oracle-clean`:
   694 records, 612 of them Korean. It has been verified (below) and converted.
2. **Continue training the existing adapter**, not a fresh LoRA. The user was
   told the previous adapter is mode-collapsed (0/12 apparatus match, chrF
   28.85 -> 21.04, mean length 369.6 -> 158.1 words) and that starting from the
   base is the safer call. They reaffirmed continuing from the adapter. Do not
   re-litigate it; `RESUME_ADAPTER` exists for exactly this.
3. **No pod today.** The pod runs tomorrow, after the free gates pass.

## Where things stand

| Step | Status |
|---|---|
| 1. Whitelist conversion adapter | done — `tools/convert_release.py`, ran clean |
| 2. preflight counts + oracle-field gate | done — passes, and fails correctly on a poisoned package |
| Resume-from-adapter path in `train.py` | done — `RESUME_ADAPTER` |
| Language-aware baseline scoring | done — was scoring 612 Korean records with the English prompt |
| 3. `token_audit.py` | **blocked** — needs the real tokenizer (see egress policy below) |
| 4. `rehearse.py` clean | **5 of 7 gates pass**; 2 skip, neither reachable from this environment |
| 5. `baseline.py` | **blocked** — `api.runpod.ai` is not reachable from this environment |

## The torch blocker is gone; a different one replaced it

The Windows application-control policy that blocked PyTorch's DLLs
(`OSError: [WinError 4551]` on `torch\lib\shm.dll`) does not occur in a cloud
session. `pip install torch` then `pip install -r requirements.txt` succeeds —
with one snag worth remembering: the base image ships a Debian-managed PyYAML
with no RECORD file, so the requirements install aborts on
`Cannot uninstall PyYAML 6.0.1`. Use `pip install --ignore-installed PyYAML -r
requirements.txt`.

Verified working set: torch 2.13.0+cu130, transformers 5.16.1, peft 0.20.0
(the same peft that wrote the adapter), accelerate 1.14.0.

**What blocks the rest is the environment's egress policy, not the code.**
Reachable: `pypi.org`, `files.pythonhosted.org`, `github.com`, `api.github.com`.
Refused with a 403 at the proxy: `huggingface.co`, `api.runpod.ai`,
`api.runpod.io`, and everything else. So `AutoProcessor.from_pretrained` and
`baseline.py` cannot run here as written. Two ways forward:

* **Allow `huggingface.co` and `api.runpod.ai`** in the environment's network
  policy (see the Claude Code on the web docs on environments). That is the
  clean fix and makes every gate runnable off-GPU, which is the whole point of
  running them before the pod.
* **Offline assets** — what was actually done this session. Three small files
  (`config.json` 4,621 B, `chat_template.jinja` 18,683 B, `tokenizer_config.json`)
  are enough to run the template and LoRA gates with no Hub and no weights. Put
  them in a directory and point `BASE_MODEL` or `OFFLINE_ASSETS` at it. They are
  gated Google files: keep them out of the repository.

### What rehearse.py reports now

With offline assets and a package present, on transformers 5.16.1:

```
[  ok  ] versions
[  ok  ] TrainingArguments signature — all 17 essential args accepted on 5.16.1
[  ok  ] chat template renders a strict prefix — template-only processor (tokeniser NOT checked)
[ skip ] loss is masked to the assistant span — needs the real tokenizer
[  ok  ] LoRA targets avoid the vision tower — 410 modules from the real config
[ skip ] resume adapter — needs the adapter repo (RESUME_ADAPTER, Hub-gated)
[  ok  ] dataset contracts
```

Two of the three expensive bugs from `CLAUDE.md` are now positively cleared on
the current library set:

* **Chat template.** The real template renders the generation prompt as the
  conversation prefix plus exactly `<|channel>thought\n<channel|>`, which the
  full rendering does not contain — so `render_texts`'s reconciliation is live,
  not dead code. Checked on 1/2/5/8-image records, English and Korean, with the
  Korean target preserved byte-for-byte through the template.
* **LoRA targets.** `discover_lora_targets` on the real config finds exactly
  **410** modules — the same count the adapter carries — all fully qualified,
  none in the vision tower. The model holds 189 `Gemma4ClippableLinear`
  projections with those same suffixes; a bare-suffix target list would match
  them and PEFT cannot adapt them. The 410 is 60x7 minus 10: the `full_attention`
  layers (5, 11, ..., 59) have no separate `v_proj`, because `attention_k_eq_v`
  is true.

The `resume adapter` gate could not run, but its load-bearing fact was confirmed
by reading `adapter_config.json` directly: **`inference_mode` is `true`**, and
`base_model_name_or_path` is `google/gemma-4-31B-it`, so `is_trainable=True`
stays mandatory.

## The data — this is the part that does not travel with the repo

The dataset is **not in this repository and not in the cloud**. Two artefacts
exist on the user's laptop and one of them has to be transferred:

- **Approved release** (read-only, frozen, 5,482 files):
  `C:\Users\VIEW LIFW\Documents\Codex\patent-dataset-factory\outputs\releases\finetune_multilingual_approved_20260902`
- **Converted package** (what the pipeline actually reads, 1.1 GB):
  `C:\Users\VIEW LIFW\OneDrive\Desktop\Gemma-claim-data\v2_approved_20260902`

Transferring the converted package is enough. Ask the user how they want to move
it; do not upload patent material anywhere without their explicit go-ahead.

If you re-run the conversion instead of transferring it, the output must match:

```
train.jsonl       69c780015eb37da06257cb33795963d77db5d8b60fd0602620ab18b792b561d3
validation.jsonl  69c28ae4b5205a9a13ed5cec2081231936e991a123100f8264c6b54e110d2db0
test.jsonl        db5533a8c06cd97fff8b953b15c2f5b9da36cff631947479d69182c0f366fe3c
```

```bash
python tools/convert_release.py --release <release dir> --out <package dir>
```

### What the release actually contains

Independently verified this session, all of it matching the dataset-building
agent's report:

- 694 records; splits 554 / 65 / 75, ids disjoint, `split` field agrees with the file
- canonical SHA-256 `ac0ce42589d3c06d689ad8ff00075f2ed9018d12bf21246c6e3133cb796401c2`
- zero leakage across splits on family, publication, target, PDF **and image**
- 3,893 unique images, every one present and hash-matching `manifests/file_manifest.jsonl`
- Korean targets are clean UTF-8 NFC; 612 Korean / 82 non-Korean (81 US, 1 EP)
- targets 80–3,475 characters; 284 records (41%) sit at the 8-image cap

### The trap in the release

Each row carries 66 fields, and `canonical_source_claim_transcription` is
**byte-identical to the target**. `semantic_evidence`, `evidence_pages`,
`source_oracle_*` and `source_boundary_next_claim_clean` are oracle material
too. Passing a row through unfiltered puts the answer in the prompt.

`tools/convert_release.py` projects each row down to `{id, images, messages}` by
whitelist. `pipeline/preflight.py` re-checks that projection at the point of use
and dies `ORACLE_FIELD_IN_MODEL_INPUT` on anything else — verified by feeding it
a deliberately poisoned package. Do not weaken either check.

## The adapter being continued

`Mepeng22/gemma-4-31b-claim-lora` (private, 489 MB), base `google/gemma-4-31B-it`,
r=16, alpha=32, dropout 0.05, fully-qualified target module names, peft 0.20.0.

**Its `adapter_config.json` has `inference_mode: true`.** Loading it without
`is_trainable=True` leaves every LoRA parameter frozen, and the run completes
with a flat loss curve and no error anywhere. `pipeline/train.py` passes the flag
and then dies `NO_TRAINABLE_PARAMETERS` if the trainable count is still zero.
`tools/rehearse.py` has a gate for this that runs when `RESUME_ADAPTER` is set.

Enable the resume path with:

```bash
export RESUME_ADAPTER=Mepeng22/gemma-4-31b-claim-lora
```

When set, the adapter's own rank and target modules win; `LORA_R`, `LORA_ALPHA`
and `LORA_DROPOUT` are ignored. `train.py` hard-fails if the adapter's base model
differs from `BASE_MODEL` or if any of its target modules is not an adaptable
language-tower Linear here.

## Run these, in this order, before any GPU

```bash
export DATA_ROOT=<converted package dir>
export OUT_DIR=<somewhere outside the dataset>
export RESUME_ADAPTER=Mepeng22/gemma-4-31b-claim-lora
```

```bash
python -m pip install torch && python -m pip install -r requirements.txt
```

```bash
PYTHONPATH=pipeline python pipeline/preflight.py
```

```bash
PYTHONPATH=pipeline python pipeline/token_audit.py
```

```bash
python tools/rehearse.py
```

If `huggingface.co` is not reachable, set `OFFLINE_ASSETS` (or point
`BASE_MODEL`) at a directory holding `config.json` and `chat_template.jinja`;
the template and LoRA gates then run without the Hub. A skip is still not a
pass — the tokenizer-level gates need the real processor either way.

```bash
python tools/baseline.py --split validation
```

`preflight.py` already passes on the converted package. `token_audit.py` and
`rehearse.py` have never run — they are the two gates the local machine could not
execute, and **a SKIP is not a PASS**. `baseline.py` is the decision point: it
scores the prompted base model through the serving endpoint, and the fine-tune is
worth its GPU time only if it beats those numbers on free-running generation.

### What token_audit is expected to reveal

Nobody has measured the sequence lengths yet. The earlier estimate assumed 256
tokens per image; **the real number is 280** — `config.json` carries
`vision_soft_tokens_per_image: 280` and `processor_config.json` carries
`image_seq_length: 280`. Every image-token estimate scales by 280/256 = 1.094:

- image tokens alone: mean 5.61 images -> 1,571; at the 8-image cap -> 2,240
- the earlier "~1,900 average / ~5,000 worst case" becomes roughly
  ~2,080 / ~5,470 on the same assumptions
- the previous release topped out at 2,362 tokens, so expect 1.3–1.6x

These are still estimates. Only `token_audit.py` against the real tokenizer
settles `recommended_max_new_tokens`, and that number drives most of the run's
cost.

Two things depend on the real number: whether any target is truncated (a hard
failure), and `recommended_max_new_tokens`, which `evaluate.py` uses and which
drives most of the run's cost.

### What baseline is expected to reveal

`baseline.py` now picks the system prompt per record from the reference language
and reports metrics split by language, because 88% of the data is Korean and the
tool was previously sending the English prompt to every record. It also reports
`language_drift` — records answered in the wrong language.

Note the endpoint `fdiltabt78bogm` serves `cyankiwi/gemma-4-31B-it-qat-AWQ-INT4`,
an **INT4** build, while training is bf16. The baseline is therefore slightly
pessimistic. That is the safe direction, but say so when reporting the number.

## Cost and time, from the previous run's measured figures

Ground truth: 91 records, 115 optimiser steps, 610.3 s wall clock on an H200,
peak VRAM 70.03 GiB. Billing history shows the previous run cost $19.91 in pods,
$10.33 of which was a single pod nobody terminated.

Current H200 SXM pricing: community $3.59/hr, secure $4.59/hr, serverless
$5.93/hr. Availability was **LOW** in both pools when checked.

Scaled to 554 records at 6 epochs = 831 optimiser steps:

| Stage | Time | Cost @ $3.59/hr |
|---|---|---|
| Boot + 62.5 GB weight download | 0.3–0.5 h | $1–2 |
| LoRA training | 1.6–2.0 h | $6–7 |
| Evaluation, 280 generations, `MAX_NEW_TOKENS=1024` | 2.5–3.5 h | $9–13 |
| Adapter push | 0.2 h | $1 |
| **Total** | **4.6–6.2 h** | **$17–23** |

**Evaluation costs more than training.** `evaluate.py` generates over every
record of every eval split for both base and tuned: (65 + 75) x 2 = 280
generations, against 23 last time. Set `MAX_NEW_TOKENS` explicitly — left to the
token audit it may resolve to ~3,000, and 280 generations that run to that limit
without hitting EOS is 11+ hours.

Cheapest sane configuration — `EVAL_SPLITS=validation` only, `MAX_NEW_TOKENS=1024`,
adapter push without the merged model — is about 3.0–3.5 h and $11–14. Spend the
test split once, after the config and checkpoint are frozen.

`tools/baseline.py` is separate and serverless: roughly $2–5 for 65 records. It
is the only thing standing between the user and another $20 spent on a run that
loses to prompting.

## Credentials

Two are needed; neither is in this repository and neither should ever be written
into a file, a log, or a commit.

- **Hugging Face**, with access to gated `google/gemma-4-31B-it` and to the
  private adapter repo. `hf auth login` stores it where every process finds it.
- **`RUNPOD_API_KEY`**, for `baseline.py` and for pod management. Account-wide:
  it can create and delete pods, not just call an endpoint.

The user set both on the local machine; they will need setting again in the cloud
environment.

## Still open

- ~~`fetch_dataset.py` still pins the old ZIP's SHA-256.~~ **Done.** It now
  skips the download entirely when a package is already under `DATA_ROOT` (the
  path a transferred package takes), and otherwise refuses with
  `DATASET_PIN_SUPERSEDED` rather than fetching v1.1.2 over the new release.
  `run_all.sh` treats that as non-retryable, so it no longer sits in a
  30-minute retry loop on a pod that bills by the hour. To use a ZIP instead,
  pin it with `DATASET_SHA256` and set `DATASET_URL` / `DRIVE_FILE_ID`.
- ~~`evaluate.py` has no per-language breakdown.~~ **Done.** It reports
  `by_language` and `language_drift` per split for both base and tuned, in the
  same shape `tools/baseline.py` emits, and `tests/test_pipeline.py` [10] fails
  if the two shapes drift apart.
- ~~The offline test suite was red and nobody had run it.~~ **Fixed.** Two
  regressions came in with the 694-record switch: `tests/make_synthetic.py`
  still built the superseded 91/11/12 split, so `preflight.py` died on the
  count gate — and because it died there, checks [7]–[9] were "passing" on the
  wrong failure and never exercised the missing-image, empty-target and
  cross-split gates they name. The fixture also carried a `metadata` key, which
  the new whitelist rejects with `ORACLE_FIELD_IN_MODEL_INPUT`. It now mirrors
  what `tools/convert_release.py` emits — exactly `{id, images, messages}`,
  plain-string contents — at 554/65/75, images up to the 8 cap, 88% Korean.
  All checks pass.
- **5 records have `claim_boundary_status: UNRESOLVED`** (4 train, 1 validation).
  They are ACCEPTED in the release and the release is frozen, so they stay. Worth
  a look if the metrics come out strange.
- **The non-Korean split is too small to measure.** 82 records total, only 4 in
  test. Any non-Korean number will be noise; say so rather than reporting it as
  a result.

## The instruction the dataset agent asked to be passed on

> Read `FINETUNING_HANDOFF.md` first and follow it exactly. Do not search, merge,
> relabel, deduplicate or resplit the data; fine-tune and evaluate on the
> provided train/validation/test split only. Preserve the Korean
> `target_claim_clean` byte-for-byte — do not translate or normalise it. Write
> every training artefact outside the dataset directory.

`tools/convert_release.py` already satisfies all of it: it reads the release
read-only, refuses to write inside it, preserves image order, and copies the
Korean targets verbatim.
