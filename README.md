# Gemma 31B — independent patent claim generator

Vision-language fine-tuning of **`google/gemma-4-31B-it`** on a frozen, 114-record
multi-image dataset of patent claim pages. Given one or more claim-page images in
order, the model drafts a single independent physical apparatus / device / system /
cartridge / assembly claim.

The dataset (`v1.1.2-independent-oracle-clean`) is private and is not in this
repository. It is treated as an immutable reference release: nothing here rewrites,
re-splits, or filters it.

## Layout

```
pipeline/
  common.py         paths, seeding, schema normalisation, model loading
  fetch_dataset.py  download + SHA-256 gate + extract + integrity manifest
  preflight.py      data-contract gate (counts, images, targets, leakage)
  token_audit.py    per-record token lengths; proves zero target truncation
  dataset.py        encoding + assistant-only loss masking + collator
  train.py          bf16 LoRA SFT with a pre-training base baseline
  evaluate.py       base vs tuned under identical conditions
  push_hub.py       publish adapter and merged model
  inference.py      standalone image(s) -> claim
  report.py         FINAL_TRAINING_REPORT.md + status flags
run_all.sh          pod-side orchestrator (resumable stages)
reproduce.sh        end-to-end reproduction from a dataset ZIP
tests/              offline contract + masking tests (no GPU, no network)
```

## Method

**bf16 LoRA on the language tower, vision tower frozen.**

91 training records against ~31B parameters is well inside the regime where
full-weight updates memorise instead of generalise, and 62.5 GB of bf16 weights
plus Adam moments does not fit any single available GPU. LoRA keeps the trainable
count ~4 orders of magnitude smaller, keeps the run resumable, and leaves the
pretrained visual features untouched.

No quantisation: at 141 GB of VRAM the weights fit in bf16, so QLoRA would add
quantisation error and a dependency on 4-bit kernels being correct for a
recently-released architecture, for no capacity benefit.

### Loss masking

The mask is derived structurally, not by pattern-matching the chat template. For
each record the generation prompt is rendered and asserted to be a strict prefix of
the full rendered conversation, and to tokenise to a strict prefix of the full token
ids. Everything up to that boundary — system text, user text, image placeholders —
is set to `-100`, as is padding. Only assistant tokens carry loss. If the prefix
property ever fails, the run stops rather than silently masking the wrong span.

### Hard failures

The pipeline stops rather than degrading on: NaN/Inf loss, target truncation,
missing images, image-order contract violations, empty assistant targets, records
appearing in more than one split, and any reference to oracle / canonical /
excluded / evaluation material from a training input.

## Data contract

- Model input is limited to `hf_multimodal/{train,validation,test}.jsonl` and the
  `images/` files those records reference.
- `metadata` is never fed to the model.
- `images[]` order is preserved; no image is dropped.
- Splits are used as shipped; validation and test are never folded into train.
- Test is read once, after the configuration and best checkpoint are frozen.

## Run

On a GPU host with the dataset ZIP in hand:

```bash
./reproduce.sh /path/to/final_dataset_v112.zip [output_dir]
```

Offline tests (no GPU, no network, no base model):

```bash
python tests/test_pipeline.py
```

## Environment

| variable | purpose |
|---|---|
| `HF_TOKEN` | Hub token with write access; required only for `push_hub.py` |
| `DATASET_URL` | direct dataset URL; otherwise `gdown` uses `DRIVE_FILE_ID` |
| `BASE_MODEL`, `BASE_MODEL_REVISION` | pin the base model |
| `EPOCHS`, `LR`, `MICRO_BS`, `GRAD_ACCUM`, `LORA_R`, `SEED` | training knobs |
| `HF_ADAPTER_REPO`, `HF_MERGED_REPO`, `HF_PRIVATE` | publication targets |

Tokens are read from the environment only. They are never written to artifacts,
logs, model cards, or this repository.

## Licence

Derivatives of `google/gemma-4-31B-it` remain subject to the Gemma Terms of Use,
including its use restrictions, and must carry those terms downstream.
