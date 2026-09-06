# v2 run — 694-record release, adapter continued (2026-09-06)

Continued `Mepeng22/gemma-4-31b-claim-lora` on the 554 training records of
`finetune_multilingual_approved_20260902`. Pushed as
`Mepeng22/gemma-4-31b-claim-lora-v2` (private). The v1 adapter is untouched.

## Configuration

| | |
|---|---|
| base | `google/gemma-4-31B-it`, bf16, LoRA r=16 on 410 language-tower modules |
| resumed from | `Mepeng22/gemma-4-31b-claim-lora`, `is_trainable=True` |
| epochs / steps | 3 / 417 (139 per epoch, effective batch 4) |
| learning rate | 5e-5, cosine, warmup 0.1 |
| best checkpoint | **checkpoint-278 (epoch 2)**, eval_loss 1.3662 |
| eval | validation only, `MAX_NEW_TOKENS=1024` |
| peak VRAM | 77.04 GiB on an H200 |
| GPU | 1× H200 SXM community, 3h24m wall clock |

`eval_loss` by epoch: 1.3833 → **1.3662** → 1.3842. Three epochs was one too many;
`load_best_model_at_end` took epoch 2.

## The headline numbers, and why they mislead

| validation (n=65) | base | tuned |
|---|---|---|
| loss (teacher-forced) | 2.5424 | **1.2861** |
| chrF (sacrebleu default, β=2) | **19.26** | 13.12 |
| well-formed claims | 22/65 | **40/65** |
| mean words | 349.6 | **104.1** |
| repetition > 0.3 | 0/65 | **9/65** |

Loss down, chrF down is the pattern CLAUDE.md warns about. Here it is an
artifact. `evaluate.py` gives the base model no system prompt, so it writes
**344.8 words against a 90.9-word reference**, and chrF's default β=2 weights
recall twice as heavily as precision — saying everything scores well.

Re-scored across β on the same 59 Korean records:

| system | β=0.5 | **β=1** | β=2 | mean words |
|---|---|---|---|---|
| base, no prompt | 7.14 | **9.56** | 14.82 | 344.8 |
| base, cut to reference length | 11.25 | **11.52** | 11.87 | 90.9 |
| **tuned, no prompt** | 14.16 | **11.85** | 10.81 | **105.3** |
| base + claim prompt (`baseline.py`) | 15.47 | **11.78** | 9.86 | 41.8 |

Truncating the base output to the reference length costs it 14.82 → 11.87
without changing a word of its content. At balanced β=1 the three
length-comparable systems are within noise of each other: 11.85, 11.78, 11.52.

## What the fine-tune actually bought and cost

| Korean, n=59 | tuned | base + prompt |
|---|---|---|
| chrF β=1 | 11.85 | 11.78 |
| mean words (ref 90.9) | **105.3** | 41.8 |
| reference numerals | 2/59 | **0/59** |
| repetition collapse | **9/59 (15%)** | 0/59 |
| well-formed | 34/59 | **58/59** |

**It bought length calibration and nothing measurable in content, and it paid
with a 15% degeneration rate.** The base model without a prompt writes reference
numerals in 58 of 59 records; with the prompt, 0 of 59. Fine-tuning cut them to
2 of 59 on its own.

Teacher-forced loss fell 2.54 → 1.29 while content overlap did not move. That is
the signature of learning the target's form and length distribution rather than
the mapping from drawings to claim — the same failure as the 91-record run, much
milder. 554 records against 410 LoRA modules is still a thin adaptation.

## Does the serving prompt work on the tuned model?

`tools/prompt_probe.py`, 12 validation records, same model, only the system turn
differs. Training carries no system turn, so this instruction is outside the
distribution the LoRA deltas were fit on.

| condition | mean chrF | numerals | mean words | looping |
|---|---|---|---|---|
| training prompt (no system turn) | 10.24 | 2/12 | 71.2 | 1/12 |
| **+ serving system prompt** | 10.40 | **0/12** | 98.9 | **3/12** |

The prompt is not ignored: numerals go to zero and length moves toward the
reference. It is not free either — repetition collapse went from 1 in 12 to 3 in
12. chrF is unchanged. So the numeral ban works, and the degeneration is the
thing to fix, not the numerals.

## Recommendation

Do not ship the v2 adapter over prompting on this evidence. On content it ties
the prompted base model, and it loses on well-formedness (34/59 vs 58/59) to a
failure the base model does not have. Keep the adapter — it is 498 MB, it is the
only artifact that calibrates length correctly, and it is the base for whatever
fixes the repetition.

The repetition collapse is the one defect worth attacking next, and it is not a
data-cleanliness problem: it appears in 15% of generations regardless of the
numerals. Candidates, cheapest first: a repetition penalty at decode time (free,
measurable on the saved predictions), fewer epochs or a lower rank, and more
records.

The test split has **not** been spent. It is still clean for a final comparison
once something is worth comparing.

## Cost

`get-billing`, hourly buckets, 2026-09-06:

| bucket (UTC) | pod GPU | pod disk | serverless |
|---|---|---|---|
| 03:00–04:00 | — | — | $0.130 (baseline.py) |
| 04:00–05:00 | $2.441 | $0.024 | $0.346 |
| 05:00–06:00 | $3.586 | $0.032 | — |
| 06:00–07:00 | $1.200 | $0.011 | — |
| 07:00–08:00 | not yet reported at the time of writing | | |

About **$7.77 measured through 07:00Z**, with the final ~40 minutes of pod time
still to appear. Evaluation, not training, was the expensive half: 417 optimiser
steps took ~50 minutes; 130 free-running generations took longer.
