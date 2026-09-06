# Final training report - Gemma 31B claim generator

## Status

```
DATA_DOWNLOAD_VERIFIED: true
DATASET_VALIDATOR_PASS: true
MULTI_IMAGE_CONTRACT_PASS: true
ZERO_TARGET_TRUNCATION_PASS: true
TRAINING_COMPLETE: true
VALIDATION_COMPLETE: true
TEST_EVALUATION_COMPLETE: false
ARTIFACTS_READY: false
```

**Not a full success - at least one flag is false.**

## Base model

- Model: `google/gemma-4-31B-it`
- Revision: `main`
- Architecture: `Gemma4ForConditionalGeneration` (vision-language, image-text-to-text)
- License: Gemma Terms of Use. Redistribution of derivatives must carry the same
  terms and use restrictions; the fine-tuned repositories are published private.

## Method and why

LoRA (bf16, no quantisation). Chosen over full fine-tuning because 91 training records
cannot support ~31B free parameters without memorisation, and because 62.5 GB of
bf16 weights plus optimiser state exceeds every single GPU available. The vision
tower is frozen; LoRA is applied to the language tower only, so visual features
are not perturbed by 91 examples.

- LoRA rank 16, alpha 32, dropout 0.05
- Target modules: 410 language-tower projections, suffixes ['down_proj', 'gate_proj', 'k_proj', 'o_proj', 'q_proj', 'up_proj', 'v_proj']
  (exact module list in the adapter's `adapter_config.json`)
- Trainable parameters: 122,429,440 of 31,395,515,952 (0.3900%)

## Data

- Version `finetune_multilingual_approved_20260902`, SHA-256 verified against the frozen release
- Counts: {'train': 554, 'validation': 65, 'test': 75}
- Splits untouched; no re-partitioning, no merging of validation/test into train
- Oracle, canonical, excluded and evaluation material never enters model input
- Max total sequence: 4075 tokens against a
  262144-token context, so no target is ever truncated

## Training

- GPU: gpu0=NVIDIA H200 vram_gb=139.8
- Peak VRAM: 77.04 GiB
- Seed: 42
- Micro batch 1 x grad-accum 4
  = effective batch 4
- Epochs requested 3.0, ran 3.0;
  417 optimiser steps
- LR 5e-05 with cosine schedule,
  warmup ratio 0.1
- Best-checkpoint rule fixed before training: minimise `eval_loss`
- Best checkpoint: `/workspace/outputs/checkpoints/checkpoint-278` (eval_loss 1.366241216659546)
- Base-model validation baseline: loss 1.549530267715454,
  perplexity 4.709257572646149
- Wall clock: 3689.1 s

## Results

**validation** (n=65)

| metric | base | fine-tuned |
|---|---|---|
| loss | 2.542392 | 1.28607 |
| perplexity | 12.71 | 3.6185 |
| ROUGE-L (F) | 0.0893 | 0.0745 |
| chrF | 19.2581 | 13.1206 |
| empty responses | 0 | 0 |
| excessive repetition | 0 | 9 |
| reads as dependent claim | 21 | 0 |
| no closing period | 33 | 9 |
| well-formed independent claims | 22/65 | 40/65 |
| mean length (words) | 349.6 | 104.1 |


_test: not evaluated_


Base and fine-tuned rows come from one model object evaluated twice, with the
adapter enabled and disabled, under identical decoding settings
({'do_sample': False, 'num_beams': 1, 'max_new_tokens': 1024}).

## Published models

{
  "adapter_repo": "https://huggingface.co/Mepeng22/gemma-4-31b-claim-lora-v2",
  "private": true
}

## Artifacts

All under `/workspace/outputs`:

- `final_model_or_adapter`
- `checkpoints`
- `training_config.yaml`
- `environment.txt`
- `dataset_preflight.json`
- `token_length_audit.json`
- `train_log.jsonl`
- `metrics.json`
- `validation_predictions.jsonl`
- `inference.py`
- `reproduce.sh`
- MISSING: `test_predictions.jsonl`

## Reproduce

```bash
bash /workspace/outputs/reproduce.sh /path/to/final_dataset_v112.zip
```
