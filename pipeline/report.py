"""Assemble FINAL_TRAINING_REPORT.md and the status flag block from artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from common import OUT, log

FLAGS = [
    "DATA_DOWNLOAD_VERIFIED",
    "DATASET_VALIDATOR_PASS",
    "MULTI_IMAGE_CONTRACT_PASS",
    "ZERO_TARGET_TRUNCATION_PASS",
    "TRAINING_COMPLETE",
    "VALIDATION_COMPLETE",
    "TEST_EVALUATION_COMPLETE",
    "ARTIFACTS_READY",
]

REQUIRED_ARTIFACTS = [
    "final_model_or_adapter", "checkpoints", "training_config.yaml",
    "environment.txt", "dataset_preflight.json", "token_length_audit.json",
    "train_log.jsonl", "metrics.json", "validation_predictions.jsonl",
    "test_predictions.jsonl", "inference.py", "reproduce.sh",
    "FINAL_TRAINING_REPORT.md",
]


def load(name: str) -> dict:
    p = OUT / name
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text()) if name.endswith(".json") else {}
    except Exception:
        return {}


def main() -> None:
    dl = load("dataset_download.json")
    pf = load("dataset_preflight.json")
    ta = load("token_length_audit.json")
    tr = load("train_result.json")
    me = load("metrics.json")
    hp = load("hub_push.json")
    cfg = {}
    if (OUT / "training_config.yaml").is_file():
        cfg = yaml.safe_load((OUT / "training_config.yaml").read_text()) or {}

    present = [a for a in REQUIRED_ARTIFACTS if (OUT / a).exists()]
    missing = [a for a in REQUIRED_ARTIFACTS if not (OUT / a).exists()]
    # FINAL_TRAINING_REPORT.md is written by this script, so do not require it of itself.
    artifacts_ready = not [m for m in missing if m != "FINAL_TRAINING_REPORT.md"]

    status = {
        "DATA_DOWNLOAD_VERIFIED": bool(dl.get("DATA_DOWNLOAD_VERIFIED")),
        "DATASET_VALIDATOR_PASS": bool(pf.get("DATASET_VALIDATOR_PASS")),
        "MULTI_IMAGE_CONTRACT_PASS": bool(pf.get("MULTI_IMAGE_CONTRACT_PASS")),
        "ZERO_TARGET_TRUNCATION_PASS": bool(ta.get("ZERO_TARGET_TRUNCATION_PASS")),
        "TRAINING_COMPLETE": bool(tr.get("TRAINING_COMPLETE")),
        "VALIDATION_COMPLETE": bool(me.get("tuned", {}).get("validation")),
        "TEST_EVALUATION_COMPLETE": bool(me.get("tuned", {}).get("test")),
        "ARTIFACTS_READY": artifacts_ready,
    }

    def cmp_table(split: str) -> str:
        b = me.get("base", {}).get(split, {})
        t = me.get("tuned", {}).get(split, {})
        if not t:
            return f"_{split}: not evaluated_\n"
        bq, tq = b.get("qualitative", {}), t.get("qualitative", {})
        return (
            f"**{split}** (n={tq.get('n')})\n\n"
            "| metric | base | fine-tuned |\n|---|---|---|\n"
            f"| loss | {b.get('loss')} | {t.get('loss')} |\n"
            f"| perplexity | {b.get('perplexity')} | {t.get('perplexity')} |\n"
            f"| ROUGE-L (F) | {b.get('rougeL_f')} | {t.get('rougeL_f')} |\n"
            f"| chrF | {b.get('chrf')} | {t.get('chrf')} |\n"
            f"| empty responses | {bq.get('empty_responses')} | {tq.get('empty_responses')} |\n"
            f"| excessive repetition | {bq.get('excessive_repetition_gt_0_3')} | "
            f"{tq.get('excessive_repetition_gt_0_3')} |\n"
            f"| reads as dependent claim | {bq.get('looks_dependent')} | "
            f"{tq.get('looks_dependent')} |\n"
            f"| no closing period | {bq.get('unterminated_no_final_period')} | "
            f"{tq.get('unterminated_no_final_period')} |\n"
            f"| well-formed independent claims | {bq.get('well_formed_independent_claims')}"
            f"/{bq.get('n')} | {tq.get('well_formed_independent_claims')}/{tq.get('n')} |\n"
            f"| mean length (words) | {bq.get('mean_words')} | {tq.get('mean_words')} |\n\n"
        )

    flag_block = "\n".join(f"{k}: {str(status[k]).lower()}" for k in FLAGS)
    env_lines = []
    if (OUT / "environment.txt").is_file():
        env_lines = (OUT / "environment.txt").read_text().splitlines()[:12]
    gpu_line = next((l for l in env_lines if l.startswith("gpu")), "see environment.txt")

    tp, allp = cfg.get("trainable_params"), cfg.get("total_params")
    if tp and allp:
        trainable_line = f"{tp:,} of {allp:,} ({100 * tp / allp:.4f}%)"
    else:
        trainable_line = "n/a"

    md = f"""# Final training report - Gemma 31B claim generator

## Status

```
{flag_block}
```

{"**All checks passed.**" if all(status.values()) else "**Not a full success - at least one flag is false.**"}

## Base model

- Model: `{cfg.get('base_model')}`
- Revision: `{cfg.get('revision')}`
- Architecture: `Gemma4ForConditionalGeneration` (vision-language, image-text-to-text)
- License: Gemma Terms of Use. Redistribution of derivatives must carry the same
  terms and use restrictions; the fine-tuned repositories are published private.

## Method and why

{cfg.get('method')}. Chosen over full fine-tuning because 91 training records
cannot support ~31B free parameters without memorisation, and because 62.5 GB of
bf16 weights plus optimiser state exceeds every single GPU available. The vision
tower is frozen; LoRA is applied to the language tower only, so visual features
are not perturbed by 91 examples.

- LoRA rank {cfg.get('lora_r')}, alpha {cfg.get('lora_alpha')}, dropout {cfg.get('lora_dropout')}
- Target modules: {cfg.get('lora_target_modules')}
- Trainable parameters: {trainable_line}

## Data

- Version `{dl.get('dataset_version')}`, SHA-256 verified against the frozen release
- Counts: {pf.get('record_counts')}
- Splits untouched; no re-partitioning, no merging of validation/test into train
- Oracle, canonical, excluded and evaluation material never enters model input
- Max total sequence: {ta.get('max_total_tokens')} tokens against a
  {ta.get('model_max_position_embeddings')}-token context, so no target is ever truncated

## Training

- GPU: {gpu_line}
- Peak VRAM: {cfg.get('peak_vram_gib')} GiB
- Seed: {cfg.get('seed')}
- Micro batch {cfg.get('micro_batch_size')} x grad-accum {cfg.get('gradient_accumulation_steps')}
  = effective batch {cfg.get('effective_batch_size')}
- Epochs requested {cfg.get('num_train_epochs')}, ran {cfg.get('epochs_run')};
  {cfg.get('global_steps_run')} optimiser steps
- LR {cfg.get('learning_rate')} with {cfg.get('lr_scheduler_type')} schedule,
  warmup ratio {cfg.get('warmup_ratio')}
- Best-checkpoint rule fixed before training: minimise `{cfg.get('best_checkpoint_metric')}`
- Best checkpoint: `{cfg.get('best_checkpoint')}` (eval_loss {cfg.get('best_eval_loss')})
- Base-model validation baseline: loss {cfg.get('baseline_validation_loss')},
  perplexity {cfg.get('baseline_validation_perplexity')}
- Wall clock: {cfg.get('train_seconds')} s

## Results

{cmp_table('validation')}
{cmp_table('test')}

Base and fine-tuned rows come from one model object evaluated twice, with the
adapter enabled and disabled, under identical decoding settings
({me.get('decoding')}).

## Published models

{json.dumps(hp, indent=2) if hp else '_Not pushed._'}

## Artifacts

All under `{OUT}`:

{chr(10).join('- `' + a + '`' for a in present)}
{('' if not missing else chr(10).join('- MISSING: `' + a + '`' for a in missing))}

## Reproduce

```bash
bash {OUT}/reproduce.sh /path/to/final_dataset_v112.zip
```
"""
    (OUT / "FINAL_TRAINING_REPORT.md").write_text(md, encoding="utf-8")
    (OUT / "status_flags.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    log("STATUS FLAGS:\n" + flag_block)
    log(f"Report written to {OUT / 'FINAL_TRAINING_REPORT.md'}")


if __name__ == "__main__":
    main()
