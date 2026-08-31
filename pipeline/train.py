"""bf16 LoRA supervised fine-tuning of Gemma 31B on the frozen claim dataset.

Why LoRA and not full fine-tuning: 91 training records against ~31B parameters
is far into the regime where full-weight updates memorise rather than
generalise, and 62.5 GB of bf16 weights plus Adam moments does not fit any
single available GPU. LoRA on the language tower keeps the trainable parameter
count ~4 orders of magnitude smaller, makes the run reproducible and
resumable, and leaves the vision tower frozen so the visual features the model
already has are not disturbed by 91 examples.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from common import (OUT, SEED, die, find_package_root, load_all, load_vlm, log,
                    set_all_seeds, write_json)
from dataset import Collator, RecordDataset

MODEL_ID = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
REVISION = os.environ.get("BASE_MODEL_REVISION", "main")
CKPT_DIR = OUT / "checkpoints"
FINAL_DIR = OUT / "final_model_or_adapter"

CFG = {
    "base_model": MODEL_ID,
    "revision": REVISION,
    "method": "LoRA (bf16, no quantisation)",
    "seed": SEED,
    "num_train_epochs": float(os.environ.get("EPOCHS", "6")),
    "micro_batch_size": int(os.environ.get("MICRO_BS", "1")),
    "gradient_accumulation_steps": int(os.environ.get("GRAD_ACCUM", "4")),
    "learning_rate": float(os.environ.get("LR", "1e-4")),
    "lr_scheduler_type": os.environ.get("SCHED", "cosine"),
    "warmup_ratio": float(os.environ.get("WARMUP", "0.1")),
    "weight_decay": float(os.environ.get("WD", "0.0")),
    "max_grad_norm": 1.0,
    "lora_r": int(os.environ.get("LORA_R", "16")),
    "lora_alpha": int(os.environ.get("LORA_ALPHA", "32")),
    "lora_dropout": float(os.environ.get("LORA_DROPOUT", "0.05")),
    "gradient_checkpointing": True,
    "bf16": True,
    "train_vision_tower": False,
    "best_checkpoint_metric": "eval_loss",
    "best_checkpoint_mode": "min",
    "early_stopping_patience": int(os.environ.get("PATIENCE", "3")),
}


def environment_txt() -> None:
    lines = [
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            lines.append(f"gpu{i}={p.name} vram_gb={round(p.total_memory/2**30, 1)}")
        lines.append(f"torch_cuda={torch.version.cuda}")
    for mod in ("transformers", "peft", "accelerate", "datasets", "safetensors"):
        try:
            lines.append(f"{mod}={__import__(mod).__version__}")
        except Exception as exc:
            lines.append(f"{mod}=<unavailable: {exc}>")
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
        ).stdout
    except Exception:
        freeze = ""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "environment.txt").write_text(
        "\n".join(lines) + "\n\n# pip freeze\n" + freeze, encoding="utf-8"
    )
    log("\n".join(lines))


def discover_lora_targets(model) -> list[str]:
    """LoRA goes on the language tower only; vision/audio stay frozen.

    Fully-qualified module names are returned rather than bare suffixes. PEFT
    matches a bare suffix everywhere it occurs, which would pull in the vision
    tower's identically named projections -- and Gemma 4 wraps those in
    Gemma4ClippableLinear, which PEFT cannot adapt. Exact names keep the
    selection to the modules that were actually inspected here.
    """
    skip = ("vision_tower", "vision_model", "vision", "audio_tower", "audio",
            "multi_modal_projector", "embed_tokens", "lm_head")
    preferred = ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj")
    chosen: list[str] = []
    suffixes: set[str] = set()
    widths: set[int] = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if any(s in name for s in skip):
            continue
        leaf = name.split(".")[-1]
        if leaf not in preferred:
            continue
        chosen.append(name)
        suffixes.add(leaf)
        widths.add(int(mod.in_features))
    if not chosen:
        die("LORA_TARGET_DISCOVERY_FAILED",
            "No adaptable Linear projection found in the language tower.")
    log(f"LoRA targets: {len(chosen)} modules, suffixes {sorted(suffixes)}, "
        f"in_features {sorted(widths)}")
    log(f"LoRA target sample: {chosen[:3]} ... {chosen[-1]}")
    return sorted(chosen)


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a", encoding="utf-8")

    def write(self, obj: dict) -> None:
        self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.fh.flush()


def main() -> None:
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    set_all_seeds(SEED)
    environment_txt()
    OUT.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    pkg = find_package_root()
    data = load_all(pkg)

    log("Encoding splits (no truncation is applied at any point).")
    train_ds = RecordDataset(processor, data["train"])
    val_ds = RecordDataset(processor, data["validation"])
    log(f"train={len(train_ds)} validation={len(val_ds)}")

    log(f"Loading {MODEL_ID}@{REVISION} in bfloat16.")
    model = load_vlm(
        MODEL_ID, REVISION, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager",
    )
    model.config.use_cache = False

    targets = discover_lora_targets(model)
    # The exact module list is written to the adapter's adapter_config.json; keep
    # the run config readable with a summary rather than several hundred names.
    CFG["lora_target_module_count"] = len(targets)
    CFG["lora_target_suffixes"] = sorted({t.split(".")[-1] for t in targets})

    peft_cfg = LoraConfig(
        r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=targets,
    )
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    CFG["trainable_params"] = trainable
    CFG["total_params"] = total
    log(f"trainable={trainable:,} / total={total:,} ({100*trainable/total:.4f}%)")

    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    eff_bs = CFG["micro_batch_size"] * CFG["gradient_accumulation_steps"] * world
    CFG["effective_batch_size"] = eff_bs
    CFG["steps_per_epoch"] = math.ceil(len(train_ds) / eff_bs)
    CFG["total_optimizer_steps"] = int(CFG["steps_per_epoch"] * CFG["num_train_epochs"])
    log(f"effective_batch_size={eff_bs} steps/epoch={CFG['steps_per_epoch']} "
        f"total_steps={CFG['total_optimizer_steps']}")

    jl = JsonlLogger(OUT / "train_log.jsonl")

    class LogAndGuard(TrainerCallback):
        """Streams metrics to train_log.jsonl and turns NaN/Inf into a hard stop."""

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            rec = {"time": time.time(), "step": state.global_step,
                   "epoch": state.epoch, **logs}
            jl.write(rec)
            log(json.dumps(rec, ensure_ascii=False))
            for k, v in logs.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    die("NAN_OR_INF_LOSS", f"{k}={v} at step {state.global_step}")

    wanted = dict(
        output_dir=str(CKPT_DIR),
        seed=SEED, data_seed=SEED,
        num_train_epochs=CFG["num_train_epochs"],
        per_device_train_batch_size=CFG["micro_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
        learning_rate=CFG["learning_rate"],
        lr_scheduler_type=CFG["lr_scheduler_type"],
        warmup_ratio=CFG["warmup_ratio"],
        weight_decay=CFG["weight_decay"],
        max_grad_norm=CFG["max_grad_norm"],
        bf16=True, tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(os.environ.get("SAVE_LIMIT", "8")),
        load_best_model_at_end=True,
        metric_for_best_model=CFG["best_checkpoint_metric"],
        greater_is_better=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        label_names=["labels"],
    )

    # TrainingArguments has been reshaped across transformers releases: v5 drops
    # or renames several long-standing arguments. Adapt to the signature that is
    # actually installed rather than pinning one, and be explicit in the log and
    # the run config about anything that could not be honoured -- a silently
    # dropped argument would change training without showing up in the report.
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    aliases = {
        "eval_strategy": ("evaluation_strategy",),
        "evaluation_strategy": ("eval_strategy",),
    }
    kwargs: dict = {}
    dropped: list[str] = []
    for key, value in wanted.items():
        if key in accepted:
            kwargs[key] = value
            continue
        alt = next((a for a in aliases.get(key, ()) if a in accepted), None)
        if alt:
            log(f"[train] TrainingArguments: '{key}' -> '{alt}' on this release")
            kwargs[alt] = value
        else:
            dropped.append(key)

    # Warmup is a training decision, not a formatting detail: if the ratio form
    # is gone, express the same intent in steps rather than losing the warmup.
    if "warmup_ratio" in dropped and "warmup_steps" in accepted:
        steps = max(1, round(CFG["warmup_ratio"] * CFG["total_optimizer_steps"]))
        kwargs["warmup_steps"] = steps
        dropped.remove("warmup_ratio")
        log(f"[train] TrainingArguments: warmup_ratio={CFG['warmup_ratio']} expressed "
            f"as warmup_steps={steps} of {CFG['total_optimizer_steps']}")
        CFG["warmup_steps"] = steps

    # Losing any of these would change what is trained or how the checkpoint is
    # chosen, so an incompatible release has to stop the run rather than quietly
    # train something else.
    essential = {
        "bf16", "gradient_checkpointing", "remove_unused_columns", "label_names",
        "eval_strategy", "save_strategy", "load_best_model_at_end",
        "metric_for_best_model", "learning_rate", "num_train_epochs",
        "per_device_train_batch_size", "gradient_accumulation_steps", "seed",
        "output_dir", "max_grad_norm", "weight_decay", "lr_scheduler_type",
    }
    blocking = sorted(set(dropped) & essential)
    if blocking:
        die("TRAINING_ARGUMENTS_INCOMPATIBLE",
            f"transformers {transformers.__version__} does not accept {blocking}, "
            "and proceeding without them would change training. Accepted "
            f"parameters: {sorted(accepted)}")
    if dropped:
        log(f"[train] TrainingArguments does not accept {dropped} on transformers "
            f"{transformers.__version__}; proceeding without them.")
    CFG["unsupported_training_arguments"] = dropped

    args = TrainingArguments(**kwargs)

    pad_id = processor.tokenizer.pad_token_id or 0
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(pad_token_id=pad_id),
        callbacks=[LogAndGuard(),
                   EarlyStoppingCallback(early_stopping_patience=CFG["early_stopping_patience"])],
    )

    # Baseline: LoRA B matrices are zero-initialised, so the model here is
    # numerically the untouched base model. Same code path, same data, same
    # collator as the post-training evaluation.
    log("Measuring base-model validation baseline before any optimiser step.")
    base_metrics = trainer.evaluate(metric_key_prefix="baseline")
    base_loss = float(base_metrics.get("baseline_loss", float("nan")))
    log(f"baseline_validation_loss={base_loss}")
    jl.write({"phase": "baseline", **base_metrics})

    resume = None
    if CKPT_DIR.is_dir() and any(CKPT_DIR.glob("checkpoint-*")):
        resume = True
        log("Existing checkpoints found; resuming.")

    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    train_seconds = time.time() - t0

    peak_vram = (
        round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available() else None
    )
    final_metrics = trainer.evaluate(metric_key_prefix="eval")
    best_ckpt = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(FINAL_DIR))
    processor.save_pretrained(str(FINAL_DIR))

    CFG.update({
        "peak_vram_gib": peak_vram,
        "train_seconds": round(train_seconds, 1),
        "best_checkpoint": best_ckpt,
        "best_eval_loss": best_metric,
        "baseline_validation_loss": base_loss,
        "baseline_validation_perplexity": (
            math.exp(base_loss) if base_loss == base_loss and base_loss < 20 else None
        ),
        "final_validation_loss": float(final_metrics.get("eval_loss", float("nan"))),
        "global_steps_run": int(trainer.state.global_step),
        "epochs_run": float(trainer.state.epoch or 0),
    })
    (OUT / "training_config.yaml").write_text(
        yaml.safe_dump(CFG, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(OUT / "train_result.json", {
        "train_runtime_metrics": result.metrics,
        "final_validation_metrics": final_metrics,
        "baseline_validation_metrics": base_metrics,
        "TRAINING_COMPLETE": True,
    })
    log(f"TRAINING_COMPLETE=true best_checkpoint={best_ckpt} "
        f"best_eval_loss={best_metric} peak_vram_gib={peak_vram}")


if __name__ == "__main__":
    main()
