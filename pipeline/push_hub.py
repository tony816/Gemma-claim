"""Publish the LoRA adapter and the merged model to the Hugging Face Hub.

The token is read from HF_TOKEN and is never written to any artifact, log line
or model card.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from common import OUT, die, load_vlm, log

MODEL_ID = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
REVISION = os.environ.get("BASE_MODEL_REVISION", "main")
ADAPTER_DIR = OUT / "final_model_or_adapter"
MERGED_DIR = OUT / "merged_model"
PRIVATE = os.environ.get("HF_PRIVATE", "1") not in ("0", "false", "False")
PUSH_MERGED = os.environ.get("PUSH_MERGED", "1") not in ("0", "false", "False")


def card(repo_kind: str, metrics: dict, cfg_text: str) -> str:
    tuned = metrics.get("tuned", {})
    base = metrics.get("base", {})

    def row(split: str) -> str:
        b, t = base.get(split, {}), tuned.get(split, {})
        if not t:
            return ""
        return (
            f"| {split} | {b.get('loss')} | {t.get('loss')} | "
            f"{b.get('rougeL_f')} | {t.get('rougeL_f')} | "
            f"{b.get('chrf')} | {t.get('chrf')} |\n"
        )

    table = (
        "| split | base loss | tuned loss | base ROUGE-L | tuned ROUGE-L | "
        "base chrF | tuned chrF |\n|---|---|---|---|---|---|---|\n"
        + row("validation") + row("test")
    )
    what = (
        "LoRA adapter (load on top of the base model)"
        if repo_kind == "adapter"
        else "Base model with the LoRA adapter merged into the weights"
    )
    return f"""---
base_model: {MODEL_ID}
library_name: {'peft' if repo_kind == 'adapter' else 'transformers'}
pipeline_tag: image-text-to-text
tags:
- gemma
- vision-language
- patent
- lora
---

# Gemma 31B - independent patent claim generation

{what}.

- **Base model**: `{MODEL_ID}` (revision `{REVISION}`)
- **Method**: bf16 LoRA supervised fine-tuning, assistant-only loss masking
- **Dataset**: private, frozen release `v1.1.2-independent-oracle-clean`
  (91 train / 11 validation / 12 test multi-image records). Not redistributed here.
- **Task**: read one or more claim-page images and emit an independent
  physical apparatus / device / system / cartridge / assembly claim.

## Results

{table}

Base and tuned numbers were produced by the same code path, the same decoding
settings and the same collator; the base row is the identical model object with
the adapter disabled.

## Intended use and limits

Trained on 91 examples. It is a domain-shaped generator, not a drafting
authority: outputs need attorney review before any filing use. The base model's
license and distribution terms apply unchanged.

## Training configuration

```yaml
{cfg_text}
```
"""


def main() -> None:
    from huggingface_hub import HfApi, whoami

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        die("HF_TOKEN_MISSING",
            "HF_TOKEN is not set; cannot publish to the Hub. Provide a token with "
            "write permission on the target account.")
    if not ADAPTER_DIR.is_dir():
        die("ADAPTER_MISSING", f"No adapter at {ADAPTER_DIR}.")

    api = HfApi(token=token)
    try:
        me = whoami(token=token)
    except Exception as exc:
        die("HF_TOKEN_INVALID", f"Hugging Face rejected the token: {exc}")
    user = os.environ.get("HF_USER") or me["name"]
    log(f"Authenticated to the Hub as {user}.")

    metrics = {}
    if (OUT / "metrics.json").is_file():
        metrics = json.loads((OUT / "metrics.json").read_text())
    cfg_text = ""
    if (OUT / "training_config.yaml").is_file():
        cfg_text = (OUT / "training_config.yaml").read_text()

    adapter_repo = os.environ.get("HF_ADAPTER_REPO", f"{user}/gemma-4-31b-claim-lora")
    merged_repo = os.environ.get("HF_MERGED_REPO", f"{user}/gemma-4-31b-claim-merged")
    pushed = {}

    # ---- adapter -------------------------------------------------------
    (ADAPTER_DIR / "README.md").write_text(card("adapter", metrics, cfg_text), encoding="utf-8")
    for extra in ("metrics.json", "training_config.yaml", "token_length_audit.json",
                  "dataset_preflight.json", "train_log.jsonl",
                  "validation_predictions.jsonl", "test_predictions.jsonl"):
        src = OUT / extra
        if src.is_file():
            shutil.copy2(src, ADAPTER_DIR / extra)

    log(f"Creating and pushing adapter repo {adapter_repo} (private={PRIVATE}).")
    api.create_repo(adapter_repo, private=PRIVATE, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(ADAPTER_DIR), repo_id=adapter_repo,
                      repo_type="model", commit_message="LoRA adapter + run artifacts")
    pushed["adapter_repo"] = f"https://huggingface.co/{adapter_repo}"
    log(f"Adapter pushed: {pushed['adapter_repo']}")

    # ---- merged --------------------------------------------------------
    if PUSH_MERGED:
        from peft import PeftModel
        from transformers import AutoProcessor

        log("Merging the adapter into the base weights (bf16).")
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        base = load_vlm(
            MODEL_ID, REVISION, dtype=torch.bfloat16, device_map={"": 0},
        )
        merged = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).merge_and_unload()
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(MERGED_DIR), safe_serialization=True, max_shard_size="4GB")
        processor.save_pretrained(str(MERGED_DIR))
        (MERGED_DIR / "README.md").write_text(card("merged", metrics, cfg_text), encoding="utf-8")
        del merged, base
        torch.cuda.empty_cache()

        log(f"Creating and pushing merged repo {merged_repo} (private={PRIVATE}).")
        api.create_repo(merged_repo, private=PRIVATE, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(MERGED_DIR), repo_id=merged_repo,
                          repo_type="model", commit_message="Merged Gemma 31B + claim LoRA")
        pushed["merged_repo"] = f"https://huggingface.co/{merged_repo}"
        log(f"Merged model pushed: {pushed['merged_repo']}")

    pushed["private"] = PRIVATE
    (OUT / "hub_push.json").write_text(json.dumps(pushed, indent=2), encoding="utf-8")
    log(f"HUB_PUSH_COMPLETE {json.dumps(pushed)}")


if __name__ == "__main__":
    main()
