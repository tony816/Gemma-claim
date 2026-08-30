"""Standalone inference: one or more claim-page images -> independent claim.

    python inference.py --images page1.png page2.png [--adapter DIR] [--merged REPO]
"""
from __future__ import annotations

import argparse
import os

import torch
from PIL import Image

DEFAULT_PROMPT = (
    "Read the attached patent claim page image(s) in order and draft a single "
    "independent physical apparatus claim."
)


def build(base: str, revision: str, adapter: str | None, merged: str | None):
    from transformers import AutoProcessor

    from common import load_vlm

    src = merged or base
    rev = "main" if merged else revision
    processor = AutoProcessor.from_pretrained(src, revision=rev)
    model = load_vlm(src, rev, dtype=torch.bfloat16, device_map="auto")
    if adapter and not merged:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return processor, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True,
                    help="Image paths, in the order the model should read them.")
    ap.add_argument("--prompt", default=os.environ.get("PROMPT", DEFAULT_PROMPT))
    ap.add_argument("--base", default=os.environ.get("BASE_MODEL", "google/gemma-4-31B-it"))
    ap.add_argument("--revision", default=os.environ.get("BASE_MODEL_REVISION", "main"))
    ap.add_argument("--adapter", default=os.environ.get("ADAPTER_DIR"))
    ap.add_argument("--merged", default=os.environ.get("MERGED_REPO"))
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    processor, model = build(args.base, args.revision, args.adapter, args.merged)
    imgs = [Image.open(p).convert("RGB") for p in args.images]

    content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": args.prompt}]
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    enc = processor(text=text, images=imgs, return_tensors="pt")
    enc = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    plen = int(enc["input_ids"].shape[-1])

    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=processor.tokenizer.pad_token_id or 0)
    print(processor.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
