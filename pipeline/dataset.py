"""Record -> model tensors, with assistant-only loss masking.

The mask is derived structurally rather than by pattern-matching the chat
template: the generation prompt for a record is rendered, checked to be a
strict prefix of the full conversation, and every token in that prefix is set
to -100. Image placeholders, system/user text and padding therefore carry no
loss; only the assistant response does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from common import Record, build_chat_messages, die, images_of

IGNORE_INDEX = -100


def render_texts(processor, rec: Record) -> tuple[str, str]:
    full = processor.apply_chat_template(
        build_chat_messages(rec, for_prompt=False),
        tokenize=False, add_generation_prompt=False,
    )
    prompt = processor.apply_chat_template(
        build_chat_messages(rec, for_prompt=True),
        tokenize=False, add_generation_prompt=True,
    )
    if not full.startswith(prompt):
        die(
            "CHAT_TEMPLATE_PREFIX_VIOLATION",
            "The rendered generation prompt is not a prefix of the rendered full "
            "conversation, so assistant-only loss masking cannot be derived "
            f"safely.\n--- prompt tail ---\n{prompt[-400:]}\n"
            f"--- full at same offset ---\n{full[:len(prompt)][-400:]}",
        )
    return full, prompt


def _call_processor(processor, text: str, imgs: list) -> dict:
    kwargs = dict(text=text, return_tensors="pt", padding=False)
    if imgs:
        try:
            return processor(images=imgs, **kwargs)
        except Exception:
            return processor(images=[imgs], **kwargs)
    return processor(**kwargs)


def encode_record(processor, rec: Record, with_labels: bool = True) -> dict[str, Any]:
    """Encode one record. Never truncates: truncation of a target is a hard failure."""
    imgs = images_of(rec)
    full_text, prompt_text = render_texts(processor, rec)

    full = _call_processor(processor, full_text, imgs)
    prompt = _call_processor(processor, prompt_text, imgs)

    full_ids = full["input_ids"][0]
    prompt_ids = prompt["input_ids"][0]
    plen = int(prompt_ids.shape[0])

    if plen >= int(full_ids.shape[0]):
        die(
            "EMPTY_ASSISTANT_TARGET",
            f"{rec.record_id}: prompt encoding is not shorter than the full "
            "encoding, so the assistant target contributes no tokens.",
        )
    if not torch.equal(full_ids[:plen], prompt_ids):
        die(
            "TOKENIZER_PREFIX_VIOLATION",
            f"{rec.record_id}: prompt token ids are not a prefix of the full "
            "token ids; assistant-only masking would mask the wrong span.",
        )

    out: dict[str, Any] = {k: v for k, v in full.items()}
    out["_prompt_len"] = plen
    out["_target_len"] = int(full_ids.shape[0]) - plen
    out["_total_len"] = int(full_ids.shape[0])
    out["_record_id"] = rec.record_id
    out["_n_images"] = len(imgs)

    if with_labels:
        labels = full_ids.clone()
        labels[:plen] = IGNORE_INDEX
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[full_ids == pad_id] = IGNORE_INDEX
        if int((labels != IGNORE_INDEX).sum()) == 0:
            die("EMPTY_ASSISTANT_TARGET", f"{rec.record_id}: all label tokens masked.")
        out["labels"] = labels.unsqueeze(0)
    return out


def encode_prompt_only(processor, rec: Record) -> dict[str, Any]:
    """Encode just the generation prompt, for evaluation-time generation."""
    imgs = images_of(rec)
    _, prompt_text = render_texts(processor, rec)
    enc = _call_processor(processor, prompt_text, imgs)
    return enc


class RecordDataset(torch.utils.data.Dataset):
    """Pre-encodes every record once; the corpus is 114 records, so this is cheap
    and it guarantees the token audit and the training run see identical tensors."""

    def __init__(self, processor, records: list[Record]):
        self.encoded = [encode_record(processor, r) for r in records]
        self.records = records

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.encoded[i]


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, feats: list[dict[str, Any]]) -> dict[str, Any]:
        keys = [k for k in feats[0] if not k.startswith("_")]
        maxlen = max(int(f["input_ids"].shape[-1]) for f in feats)
        batch: dict[str, Any] = {}

        for key in keys:
            vals = [f[key] for f in feats]
            if key in ("input_ids", "attention_mask", "labels", "token_type_ids"):
                pad = {
                    "input_ids": self.pad_token_id,
                    "attention_mask": 0,
                    "labels": IGNORE_INDEX,
                    "token_type_ids": 0,
                }[key]
                rows = []
                for v in vals:
                    v = v[0] if v.dim() == 2 else v
                    if v.shape[0] < maxlen:
                        v = torch.cat(
                            [v, torch.full((maxlen - v.shape[0],), pad, dtype=v.dtype)]
                        )
                    rows.append(v)
                batch[key] = torch.stack(rows)
            else:
                # Vision tensors: concatenate along the image axis so the order of
                # images across the batch matches the order of their placeholders.
                try:
                    batch[key] = torch.cat([v for v in vals], dim=0)
                except Exception:
                    batch[key] = vals[0] if len(vals) == 1 else torch.stack(vals)
        return batch
