"""Record -> model tensors, with assistant-only loss masking.

The mask is derived structurally rather than by pattern-matching the chat
template. The real generation prompt is rendered -- the exact text the model
is given at inference -- and the assistant body is located with a sentinel
render, so the training sequence is that prompt followed by the template's own
rendering of the response. The prompt is a strict prefix of that sequence by
construction, and every token in it is set to -100. Image placeholders,
system/user text and padding therefore carry no loss; only the assistant
response does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from common import Record, build_chat_messages, die, images_of

IGNORE_INDEX = -100


SENTINEL = "ZQXASSISTANTBODYQZX"

# Gemma 4 renders a generation prompt in non-thinking mode as the conversation
# prefix plus an empty, immediately closed thought channel. That is the prompt
# the model is actually given at inference, so it is also what the training
# target has to follow -- the plain full-conversation rendering omits it.
THOUGHT_OPENER = "<|channel>thought\n<channel|>"


def render_texts(processor, rec: Record) -> tuple[str, str]:
    """Render (full, prompt) such that prompt is a strict prefix of full.

    The prompt is the real generation prompt, so training matches inference
    exactly. The full sequence is that prompt followed by the assistant body
    and the turn terminator, both taken verbatim from the template's own
    rendering rather than assembled by hand.

    The assistant body is located with a sentinel render instead of by
    pattern-matching the template, so nothing here depends on Gemma's markup.
    """
    msgs_full = build_chat_messages(rec, for_prompt=False)
    prompt = processor.apply_chat_template(
        build_chat_messages(rec, for_prompt=True),
        tokenize=False, add_generation_prompt=True,
    )
    full_render = processor.apply_chat_template(
        msgs_full, tokenize=False, add_generation_prompt=False,
    )

    probe_msgs = [dict(m) for m in msgs_full]
    probe_msgs[-1] = dict(probe_msgs[-1])
    probe_msgs[-1]["content"] = [{"type": "text", "text": SENTINEL}]
    probe = processor.apply_chat_template(
        probe_msgs, tokenize=False, add_generation_prompt=False,
    )
    if probe.count(SENTINEL) != 1:
        die("CHAT_TEMPLATE_PROBE_FAILED",
            f"{rec.record_id}: the sentinel appears {probe.count(SENTINEL)} times in "
            "the probe rendering, so the assistant body cannot be located.")
    head, _, tail = probe.partition(SENTINEL)

    if not full_render.startswith(head):
        die("CHAT_TEMPLATE_PREFIX_VIOLATION",
            f"{rec.record_id}: the template does not render the conversation prefix "
            "identically with and without the real assistant content.\n"
            f"--- head tail ---\n{head[-400:]}\n"
            f"--- full at same offset ---\n{full_render[:len(head)][-400:]}")

    # The generation prompt must be the conversation prefix, optionally plus the
    # empty thought channel. Anything else means the template changed shape and
    # the mask boundary can no longer be trusted.
    delta = prompt[len(head):] if prompt.startswith(head) else None
    if delta is None or delta not in ("", THOUGHT_OPENER):
        die("CHAT_TEMPLATE_PREFIX_VIOLATION",
            f"{rec.record_id}: the generation prompt is not the conversation prefix "
            "plus at most an empty thought channel, so assistant-only loss masking "
            f"cannot be derived safely.\n--- prompt tail ---\n{prompt[-400:]}\n"
            f"--- conversation prefix tail ---\n{head[-400:]}")

    body = full_render[len(head):]
    if not body.strip():
        die("EMPTY_ASSISTANT_TARGET",
            f"{rec.record_id}: the assistant turn renders to nothing.")
    return prompt + body, prompt


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
