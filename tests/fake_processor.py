"""A minimal stand-in for Gemma4Processor: enough to exercise masking logic."""
from __future__ import annotations

import torch

IMG_TOKENS = 280
PAD_ID = 0


class _Tok:
    pad_token_id = PAD_ID

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


class FakeProcessor:
    """Renders a gemma-like template and tokenises by whitespace.

    An <image> placeholder expands to IMG_TOKENS ids, mirroring the real
    processor, so prefix and label-mask arithmetic is tested realistically.
    """

    tokenizer = _Tok()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = []
        for m in messages:
            # Gemma renders the assistant role as "model"; mirror that exactly,
            # otherwise the generation prompt would not be a prefix of the
            # rendered conversation and the mask check would trip on the fake.
            role = "model" if m["role"] == "assistant" else m["role"]
            out.append(f"<start_of_turn>{role}\n")
            for part in m["content"]:
                out.append("<image>" if part["type"] == "image" else part.get("text", ""))
            out.append("<end_of_turn>\n")
        if add_generation_prompt:
            # Gemma 4 opens an empty thought channel in non-thinking mode, so the
            # generation prompt is deliberately NOT a prefix of the plain
            # conversation rendering. Mirror that, otherwise the fake would not
            # exercise the code path that reconciles the two.
            out.append("<start_of_turn>model\n<|channel>thought\n<channel|>")
        return "".join(out)

    def __call__(self, text=None, images=None, return_tensors="pt", padding=False):
        import re
        import zlib

        # Real tokenisers match special markers atomically, which is what keeps
        # a concatenated prompt+body tokenising as prompt tokens followed by
        # body tokens. Model that here, otherwise the fake would merge across
        # the boundary and fail a check the real tokeniser passes.
        special = re.compile(r"<\|?[a-z_]+\|?>")
        chunks = text.split("<image>")
        ids: list[int] = []
        for i, chunk in enumerate(chunks):
            for piece in filter(None, re.split(r"(" + special.pattern + r")", chunk)):
                if special.fullmatch(piece):
                    ids.append(zlib.crc32(piece.encode()) % 1000 + 100000)
                else:
                    ids.extend(zlib.crc32(w.encode()) % 10000 + 10 for w in piece.split())
            if i < len(chunks) - 1:
                ids.extend([7] * IMG_TOKENS)
        t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        out = {"input_ids": t, "attention_mask": torch.ones_like(t)}
        if images:
            flat = images[0] if images and isinstance(images[0], list) else images
            out["pixel_values"] = torch.zeros(len(flat), 3, 16, 16)
        return out
