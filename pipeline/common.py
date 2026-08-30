"""Shared paths, seeding, logging and dataset-schema normalisation.

The dataset is a frozen reference release (v1.1.2-independent-oracle-clean).
Nothing in this package ever writes into the extracted dataset tree; model
specific conversions are emitted under OUT/converted/ instead.
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", WORKSPACE / "data" / "v112"))
OUT = Path(os.environ.get("OUT_DIR", WORKSPACE / "outputs"))

SPLITS = ("train", "validation", "test")

# Only these inputs may reach the model. Everything else in the package is
# oracle / provenance material and would leak the answer.
ALLOWED_SUBDIRS = ("hf_multimodal", "images")
FORBIDDEN_SUBDIRS = (
    "metadata/source_oracle_pages",
    "canonical",
    "excluded",
    "evaluation",
)

SEED = int(os.environ.get("SEED", "42"))


def log(msg: str) -> None:
    print(f"[{os.environ.get('STAGE', 'pipeline')}] {msg}", flush=True)


def die(code: str, msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"\n### HARD_FAILURE {code}\n{msg}\n", flush=True)
    sys.exit(2)


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def find_package_root(base: Path = DATA_ROOT) -> Path:
    """The package root is the directory that directly contains hf_multimodal/."""
    if (base / "hf_multimodal").is_dir():
        return base
    for cand in sorted(base.glob("*")):
        if cand.is_dir() and (cand / "hf_multimodal").is_dir():
            return cand
    for cand in sorted(base.glob("*/*")):
        if cand.is_dir() and (cand / "hf_multimodal").is_dir():
            return cand
    die(
        "DATASET_LAYOUT_UNEXPECTED",
        f"No directory containing hf_multimodal/ found under {base}. "
        f"Top level entries: {[p.name for p in sorted(base.glob('*'))][:40]}",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                die("DATASET_JSONL_MALFORMED", f"{path}:{lineno}: {exc}")
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Schema normalisation
# --------------------------------------------------------------------------

@dataclass
class Record:
    """One training example, normalised away from the on-disk encoding.

    `messages` keeps the original role order. Image placeholders stay in the
    exact position and order they occupied in the source record, because the
    array order is part of the data contract.
    """

    index: int
    split: str
    messages: list[dict[str, Any]]      # normalised chat turns
    images: list[Path]                  # absolute paths, source order preserved
    target: str                         # assistant target text
    record_id: str
    raw_keys: list[str] = field(default_factory=list)


def _content_to_parts(content: Any) -> list[dict[str, Any]]:
    """Normalise a message `content` into a list of {type: image|text} parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        content = [content]
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("image", "image_url", "input_image"):
            parts.append({"type": "image"})
        elif ptype in ("text", "input_text", None):
            txt = part.get("text", part.get("value", ""))
            parts.append({"type": "text", "text": txt})
        else:
            # Unknown part types are preserved as text so nothing is dropped.
            parts.append({"type": "text", "text": json.dumps(part, ensure_ascii=False)})
    return parts


def _text_of(parts: Iterable[dict[str, Any]]) -> str:
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def _count_images(parts: Iterable[dict[str, Any]]) -> int:
    return sum(1 for p in parts if p.get("type") == "image")


def normalise_record(raw: dict[str, Any], idx: int, split: str, pkg_root: Path) -> Record:
    msgs = raw.get("messages") or raw.get("conversations") or raw.get("conversation")
    if not msgs:
        die(
            "DATASET_SCHEMA_UNEXPECTED",
            f"{split}[{idx}] has no messages/conversations key. Keys: {sorted(raw)}",
        )

    norm: list[dict[str, Any]] = []
    for m in msgs:
        role = m.get("role") or m.get("from") or ""
        role = {"human": "user", "gpt": "assistant", "model": "assistant"}.get(role, role)
        content = m.get("content", m.get("value"))
        norm.append({"role": role, "content": _content_to_parts(content)})

    if not norm or norm[-1]["role"] != "assistant":
        die(
            "DATASET_SCHEMA_UNEXPECTED",
            f"{split}[{idx}] last message role is "
            f"{norm[-1]['role'] if norm else 'N/A'}, expected 'assistant'.",
        )

    target = _text_of(norm[-1]["content"])

    raw_images = raw.get("images") or raw.get("image") or []
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    images = [(pkg_root / str(p)).resolve() for p in raw_images]

    rid = str(
        raw.get("id")
        or raw.get("record_id")
        or (raw.get("metadata") or {}).get("record_id")
        or f"{split}-{idx:04d}"
    )
    return Record(
        index=idx,
        split=split,
        messages=norm,
        images=images,
        target=target,
        record_id=rid,
        raw_keys=sorted(raw),
    )


def load_split(split: str, pkg_root: Path) -> list[Record]:
    path = pkg_root / "hf_multimodal" / f"{split}.jsonl"
    if not path.is_file():
        die("DATASET_SPLIT_MISSING", f"Missing {path}")
    return [
        normalise_record(raw, i, split, pkg_root)
        for i, raw in enumerate(read_jsonl(path))
    ]


def load_all(pkg_root: Path) -> dict[str, list[Record]]:
    return {s: load_split(s, pkg_root) for s in SPLITS}


def build_chat_messages(rec: Record, for_prompt: bool) -> list[dict[str, Any]]:
    """Chat turns for the processor.

    for_prompt=True drops the final assistant turn so the caller can render a
    generation prompt; the rendered prompt is then a strict textual prefix of
    the full conversation, which is what the assistant-only loss mask relies on.
    """
    msgs = rec.messages[:-1] if for_prompt else rec.messages
    return [{"role": m["role"], "content": list(m["content"])} for m in msgs]


def images_of(rec: Record) -> list[Any]:
    from PIL import Image

    out = []
    for p in rec.images:
        with Image.open(p) as im:
            out.append(im.convert("RGB").copy())
    return out


def load_vlm(model_id: str, revision: str = "main", **kw):
    """Load the vision-language model through whichever auto class registers it.

    Gemma 4 is exposed as Gemma4ForConditionalGeneration; the auto-class mapping
    it lands in differs between transformers releases, so try the specific class
    first and fall back rather than pinning one path.
    """
    import transformers

    errors = []
    cls_name = "Gemma4ForConditionalGeneration"
    if hasattr(transformers, cls_name):
        try:
            return getattr(transformers, cls_name).from_pretrained(
                model_id, revision=revision, **kw
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{cls_name}: {exc}")

    for auto in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                 "AutoModelForCausalLM"):
        if not hasattr(transformers, auto):
            continue
        try:
            return getattr(transformers, auto).from_pretrained(
                model_id, revision=revision, **kw
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{auto}: {exc}")

    die("MODEL_LOAD_FAILED", f"Could not load {model_id}@{revision}:\n" + "\n".join(errors))
