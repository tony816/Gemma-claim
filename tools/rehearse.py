#!/usr/bin/env python3
"""Run the whole training pipeline without weights, a GPU, or any billing.

Why this exists: in the previous run every pipeline bug was discovered for the
first time on a $1.50-2.00/hr GPU, one restart cycle at a time. All three of the
expensive ones - the chat-template prefix violation, LoRA targets re-matching
the vision tower, and the TrainingArguments signature change - needed no model
weights to reproduce. This reproduces them for nothing.

    python tools/rehearse.py

Exit code 0 means every gate that could run passed. A gate that could not run
prints SKIP with the reason and is counted separately: **a skip is not a pass**,
and the summary says so. Nothing here starts a GPU or touches the network except
to fetch a tokenizer and a config (tens of MB, no weights).
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tests"))

MODEL_ID = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
REVISION = os.environ.get("BASE_MODEL_REVISION", "main")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(gate: str, status: str, detail: str = "") -> None:
    results.append((gate, status, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip "}[status]
    print(f"[{mark}] {gate}" + (f" — {detail}" if detail else ""))


def gate(name):
    """Run a check. It returns a detail string, raises Skip, or raises to fail."""
    def deco(fn):
        def run(*a, **kw):
            try:
                record(name, PASS, fn(*a, **kw) or "")
            except Skip as exc:
                record(name, SKIP, str(exc))
            except Exception as exc:  # noqa: BLE001
                record(name, FAIL, f"{type(exc).__name__}: {exc}")
                if os.environ.get("REHEARSE_TRACE"):
                    traceback.print_exc()
        return run
    return deco


class Skip(Exception):
    pass


# --------------------------------------------------------------------------
# Gate 1 — the installed library versions, and whether TrainingArguments still
# accepts every argument the run depends on. Pure Python; always runs.
# --------------------------------------------------------------------------
@gate("versions")
def check_versions() -> str:
    import torch
    import transformers
    mods = [("torch", torch.__version__), ("transformers", transformers.__version__)]
    for name in ("accelerate", "peft", "safetensors"):
        try:
            mods.append((name, __import__(name).__version__))
        except Exception:  # noqa: BLE001
            mods.append((name, "MISSING"))
    missing = [n for n, v in mods if v == "MISSING"]
    if missing:
        raise RuntimeError(f"not installed: {missing}")
    return ", ".join(f"{n} {v}" for n, v in mods)


@gate("TrainingArguments signature")
def check_training_arguments() -> str:
    """The v5 reshuffle dropped warmup_ratio and renamed evaluation_strategy.
    Compare the names the run needs against the installed signature."""
    import transformers
    from transformers import TrainingArguments
    from train import ESSENTIAL_TRAINING_ARGS, TRAINING_ARG_ALIASES

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    missing = []
    for key in sorted(ESSENTIAL_TRAINING_ARGS):
        if key in accepted:
            continue
        if any(a in accepted for a in TRAINING_ARG_ALIASES.get(key, ())):
            continue
        missing.append(key)
    if missing:
        raise RuntimeError(
            f"transformers {transformers.__version__} does not accept {missing}; "
            "training would silently change. Pin a compatible release.")
    return f"all {len(ESSENTIAL_TRAINING_ARGS)} essential args accepted on {transformers.__version__}"


# --------------------------------------------------------------------------
# Gate 2 — chat template and assistant-only loss masking. Needs the real
# processor (tokenizer files only, no weights). Falls back to the offline fake
# so the logic is still exercised when the Hub is unreachable.
# --------------------------------------------------------------------------
_SPLITS: dict | None = None


def _dataset_splits() -> dict:
    """Load the dataset once, or return {} if there is none.

    common.die() raises SystemExit and prints a banner, so both are contained
    here rather than at every call site.
    """
    global _SPLITS
    if _SPLITS is not None:
        return _SPLITS
    import contextlib
    import io
    try:
        from common import find_package_root, load_all
        with contextlib.redirect_stdout(io.StringIO()):
            _SPLITS = load_all(find_package_root())
    except (Exception, SystemExit):  # noqa: BLE001
        _SPLITS = {}
    return _SPLITS


_TMP: Path | None = None


def _sample_record(n_images: int = 2):
    """A real record when the dataset is present, a synthetic one otherwise.

    Rehearsing against the actual data is what catches record-specific problems
    (a system turn the template rejects, an unusual part order), so the real one
    is preferred and the synthetic is only a fallback.
    """
    global _TMP
    from common import Record

    recs = _dataset_splits().get("train") or []
    if recs:
        return recs[0]

    if _TMP is None:
        import tempfile
        from PIL import Image
        _TMP = Path(tempfile.mkdtemp(prefix="rehearse-"))
        for i in range(n_images):
            Image.new("RGB", (64, 48), (255, 255, 255)).save(_TMP / f"fig{i}.png")

    images = [_TMP / f"fig{i}.png" for i in range(n_images)]
    target = "An apparatus comprising: a housing and a sensor disposed therein."
    return Record(
        index=0, split="train", record_id="rehearsal",
        messages=[
            {"role": "user", "content": (
                [{"type": "image"} for _ in range(n_images)]
                + [{"type": "text", "text": "Draft an independent apparatus claim."}])},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ],
        images=images,
        target=target,
    )


def _load_processor():
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION), "real"
    except Exception as exc:  # noqa: BLE001
        try:
            from fake_processor import FakeProcessor  # type: ignore
            return FakeProcessor(), f"fake ({type(exc).__name__})"
        except Exception:
            raise Skip(f"no processor available: {exc}") from exc


@gate("chat template renders a strict prefix")
def check_chat_template() -> str:
    """Gemma 4's generation prompt opens an empty thought channel that the full
    rendering does not contain. render_texts must reconcile the two."""
    from dataset import render_texts

    processor, kind = _load_processor()
    rec = _sample_record()
    full, prompt = render_texts(processor, rec)

    if not full.startswith(prompt):
        raise RuntimeError("prompt is not a prefix of the full sequence")
    body = full[len(prompt):]
    if not body.strip():
        raise RuntimeError("assistant target is empty")
    if rec.target.split()[0] not in body:
        raise RuntimeError("assistant text missing from the trained body")
    detail = f"{kind} processor, prompt {len(prompt)} chars, target {len(body)} chars"
    if kind != "real":
        raise Skip(detail + " — real processor unreachable, template not verified")
    return detail


@gate("loss is masked to the assistant span")
def check_label_masking() -> str:
    from dataset import encode_record

    processor, kind = _load_processor()
    if kind != "real":
        raise Skip("needs the real tokenizer to align labels")
    enc = encode_record(processor, _sample_record())
    labels = enc["labels"]
    labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    supervised = [i for i, v in enumerate(labels) if v != -100]
    if not supervised:
        raise RuntimeError("no supervised token — the whole sequence is masked")
    if supervised != list(range(supervised[0], supervised[-1] + 1)):
        raise RuntimeError("supervised tokens are not one contiguous span")
    if supervised[0] == 0:
        raise RuntimeError("supervision starts at token 0 — the prompt is being trained on")
    return f"{len(supervised)}/{len(labels)} tokens supervised, contiguous from {supervised[0]}"


# --------------------------------------------------------------------------
# Gate 3 — LoRA target discovery on a weightless model skeleton. init_empty_weights
# builds the full module tree on the meta device: no download of weights, no VRAM.
# --------------------------------------------------------------------------
@gate("LoRA targets avoid the vision tower")
def check_lora_targets() -> str:
    import torch
    from train import discover_lora_targets

    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION)
        with init_empty_weights():
            model = _model_from_config(cfg)
        kind = "real config"
    except Skip:
        raise
    except Exception as exc:  # noqa: BLE001
        model, kind = _toy_vlm(), f"toy skeleton ({type(exc).__name__})"

    targets = discover_lora_targets(model)
    if not targets:
        raise RuntimeError("no targets found")
    bad = [t for t in targets if any(s in t for s in
           ("vision", "audio", "multi_modal_projector", "embed_tokens", "lm_head"))]
    if bad:
        raise RuntimeError(f"{len(bad)} target(s) land outside the language tower, e.g. {bad[:3]}")
    if not all("." in t for t in targets):
        raise RuntimeError("targets are bare suffixes; PEFT would match them everywhere")
    detail = f"{len(targets)} modules from the {kind}"
    if kind != "real config":
        raise Skip(detail + " — real config unreachable, only the selection rule was checked")
    return detail


def _model_from_config(cfg):
    import transformers
    for name in ("Gemma4ForConditionalGeneration", "AutoModelForImageTextToText",
                 "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            return cls.from_config(cfg)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("no auto class could build this config")


def _toy_vlm():
    """A module tree shaped like Gemma 4: a vision tower and a language tower
    that use the same projection names. Enough to prove the selection rule."""
    import torch.nn as nn

    def block(dim):
        b = nn.Module()
        b.self_attn = nn.Module()
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(b.self_attn, p, nn.Linear(dim, dim, bias=False))
        b.mlp = nn.Module()
        for p in ("gate_proj", "up_proj", "down_proj"):
            setattr(b.mlp, p, nn.Linear(dim, dim, bias=False))
        return b

    root = nn.Module()
    root.model = nn.Module()
    root.model.vision_tower = nn.ModuleList([block(64) for _ in range(2)])
    root.model.language_model = nn.Module()
    root.model.language_model.layers = nn.ModuleList([block(128) for _ in range(3)])
    root.model.language_model.embed_tokens = nn.Linear(128, 128, bias=False)
    root.lm_head = nn.Linear(128, 128, bias=False)
    return root


# --------------------------------------------------------------------------
# Gate 4 — dataset contracts, when a package is present.
# --------------------------------------------------------------------------
@gate("dataset contracts")
def check_dataset() -> str:
    splits = _dataset_splits()
    if not splits:
        raise Skip("no dataset package found under the configured data root")
    counts = {k: len(v) for k, v in splits.items()}
    problems = []
    for split, recs in splits.items():
        for r in recs:
            if not (r.target or "").strip():
                problems.append(f"{r.record_id}: empty target")
            for p in r.images:
                if not Path(p).exists():
                    problems.append(f"{r.record_id}: missing image {p}")
    if problems:
        raise RuntimeError(f"{len(problems)} problem(s), e.g. {problems[:3]}")

    n_train = counts.get("train", 0)
    warn = ""
    if n_train and n_train < 500:
        warn = (f"  ⚠ {n_train} training records. The previous run collapsed at 91. "
                "Score the prompted baseline (tools/baseline.py) before spending a GPU.")
    return f"{counts}" + ("\n" + warn if warn else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    print(f"Rehearsing the pipeline for {MODEL_ID}@{REVISION} — no GPU, no weights.\n")
    check_versions()
    check_training_arguments()
    check_chat_template()
    check_label_masking()
    check_lora_targets()
    check_dataset()

    failed = [r for r in results if r[1] == FAIL]
    skipped = [r for r in results if r[1] == SKIP]
    passed = [r for r in results if r[1] == PASS]

    print(f"\n{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if skipped:
        print("\nSkipped gates were NOT verified. Re-run where these can execute:")
        for name, _, detail in skipped:
            print(f"  · {name} — {detail}")
    if failed:
        print("\nDo not start a GPU. Fix these first:")
        for name, _, detail in failed:
            print(f"  · {name} — {detail}")
        return 1
    if skipped:
        print("\nNo failures, but the run is only as verified as the gates that ran.")
        return 0
    print("\nEvery gate passed. The pipeline is safe to put on a GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
