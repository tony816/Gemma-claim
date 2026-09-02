"""Offline checks of the data contract and the assistant-only loss mask.

Runs against a synthetic package with the real split shape, so it needs no GPU,
no network and no base-model download.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE / "pipeline"))
sys.path.insert(0, str(HERE))

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pipetest-"))
    pkg = tmp / "pkg"
    subprocess.run([sys.executable, str(HERE / "make_synthetic.py"), str(pkg)], check=True)

    os.environ["DATA_ROOT"] = str(pkg)
    os.environ["OUT_DIR"] = str(tmp / "outputs")

    import common
    common.DATA_ROOT = pkg
    common.OUT = tmp / "outputs"

    from common import SPLITS, find_package_root, load_all
    from dataset import IGNORE_INDEX, encode_record
    from fake_processor import IMG_TOKENS, FakeProcessor

    print("\n[1] package discovery + normalisation")
    root = find_package_root(pkg)
    check("find_package_root locates hf_multimodal", root == pkg, str(root))
    data = load_all(root)
    check("split sizes 554/65/75",
          {s: len(data[s]) for s in SPLITS} == {"train": 554, "validation": 65, "test": 75},
          str({s: len(data[s]) for s in SPLITS}))
    r0 = data["train"][0]
    check("images resolved to existing files", all(p.is_file() for p in r0.images))
    check("assistant target non-empty", bool(r0.target.strip()))
    check("roles normalised", [m["role"] for m in r0.messages] == ["user", "assistant"])

    print("\n[2] image ordering is preserved end to end")
    raw_first = [str(p.name) for p in r0.images]
    check("image order matches source array",
          raw_first == sorted(raw_first, key=lambda n: int(n.split('_p')[1].split('.')[0])),
          str(raw_first))

    print("\n[3] assistant-only loss mask")
    proc = FakeProcessor()
    enc = encode_record(proc, r0)
    ids, labels = enc["input_ids"][0], enc["labels"][0]
    plen = enc["_prompt_len"]
    check("labels align with input_ids", labels.shape == ids.shape)
    check("every prompt token is ignored", bool((labels[:plen] == IGNORE_INDEX).all()))
    check("no target token is ignored", bool((labels[plen:] != IGNORE_INDEX).all()))
    check("target tokens equal input tokens", bool((labels[plen:] == ids[plen:]).all()))
    check("target length is positive", enc["_target_len"] > 0, str(enc["_target_len"]))
    n_img_tokens = int((ids == 7).sum())
    check("image tokens expand per image and sit inside the masked prompt",
          n_img_tokens == IMG_TOKENS * len(r0.images)
          and int((ids[:plen] == 7).sum()) == n_img_tokens,
          f"{n_img_tokens} vs {IMG_TOKENS * len(r0.images)}")

    print("\n[4] multi-image records")
    multi = [r for r in data["train"] if len(r.images) >= 2][:3]
    check("dataset contains multi-image records", bool(multi))
    for rec in multi:
        e = encode_record(proc, rec)
        check(f"{rec.record_id}: all {len(rec.images)} images encoded",
              int((e["input_ids"][0] == 7).sum()) == IMG_TOKENS * len(rec.images))
        check(f"{rec.record_id}: pixel_values row per image",
              e["pixel_values"].shape[0] == len(rec.images))

    print("\n[5] collator")
    from dataset import Collator
    batch = Collator(pad_token_id=0)([encode_record(proc, r) for r in data["train"][:2]])
    check("batch dim is 2", batch["input_ids"].shape[0] == 2)
    check("padding is masked out of the loss",
          bool((batch["labels"][batch["input_ids"] == 0] == IGNORE_INDEX).all()))

    print("\n[6] preflight gate on the synthetic package")
    env = {**os.environ, "DATA_ROOT": str(pkg), "OUT_DIR": str(tmp / "outputs"),
           "PYTHONPATH": str(CODE / "pipeline")}
    p = subprocess.run([sys.executable, str(CODE / "pipeline" / "preflight.py")],
                       capture_output=True, text=True, env=env)
    check("preflight passes clean data", p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:])

    print("\n[7] preflight rejects a missing image")
    victim = next(pkg.glob("images/*.png"))
    victim.rename(victim.with_suffix(".hidden"))
    p = subprocess.run([sys.executable, str(CODE / "pipeline" / "preflight.py")],
                       capture_output=True, text=True, env=env)
    check("missing image is a hard failure",
          p.returncode != 0 and "MISSING_IMAGE" in (p.stdout + p.stderr))
    victim.with_suffix(".hidden").rename(victim)

    print("\n[8] preflight rejects an empty target")
    tf = pkg / "hf_multimodal" / "validation.jsonl"
    orig = tf.read_text()
    import json as _json
    rows = [_json.loads(l) for l in orig.splitlines() if l.strip()]
    rows[0]["messages"][-1]["content"] = [{"type": "text", "text": "   "}]
    tf.write_text("\n".join(_json.dumps(r, ensure_ascii=False) for r in rows))
    p = subprocess.run([sys.executable, str(CODE / "pipeline" / "preflight.py")],
                       capture_output=True, text=True, env=env)
    check("empty target is a hard failure",
          p.returncode != 0 and "EMPTY_ASSISTANT_TARGET" in (p.stdout + p.stderr))
    tf.write_text(orig)

    print("\n[9] preflight rejects a leaked record across splits")
    trf = pkg / "hf_multimodal" / "train.jsonl"
    orig_tr = trf.read_text()
    val_rows = [_json.loads(l) for l in (pkg / "hf_multimodal" / "validation.jsonl").read_text().splitlines() if l.strip()]
    tr_rows = [_json.loads(l) for l in orig_tr.splitlines() if l.strip()]
    tr_rows[0]["id"] = val_rows[0]["id"]
    trf.write_text("\n".join(_json.dumps(r, ensure_ascii=False) for r in tr_rows))
    p = subprocess.run([sys.executable, str(CODE / "pipeline" / "preflight.py")],
                       capture_output=True, text=True, env=env)
    check("cross-split record id is a hard failure",
          p.returncode != 0 and "TEST_LEAKAGE" in (p.stdout + p.stderr))
    trf.write_text(orig_tr)

    print("\n[10] per-language reporting agrees between evaluate and baseline")
    # 88% of the release is Korean, so a single aggregate hides the smaller
    # group entirely. The handoff requires the two reported separately, and
    # requires evaluate.py and tools/baseline.py to report them the same way.
    sys.path.insert(0, str(CODE / "serving"))
    from metrics import by_language, claim_form_checks, language_drift

    KO = "\uc81c1 \ud558\uc6b0\uc9d5; \uc0c1\uae30 \ud558\uc6b0\uc9d5 \ub0b4\ubd80\uc5d0 \ubc30\uce58\ub418\ub294 \uc13c\uc11c\ubd80\ub97c \ud3ec\ud568\ud558\ub294, \uc7a5\uce58."
    EN = "An apparatus comprising: a housing and a sensor disposed therein."
    mk = lambda ref, pred: {"reference": ref, "prediction": pred,
                            "checks": claim_form_checks(pred)}
    rows = [mk(KO, KO), mk(KO, KO), mk(KO, EN), mk(EN, EN)]
    bl = by_language(rows)
    check("both languages are reported separately", sorted(bl) == ["en", "ko"], str(sorted(bl)))
    check("records are bucketed by reference language",
          bl.get("ko", {}).get("n") == 3 and bl.get("en", {}).get("n") == 1)
    check("answering in the wrong language is counted",
          language_drift(rows) == 1, str(language_drift(rows)))
    # tools/baseline.py builds each by_language entry as
    # {"n": ..., **text_metrics(...), **qualitative_summary(...)}; evaluate.py
    # has to emit the same keys or the two cannot be compared side by side.
    baseline_keys = {"n", "rougeL_f", "chrf", "empty_responses", "looks_dependent",
                     "missing_apparatus_noun", "missing_transition_phrase",
                     "unterminated_no_final_period", "excessive_repetition_gt_0_3",
                     "well_formed_independent_claims", "well_formed_rate", "mean_words"}
    missing = sorted(baseline_keys - set(bl.get("ko", {})))
    check("evaluate's by_language matches baseline's shape", not missing, str(missing))

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
