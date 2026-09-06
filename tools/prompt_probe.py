"""Does the Korean system prompt suppress reference numerals on the tuned model?

The fine-tune never saw a system turn: the training prompt is one English line
and the drawings. Telling the tuned model "도면 부호를 쓰지 마십시오" at serving
time is therefore an instruction from outside its training distribution, and it
can go either way -- ignored, or obeyed at the cost of the gains the fine-tune
bought. Both outcomes change what we should ship, so measure rather than assume.

Run on the pod, after the pipeline is done, with the model already cached.
"""
import json
import os
import re
import sys
from pathlib import Path

CODE = Path("/workspace/code")
sys.path.insert(0, str(CODE / "pipeline"))
sys.path.insert(0, str(CODE / "serving"))

import torch  # noqa: E402
from common import find_package_root, load_all, load_vlm  # noqa: E402
from dataset import render_texts, _call_processor  # noqa: E402
from claim_prompt import system_prompt, sanitise  # noqa: E402
from evaluate import claim_form_checks  # noqa: E402
import sacrebleu  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from peft import PeftModel  # noqa: E402

MODEL_ID = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
ADAPTER = Path("/workspace/outputs/final_model_or_adapter")
N = int(os.environ.get("PROBE_N", "12"))
MAX_NEW = int(os.environ.get("PROBE_MAX_NEW", "512"))

REF_NUM = re.compile(
    r"[가-힣A-Za-z\)]\s?\(\s*\d{1,4}[a-zA-Z]?(?:\s*[,;]\s*\d{1,4}[a-zA-Z]?)*\s*\)"
    r"|[가-힣]\s+\d{1,4}(?=\s*[은는이가을를과와의및;,.])"
)


def rep8(text):
    t = text.split()
    if len(t) < 9:
        return 0.0
    g = [" ".join(t[i:i + 8]) for i in range(len(t) - 7)]
    return round(1 - len(set(g)) / len(g), 3)


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    recs = load_all(find_package_root())["validation"][:N]
    print(f"loading {MODEL_ID} + adapter", flush=True)
    model = load_vlm(MODEL_ID, "main", dtype=torch.bfloat16,
                     device_map={"": 0}, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    model.config.use_cache = True

    def generate(rec, with_system):
        """with_system=False reproduces training exactly; True prepends the
        serving system turn, which is the only thing that differs."""
        _, prompt_text = render_texts(processor, rec)
        if with_system:
            sysmsg = [{"role": "system",
                       "content": [{"type": "text", "text": system_prompt("ko")}]}]
            head = processor.apply_chat_template(sysmsg, tokenize=False,
                                                 add_generation_prompt=False)
            prompt_text = head + prompt_text
        from common import images_of
        enc = _call_processor(processor, prompt_text, images_of(rec))
        enc = {k: (v.to(model.device) if torch.is_tensor(v) else v)
               for k, v in enc.items()}
        plen = int(enc["input_ids"].shape[-1])
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=processor.tokenizer.pad_token_id or 0)
        return processor.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()

    rows = []
    for i, rec in enumerate(recs, 1):
        row = {"record_id": rec.record_id, "reference": rec.target}
        for tag, flag in (("train_prompt", False), ("with_system", True)):
            g = generate(rec, flag)
            cleaned, removed = sanitise(g, "ko")
            row[tag] = {
                "text": g,
                "numerals": bool(REF_NUM.search(g)),
                "numerals_after_sanitise": bool(REF_NUM.search(cleaned)),
                "chrf": round(sacrebleu.sentence_chrf(g, [rec.target]).score, 2),
                "chrf_sanitised": round(sacrebleu.sentence_chrf(cleaned, [rec.target]).score, 2),
                "words": len(g.split()),
                "rep8": rep8(g),
                "well_formed": claim_form_checks(g),
            }
        rows.append(row)
        a, b = row["train_prompt"], row["with_system"]
        print(f"[{i}/{len(recs)}] {rec.record_id}: "
              f"train_prompt chrf={a['chrf']:5.2f} num={int(a['numerals'])} w={a['words']:3d} | "
              f"with_system chrf={b['chrf']:5.2f} num={int(b['numerals'])} w={b['words']:3d}",
              flush=True)

    out = Path("/workspace/outputs/prompt_probe.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== summary over", len(rows), "validation records ===")
    print(f"{'condition':14s} {'mean chrF':>9s} {'chrF(san)':>9s} {'numerals':>9s} "
          f"{'after san':>9s} {'mean words':>10s} {'looping':>8s}")
    for tag in ("train_prompt", "with_system"):
        rs = [r[tag] for r in rows]
        n = len(rs)
        print(f"{tag:14s} {sum(x['chrf'] for x in rs)/n:9.2f} "
              f"{sum(x['chrf_sanitised'] for x in rs)/n:9.2f} "
              f"{sum(x['numerals'] for x in rs):6d}/{n:<3d}"
              f"{sum(x['numerals_after_sanitise'] for x in rs):6d}/{n:<3d}"
              f"{sum(x['words'] for x in rs)/n:10.1f} "
              f"{sum(1 for x in rs if x['rep8'] > 0.3):5d}/{n}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
