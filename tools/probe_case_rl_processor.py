"""Exercise the new draft targets with the real cached processor, CPU only.

No model weights are loaded. This is an SFT input/masking probe, not a GRPO
trainer compatibility test or proof of output quality.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from tools.prepare_case_rl import read_jsonl, save_json


def main():
    import torch
    from transformers import AutoProcessor
    from common import normalise_record
    from dataset import encode_record

    torch.set_num_threads(2)
    revision = "842da3794eaa0b77d5f08bae87a17459d91ff475"
    processor = AutoProcessor.from_pretrained("google/gemma-4-31B-it",
                                             revision=revision, local_files_only=True)
    out, source = ROOT / "data/case_rl", ROOT / "data/rl_source_v2"
    tasks = {r["source_record_id"]: r for r in read_jsonl(out / "tasks.train.jsonl")}
    results = []
    for i, draft in enumerate(read_jsonl(out / "claimsets.visual_drafts.jsonl")):
        task = tasks[draft["source_record_id"]]
        response = json.dumps(draft["claimset_target"], ensure_ascii=False, separators=(",", ":"))
        raw = {"id": draft["source_record_id"], "images": task["images"],
               "messages": task["prompt"] + [{"role": "assistant", "content": response}]}
        record = normalise_record(raw, i, "train", source)
        encoded = encode_record(processor, record)
        labels = encoded["labels"][0]
        assert bool((labels[:encoded["_prompt_len"]] == -100).all())
        assert bool((labels[encoded["_prompt_len"]:] != -100).any())
        results.append({"record_id": record.record_id, "images": encoded["_n_images"],
                        "prompt_tokens_including_images": encoded["_prompt_len"],
                        "target_tokens": encoded["_target_len"],
                        "total_tokens": encoded["_total_len"],
                        "assistant_only_mask_verified": True, "device": "cpu"})
    report = {"processor_revision": revision, "model_weights_loaded": False,
              "passed": True, "probe": "four_draft_SFT_records_only",
              "grpo_compatibility_verified": False, "training_ready": False,
              "records": results}
    save_json(out / "processor_probe.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
