"""Build a synthetic package with the real shape (91/11/12, multi-image) for CI."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthpkg")
COUNTS = {"train": 91, "validation": 11, "test": 12}
random.seed(0)

(ROOT / "hf_multimodal").mkdir(parents=True, exist_ok=True)
(ROOT / "images").mkdir(parents=True, exist_ok=True)

TARGET = (
    "A sample preparation cartridge comprising: a housing defining a first chamber; "
    "a valve assembly disposed within the housing and configured to selectively couple "
    "the first chamber to a second chamber; and a plunger movable along a longitudinal "
    "axis of the housing."
)

n = 0
for split, count in COUNTS.items():
    with (ROOT / "hf_multimodal" / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(count):
            n += 1
            fam = f"US{9000000 + n}"
            k = random.choice([1, 2, 3])
            imgs = []
            for j in range(k):
                rel = f"images/{fam}_p{j}.png"
                Image.new("RGB", (64, 80), (240, 240, 240)).save(ROOT / rel)
                imgs.append(rel)
            # The frozen release carries images only in the top-level array and
            # keeps message content as plain text, so mirror that exactly.
            content = "Draft one independent apparatus claim from these pages."
            fh.write(json.dumps({
                "id": f"{fam}__c1",
                "images": imgs,
                "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant",
                     "content": [{"type": "text", "text": TARGET}]},
                ],
                "metadata": {"family": fam, "split": split},
            }, ensure_ascii=False) + "\n")
print(f"synthetic package at {ROOT} ({n} records)")
