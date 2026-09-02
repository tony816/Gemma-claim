"""Build a synthetic package with the real shape (554/65/75, multi-image) for CI.

The shape tracks the frozen release the pipeline is pinned to: the split counts
preflight.py enforces, images capped at 8 per record, and a Korean-majority mix
(the release is 612 Korean of 694), so the loss mask and the collator are
exercised on the text the run will actually train on.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthpkg")
COUNTS = {"train": 554, "validation": 65, "test": 75}
# 41% of the release sits at the 8-image cap, so the synthetic mix has to
# reach it too -- that is where the longest sequences come from.
IMAGE_COUNTS = [1, 2, 3, 5, 8]
# The release is 88% Korean. Scoring or masking that assumed English would
# pass on an English-only fixture and fail on the real data.
KOREAN_SHARE = 0.88
random.seed(0)

(ROOT / "hf_multimodal").mkdir(parents=True, exist_ok=True)
(ROOT / "images").mkdir(parents=True, exist_ok=True)

TARGET_EN = (
    "A sample preparation cartridge comprising: a housing defining a first chamber; "
    "a valve assembly disposed within the housing and configured to selectively couple "
    "the first chamber to a second chamber; and a plunger movable along a longitudinal "
    "axis of the housing."
)
TARGET_KO = (
    "\uc81c1 \ucc54\ubc84\ub97c \ud55c\uc815\ud558\ub294 \ud558\uc6b0\uc9d5; "
    "\uc0c1\uae30 \ud558\uc6b0\uc9d5 \ub0b4\ubd80\uc5d0 \ubc30\uce58\ub418\uace0, "
    "\uc0c1\uae30 \uc81c1 \ucc54\ubc84\ub97c \uc81c2 \ucc54\ubc84\uc5d0 "
    "\uc120\ud0dd\uc801\uc73c\ub85c \uc5f0\uacb0\ud558\ub3c4\ub85d "
    "\uad6c\uc131\ub418\ub294 \ubc38\ube0c \uc870\ub9bd\uccb4; \ubc0f "
    "\uc0c1\uae30 \ud558\uc6b0\uc9d5\uc758 \uc885\ubc29\ud5a5 \ucd95\uc744 "
    "\ub530\ub77c \uc774\ub3d9 \uac00\ub2a5\ud55c \ud50c\ub7f0\uc800\ub97c "
    "\ud3ec\ud568\ud558\ub294, \uc2dc\ub8cc \uc900\ube44 \uce74\ud2b8\ub9ac\uc9c0."
)

n = 0
for split, count in COUNTS.items():
    with (ROOT / "hf_multimodal" / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(count):
            n += 1
            fam = f"US{9000000 + n}"
            k = random.choice(IMAGE_COUNTS)
            korean = random.random() < KOREAN_SHARE
            target = TARGET_KO if korean else TARGET_EN
            imgs = []
            for j in range(k):
                rel = f"images/{fam}_p{j}.png"
                Image.new("RGB", (64, 80), (240, 240, 240)).save(ROOT / rel)
                imgs.append(rel)
            # This mirrors tools/convert_release.py's output, not the release
            # itself: that is what DATA_ROOT points at and what preflight.py
            # reads. Both contents are plain strings, image placeholders are
            # absent (common.normalise_record materialises them in array order),
            # and the record carries exactly {id, images, messages} -- a
            # `metadata` key here would trip ORACLE_FIELD_IN_MODEL_INPUT.
            content = ("\ub3c4\uba74\uc744 \uadfc\uac70\ub85c "
                       "\ub3c5\ub9bd \uc7a5\uce58\ud56d 1\uac1c\ub97c "
                       "\uc791\uc131\ud558\ub77c." if korean else
                       "Draft one independent apparatus claim from these pages.")
            fh.write(json.dumps({
                "id": f"{fam}__c1",
                "images": imgs,
                "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": target},
                ],
            }, ensure_ascii=False) + "\n")
print(f"synthetic package at {ROOT} ({n} records)")
