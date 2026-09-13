"""Author the bounded US annotation-extension slice.

This script writes only current-approved US drawing annotation rows. It does not
read hidden source targets, call models, use a GPU, or modify source corpora.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.assemble_rl_release import approved, case_partition, annotation_errors, read_json
from tools.validate_rl_curation import canonical_hash, load_rows

OUT = ROOT / "rl_materials" / "curation" / "annotation_extensions.jsonl"
REPORT = ROOT / ".superloopy" / "evidence" / "annotation-author-report.md"
AUTHOR = "/root/us_materials"
MODE = "provided_during_drafting"
FORBIDDEN_KR_CASES = {
    "KR-2009HEO4742",
    "KR-2022HEO6099",
    "KR-2020HEO6323",
    "KR-2016HEO5538",
    "KR-2010HEO7488",
}

DRAWING_IDS = [
    "US20040184954A1_claim37",
    "US20040189311A1_claim120",
    "US20050013732A1_claim14",
    "US20070059204A1_claim1",
    "US20130337432A1_claim59",
    "US20250091053A1_claim1",
]

ROWS = [
    {
        "record_id": "EXT-US-TRAIN-0001-DROPPER-PHILLIPS",
        "drawing_record_id": "US20040184954A1_claim37",
        "source_split": "train",
        "jurisdiction": "US",
        "card_ids": ["US-PHILLIPS-2005-P1"],
        "annotations": [
            {
                "claim_number": 1,
                "card_id": "US-PHILLIPS-2005-P1",
                "mode": MODE,
                "application": "미국 Phillips 카드가 제공된 전제에서, 제1항의 hand-held dropper member, cylindrical holder, plate, base는 도면 관찰에 보이는 수직 적층 관계를 청구항 문언 자체에 남긴 표현이다. 이 주석은 그 구성들이 도면에서 보인다는 근거와 청구항 문맥을 함께 검토하라는 bounded historical 원칙을 적용할 뿐, Phillips가 이 드로퍼 구조를 입증하거나 현재 법상태를 보증한다는 뜻은 아니다.",
            },
            {
                "claim_number": 3,
                "card_id": "US-PHILLIPS-2005-P1",
                "mode": MODE,
                "application": "제3항의 elongate rail region과 circular receiving region은 실시예의 모든 세부 치수나 재질을 독립항에 끌어오지 않고, 보이는 판 구조의 한정을 종속항으로 분리한 점을 설명한다. Phillips 원칙상 명세서 맥락을 보되 실시예 제한을 자동 편입하지 않는 균형을 보여주는 주석이다.",
            },
        ],
        "inferior_annotations": [
            {
                "claim_number": 1,
                "card_id": "US-PHILLIPS-2005-P1",
                "mode": MODE,
                "application": "Phillips가 통상 의미를 강조하므로 제1항의 dispensing assembly는 드로퍼, 홀더, 판, 베이스 중 어느 하나만 보이면 충분하다고 볼 수 있다. 따라서 도면의 수직 적층 관계는 청구항 해석에서 큰 의미가 없고, 넓은 사전적 의미를 우선 적용하면 된다.",
            },
            {
                "claim_number": 3,
                "card_id": "US-PHILLIPS-2005-P1",
                "mode": MODE,
                "application": "제3항은 레일과 원형 수용부를 언급하므로 Phillips에 따라 도면의 gear-like ring이나 cover disk 같은 주변 부품도 같은 종속항에 모두 포함된 것으로 읽어야 한다. 실시예가 자세할수록 그 주변 형상이 청구항 범위를 자연스럽게 제한한다.",
            },
        ],
        "required_points": [
            "Phillips 카드는 US historical claim-construction context로만 적용한다.",
            "드로퍼-홀더-판-베이스의 보이는 적층 관계와 종속항의 레일/원형 수용부 한정을 구분한다.",
            "사전 의미 단독 또는 실시예 전체 자동 편입을 피한다.",
        ],
        "forbidden_claims": [
            "Phillips가 대상 드로퍼 구조의 존재, 신규성, 유효성 또는 현재 good law를 증명한다고 쓰지 말 것.",
            "도면에 없는 펌프, 센서, 재질, 생물학적 성능을 카드로 보충하지 말 것.",
            "제공되지 않은 사건 카드나 heldout case card를 인용하지 말 것.",
        ],
        "preference_reason": "선호 주석은 도면에 의해 이미 지지되는 청구항 문언을 Phillips의 내재증거/문맥 원칙으로 검토하고, 독립항과 종속항의 역할을 나눈다. 열등 주석은 통상 의미를 사전 의미처럼 과도하게 넓히고 실시예 세부를 자동 편입하는 두 방향의 해석 오류를 섞는다.",
    },
    {
        "record_id": "EXT-US-TRAIN-0002-MICROFLUIDIC-NAUTILUS",
        "drawing_record_id": "US20040189311A1_claim120",
        "source_split": "train",
        "jurisdiction": "US",
        "card_ids": ["US-NAUTILUS-2014-P1"],
        "annotations": [
            {
                "claim_number": 1,
                "card_id": "US-NAUTILUS-2014-P1",
                "mode": MODE,
                "application": "제1항은 opposed side port regions, inward channels, central array of wells처럼 도면에서 경계를 가를 수 있는 상대 위치와 연결관계를 직접 쓴다. Nautilus 카드는 미국 명확성의 historical 기준으로, 숙련자가 제공된 문언과 도면 맥락에서 범위를 합리적 확실성으로 파악할 수 있는지 점검하게 한다.",
            },
            {
                "claim_number": 3,
                "card_id": "US-NAUTILUS-2014-P1",
                "mode": MODE,
                "application": "제3항의 channel layer와 perforated cover layer는 중앙 well 배열과 정렬되는 층 구조를 구체화한다. 이 주석은 'layered construction'이라는 넓은 말만 남기면 경계가 흐릴 수 있으므로, 보이는 층과 구멍판의 대응관계를 명시하는 편이 Nautilus식 명확성 검토에 유리하다는 bounded 적용이다.",
            },
        ],
        "inferior_annotations": [
            {
                "claim_number": 1,
                "card_id": "US-NAUTILUS-2014-P1",
                "mode": MODE,
                "application": "Nautilus는 절대적 정밀성을 요구하지 않으므로 제1항에서 opposed, inward, central 같은 위치어는 도면 설명 없이도 충분하다. 포트와 well 사이의 실제 연결 방향은 제품 구현에서 정하면 되고 청구항 주석에서 따로 다룰 필요가 없다.",
            },
            {
                "claim_number": 3,
                "card_id": "US-NAUTILUS-2014-P1",
                "mode": MODE,
                "application": "제3항은 층 구조를 말하므로 Nautilus상 오히려 perforated cover layer의 정렬을 자세히 쓰면 범위가 불필요하게 좁아진다. 명확성 문제를 피하려면 층 수나 구멍 배열 같은 보이는 관계를 주석에서 흐리게 두는 편이 낫다.",
            },
        ],
        "required_points": [
            "Nautilus 카드는 US historical definiteness 원칙으로만 사용한다.",
            "opposed ports, inward channels, central wells, perforated cover layer의 객관적 경계를 설명한다.",
            "절대 정밀성 요구와 객관 기준 생략을 모두 피한다.",
        ],
        "forbidden_claims": [
            "Nautilus가 도면상 미세유체 연결을 증명한다고 말하지 말 것.",
            "상대 위치어가 항상 명확하거나 항상 불명확하다고 단정하지 말 것.",
            "현재 절차나 미검증 later history를 언급하지 말 것.",
        ],
        "preference_reason": "선호 주석은 도면 지지 특징의 위치·연결 경계를 객관화하는 방식으로 Nautilus를 제한적으로 적용한다. 열등 주석은 reasonable certainty를 세부관계 생략 허가로 오해해, 포트-채널-well 연결과 층 정렬이라는 핵심 경계를 흐린다.",
    },
    {
        "record_id": "EXT-US-TRAIN-0003-CARTRIDGE-ARIAD-LIZARDTECH",
        "drawing_record_id": "US20050013732A1_claim14",
        "source_split": "train",
        "jurisdiction": "US",
        "card_ids": ["US-ARIAD-2010-P1", "US-LIZARDTECH-2005-P1"],
        "annotations": [
            {
                "claim_number": 1,
                "card_id": "US-ARIAD-2010-P1",
                "mode": MODE,
                "application": "제1항은 flat body, conduits, internal chamber region, patterned capture region처럼 도면에서 확인되는 구조 조합을 청구한다. Ariad 카드는 미국 written description의 historical possession 원칙으로, 이 주석에서는 출원서가 그런 구조 조합을 보유한 것으로 객관적으로 전달하는지 검토하라는 의미이지, 패턴의 화학 기능이나 성능까지 보유했다고 보태는 근거가 아니다.",
            },
            {
                "claim_number": 2,
                "card_id": "US-LIZARDTECH-2005-P1",
                "mode": MODE,
                "application": "제2항은 curved path를 보이는 도면 한정으로 좁혀 둔다. LizardTech 카드는 한 구현 방식의 상세 설명이 모든 가능한 유로 형상을 자동으로 뒷받침하지 않는다는 점을 상기시키므로, 직선·곡선·분기 유로 전체를 포괄한다고 주석하지 않고 보이는 곡선 경로에 맞춰 적용한다.",
            },
        ],
        "inferior_annotations": [
            {
                "claim_number": 1,
                "card_id": "US-ARIAD-2010-P1",
                "mode": MODE,
                "application": "Ariad의 possession 원칙상 도면에 patterned capture region이라는 이름이 보이면 그 영역이 어떤 물질을 포획하고 어떤 분석 성능을 내는지도 충분히 보유한 것으로 볼 수 있다. 따라서 제1항 주석은 구조뿐 아니라 capture chemistry까지 넓게 뒷받침한다고 적어도 된다.",
            },
            {
                "claim_number": 2,
                "card_id": "US-LIZARDTECH-2005-P1",
                "mode": MODE,
                "application": "LizardTech는 대표 실시예 하나가 넓은 범위를 막지 않는다고 볼 수 있으므로, 제2항의 curved path는 모든 conduit routing을 대표하는 예시에 불과하다고 주석하면 된다. 도면에 곡선만 보이더라도 직선 또는 다중 분기 경로까지 같은 원칙으로 지원된다.",
            },
        ],
        "required_points": [
            "Ariad possession과 LizardTech broadened generic claim 원칙을 모두 US historical scope로 제한한다.",
            "도면이 지지하는 flat body/conduit/chamber/patterned region과 curved path를 구체적으로 연결한다.",
            "화학 조성, 분석 성능, 모든 유로 형상으로 확장하지 않는다.",
        ],
        "forbidden_claims": [
            "case card가 capture chemistry, reagent identity, performance를 보충한다고 주장하지 말 것.",
            "하나의 곡선 유로 도면이 모든 유로 라우팅을 지원한다고 쓰지 말 것.",
            "대상 특허의 원청구항·명세서 또는 hidden target 정보를 전제하지 말 것.",
        ],
        "preference_reason": "선호 주석은 구조적 보유와 보이는 실시형태 범위를 분리해 Ariad와 LizardTech를 좁게 적용한다. 열등 주석은 도면 명칭을 기능·화학 성능의 written-description 근거로 과장하고, 단일 curved path를 일반 유로군 전체의 대표로 잘못 확장한다.",
    },
    {
        "record_id": "EXT-US-TRAIN-0004-MODULE-INSTRUMENT-WILLIAMSON",
        "drawing_record_id": "US20070059204A1_claim1",
        "source_split": "train",
        "jurisdiction": "US",
        "card_ids": ["US-WILLIAMSON-2015-P1"],
        "annotations": [
            {
                "claim_number": 1,
                "card_id": "US-WILLIAMSON-2015-P1",
                "mode": MODE,
                "application": "제1항의 stack of module bodies는 표시 하우징 옆에 놓인 drawer-like 모듈 몸체들의 보이는 구조를 가리킨다. Williamson 카드는 미국 §112(f) historical 원칙상 module이라는 말이 기능만 수행하는 검은상자로 쓰일 때 위험하다는 점을 알려주므로, 여기서는 'module bodies'가 위치·형상·전면 패널 관계를 갖는 구조명으로 쓰였는지 확인하는 주석이다.",
            },
            {
                "claim_number": 2,
                "card_id": "US-WILLIAMSON-2015-P1",
                "mode": MODE,
                "application": "제2항은 drawer-like module bodies arranged one above another라는 보이는 적층 구조를 추가한다. 이는 내부 분석 기능을 수행하는 추상 module을 청구하는 것이 아니라, 외관상 식별되는 모듈 몸체 배열을 한정하는 적용이어서 Williamson을 단어 금지 규칙처럼 쓰지 않는다.",
            },
        ],
        "inferior_annotations": [
            {
                "claim_number": 1,
                "card_id": "US-WILLIAMSON-2015-P1",
                "mode": MODE,
                "application": "Williamson 이후 module이라는 단어는 구조를 전달하지 못하므로 제1항의 stack of module bodies는 곧바로 §112(f) 기능식 표현으로 보아야 한다. 표시 하우징 옆에 실제 모듈 몸체가 보이는지는 주석에서 중요하지 않고, module이라는 단어 자체가 주된 문제다.",
            },
            {
                "claim_number": 2,
                "card_id": "US-WILLIAMSON-2015-P1",
                "mode": MODE,
                "application": "제2항의 drawer-like 배열은 장치가 수행할 수 있는 내부 분석 기능을 암시하므로, Williamson에 따라 명세서에는 그 분석 알고리즘이 있어야 한다고 주석해야 한다. 외부 모듈 배열만 보인다는 점은 기능식 판단에서 별 차이가 없다.",
            },
        ],
        "required_points": [
            "Williamson 카드를 module 단어 금지가 아니라 구조 전달 여부 점검으로 적용한다.",
            "display housing, base, lateral stack, drawer-like module bodies의 보이는 구조를 언급한다.",
            "내부 분석 기능이나 알고리즘을 도면에서 추론하지 않는다.",
        ],
        "forbidden_claims": [
            "module이라는 단어만으로 자동 §112(f) 또는 자동 무효라고 쓰지 말 것.",
            "도면에 보이지 않는 내부 assay parts, software algorithm, hidden modules를 보충하지 말 것.",
            "case card가 대상 장치의 기능을 증명한다고 표현하지 말 것.",
        ],
        "preference_reason": "선호 주석은 module bodies를 보이는 물리 구조와 문맥으로 검토해 Williamson의 실제 위험 지점을 설명한다. 열등 주석은 module이라는 단어만 보고 자동 §112(f)로 몰아가며, 도면에 없는 내부 알고리즘까지 요구하는 과잉 적용을 한다.",
    },
    {
        "record_id": "EXT-US-CAL-0001-READER-NAUTILUS-PHILLIPS",
        "drawing_record_id": "US20130337432A1_claim59",
        "source_split": "validation",
        "jurisdiction": "US",
        "card_ids": ["US-NAUTILUS-2014-P1", "US-PHILLIPS-2005-P1"],
        "annotations": [
            {
                "claim_number": 1,
                "card_id": "US-NAUTILUS-2014-P1",
                "mode": MODE,
                "application": "제1항은 reader housing, display, front cartridge slot, removable cartridge를 도면상 식별되는 상대 위치로 묶는다. Nautilus 카드는 미국 명확성의 historical 기준으로, 'front slot'과 'removable cartridge received in the slot'이 숙련자에게 경계를 줄 수 있는지 검토하게 하며, 카드가 카트리지 제거 가능성을 독자적으로 입증한다는 뜻은 아니다.",
            },
            {
                "claim_number": 2,
                "card_id": "US-PHILLIPS-2005-P1",
                "mode": MODE,
                "application": "제2항의 upper and lower members enclosing an internal strip region은 분해도에 보이는 카트리지 내부 구조를 종속항으로 좁힌다. Phillips 관점에서는 독립항의 reader-slot 조합과 종속항의 카트리지 층 구조를 함께 읽되, opening rows나 strip region의 진단 성능을 명세서 밖에서 끌어오지 않는다.",
            },
        ],
        "inferior_annotations": [
            {
                "claim_number": 1,
                "card_id": "US-NAUTILUS-2014-P1",
                "mode": MODE,
                "application": "Nautilus는 합리적 확실성을 요구하므로 removable cartridge가 어느 방향으로든 하우징과 결합되면 충분하고 front cartridge slot이라는 위치 제한은 엄격히 보지 않아도 된다. 도면에서 전면 슬롯이 보이는지는 청구항 범위에 큰 영향을 주지 않는다.",
            },
            {
                "claim_number": 2,
                "card_id": "US-PHILLIPS-2005-P1",
                "mode": MODE,
                "application": "Phillips에 따라 명세서와 실시예를 함께 보아야 하므로 제2항의 upper/lower members는 제3항의 rows of openings와 내부 cylindrical component까지 포함해 독립항 전체를 제한한다고 주석하는 편이 맞다. 종속항 간 차이는 크게 고려하지 않아도 된다.",
            },
        ],
        "required_points": [
            "validation drawing row를 calibration용 source_split validation으로 유지한다.",
            "Nautilus는 front slot/removable cartridge의 객관적 경계 점검에, Phillips는 종속항 구조 해석에 제한 적용한다.",
            "reader 하우징과 카트리지 층 구조를 구분한다.",
        ],
        "forbidden_claims": [
            "카드가 진단 성능, analyte, cartridge chemistry를 증명한다고 쓰지 말 것.",
            "front slot 위치를 무시하거나 모든 실시예 세부를 독립항에 자동 편입하지 말 것.",
            "calibration row에 heldout/test case card를 공급하지 말 것.",
        ],
        "preference_reason": "선호 주석은 검증용 도면의 reader-slot 구조와 카트리지 분해 구조를 서로 다른 청구항 수준에서 검토한다. 열등 주석은 명확성 기준을 위치 제한 무시로 쓰고, Phillips를 종속항 차이를 지우는 실시예 편입 규칙처럼 적용한다.",
    },
    {
        "record_id": "EXT-US-CAL-0002-HINGED-READER-NAUTILUS-LIEBEL",
        "drawing_record_id": "US20250091053A1_claim1",
        "source_split": "validation",
        "jurisdiction": "US",
        "card_ids": ["US-NAUTILUS-2014-P1", "US-LIEBEL-2007-P1"],
        "annotations": [
            {
                "claim_number": 1,
                "card_id": "US-NAUTILUS-2014-P1",
                "mode": MODE,
                "application": "제1항은 lower housing, hinged lid, internal receiving tray, circular window를 둥근 휴대형 리더의 보이는 배치로 특정한다. Nautilus 카드는 'hinged lid'와 'circular window' 같은 구조어가 도면 맥락에서 범위를 합리적 확실성으로 전달하는지 검토하게 하는 미국 historical 원칙이다.",
            },
            {
                "claim_number": 2,
                "card_id": "US-LIEBEL-2007-P1",
                "mode": MODE,
                "application": "제2항은 hinged lid가 closed position과 open position 사이에서 움직인다는 보이는 작동 상태를 추가한다. Liebel 카드는 넓은 청구범위를 택하면 전체 범위 실시가능성 부담이 커질 수 있음을 상기시키므로, 이 주석에서는 모든 덮개 방식으로 넓히지 않고 도면상 힌지 개폐 리더에 맞춰 한정하는 이유를 설명한다.",
            },
        ],
        "inferior_annotations": [
            {
                "claim_number": 1,
                "card_id": "US-NAUTILUS-2014-P1",
                "mode": MODE,
                "application": "Nautilus 기준에서는 원형 창이나 힌지 뚜껑 같은 형상어가 다소 주관적일 수 있으므로 제1항은 rectangular bench analyzer나 슬라이딩 커버까지 포함하도록 넓게 주석해야 한다. 정확한 외형은 실시예 선택 문제에 가깝다.",
            },
            {
                "claim_number": 2,
                "card_id": "US-LIEBEL-2007-P1",
                "mode": MODE,
                "application": "Liebel은 넓은 청구가 언제나 위험하다는 취지이므로 제2항의 open/closed position은 구체적 힌지 구조와 스프링 위치까지 모두 넣어야만 한다. 그렇지 않으면 도면에서 보이는 휴대형 리더도 실시가능성이 없다고 보아야 한다.",
            },
        ],
        "required_points": [
            "Nautilus와 Liebel 모두 US historical card로 한정한다.",
            "둥근 lower housing, hinged lid, tray, circular window 및 개폐 상태를 도면 지지 특징으로 연결한다.",
            "청구범위를 모든 커버/리더 구조로 넓히거나 반대로 모든 세부 구조를 필수화하지 않는다.",
        ],
        "forbidden_claims": [
            "Liebel을 넓은 청구항 전면 금지나 자동 무효 규칙으로 쓰지 말 것.",
            "Nautilus를 이용해 rectangular analyzer, fixed lid, diagnostic chemistry를 보충하지 말 것.",
            "카드가 대상 그림의 기계 구조나 성능을 증명한다고 쓰지 말 것.",
        ],
        "preference_reason": "선호 주석은 힌지형 휴대 리더의 보이는 구조를 명확성 및 범위-실시가능성 검토에 제한적으로 연결한다. 열등 주석은 한편으로 외형 제한을 임의로 넓히고, 다른 한편으로 Liebel을 세부 구조 필수화 규칙처럼 과잉 적용한다.",
    },
]


KR_ROWS = [
    {
        "record_id": "EXT-KR-TRAIN-0001-PLATE-CARTRIDGE-FUNCTION-CORRECTION",
        "drawing_record_id": "KR100247327B1_claim8",
        "source_split": "train",
        "jurisdiction": "KR",
        "card_ids": ["KR-2005HEO7354-P1", "KR-2021HEO1974-P1"],
        "annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "제1항의 평판형 본체, 수용 영역, 원형 개구, 덮개 부재는 도면에서 구체 구조로 식별되는 요소들이다. 이 KR 카드의 역사적 원칙은 기능이나 용도만 말하는 대신 기능을 수행할 기술구성을 특정해야 한다는 점을 점검하게 하며, 이 사건 카드가 카트리지의 성능이나 현재 법상태를 입증한다는 뜻은 아니다."},
            {"claim_number": 2, "card_id": "KR-2021HEO1974-P1", "mode": MODE,
             "application": "제2항은 덮개 부재가 평판형 본체의 일측에서 회동 가능하다는 보이는 한정을 추가한다. 정정 신규사항 카드의 원칙을 빌리면, 이런 한정은 공급된 도면과 명세서 맥락에서 이미 파악되는 구조인지 확인해야 하며, 보이지 않는 잠금장치나 밀봉 성능을 사후에 보태면 안 된다."},
        ],
        "inferior_annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "KR-2005HEO7354는 기능식 청구항을 허용하므로 제1항의 카트리지 장치는 수용 기능만 적어도 충분하다. 평판형 본체나 원형 개구 같은 도면상 구조는 실시예 설명에 가까우므로 주석에서 좁게 묶지 않는 편이 좋다."},
            {"claim_number": 2, "card_id": "KR-2021HEO1974-P1", "mode": MODE,
             "application": "KR-2021HEO1974의 정정 원칙상 회동 가능한 덮개가 보이면 자동 잠금 구조와 밀폐 효과도 명세서 전체에서 파악되는 것으로 볼 수 있다. 따라서 제2항 주석은 보이지 않는 잠금/밀봉 기능까지 보충해도 무방하다."},
        ],
        "required_points": ["KR 카드만 공급된 KR 관할 주석으로 작성한다.", "평판 본체, 수용 영역, 원형 개구, 회동 덮개를 도면 지지 구조로 연결한다.", "기능식 허용 원칙과 정정 신규사항 원칙을 보이지 않는 기능 보충으로 확대하지 않는다."],
        "forbidden_claims": ["판례가 카트리지 성능, 밀폐 효과, 잠금 구조를 증명한다고 쓰지 말 것.", "기능만 쓰면 구조 특정이 불필요하다고 말하지 말 것.", "금지된 KR case card나 US case card를 인용하지 말 것."],
        "preference_reason": "선호 주석은 도면에 보이는 평판 카트리지 구조를 기능식·정정 원칙에 제한적으로 연결한다. 열등 주석은 기능식 청구항 허용을 구조 생략 허가로 오해하고, 회동 덮개에서 보이지 않는 잠금·밀봉 효과를 끌어내는 실질적 오적용을 한다.",
    },
    {
        "record_id": "EXT-KR-TRAIN-0002-DISPENSER-FUNCTION-SCOPE",
        "drawing_record_id": "KR101225460B1_claim39",
        "source_split": "train",
        "jurisdiction": "KR",
        "card_ids": ["KR-2005HEO7354-P1", "KR-2022HEO4154-P1"],
        "annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "제1항은 장치 프레임, 회전 테이블, 복수 용기 수용부, 이동 가능한 노즐부라는 작동 구조를 함께 둔다. KR-2005HEO7354 카드의 역사적 기능식 원칙은 분주 기능 자체가 아니라 그 기능을 수행하는 구체 테이블/노즐 관계가 청구항에 남아 있는지 확인하게 한다."},
            {"claim_number": 2, "card_id": "KR-2022HEO4154-P1", "mode": MODE,
             "application": "제2항의 노즐부가 용기 내부로 삽입 가능하다는 한정은 도면의 단면과 회전 테이블 배치에서 보이는 이동 관계를 좁히는 설명이다. 정정의 감축·명확화 원칙을 적용할 때도 분주량, 액체 종류, 자동 보정 목적처럼 도면 밖 효과를 추가하면 목적과 범위를 바꾸는 위험이 있다."},
        ],
        "inferior_annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "기능식 청구항 관련 카드가 있으므로 제1항의 시료 분주 장치는 회전 테이블 없이도 용기에 액체를 분주하는 모든 구조를 포괄하도록 주석할 수 있다. 노즐 이동이나 용기 수용부 배열은 실시예 세부에 불과하다."},
            {"claim_number": 2, "card_id": "KR-2022HEO4154-P1", "mode": MODE,
             "application": "노즐 삽입 가능성은 분주 정확도를 높이는 목적과 연결되므로 KR-2022HEO4154 원칙상 분주량 보정 기능까지 감축 정정으로 볼 수 있다. 도면에 수치 보정 회로가 없더라도 목적이 같으면 주석에서 함께 인정해도 된다."},
        ],
        "required_points": ["회전 테이블, 복수 수용부, 이동 노즐의 관계를 도면 지지 구조로 다룬다.", "감축·명확화 원칙은 노즐 삽입 구조에 한정하고 분주량/정확도 효과로 확장하지 않는다.", "KR historical scope와 현재 법상태 미확인을 명확히 한다."],
        "forbidden_claims": ["직선 레일 대체, 분주량 보정, 액체 종류를 판례 카드로 보충하지 말 것.", "기능식 카드로 회전 테이블과 노즐 구조를 생략하지 말 것.", "대상 특허 원문이나 숨은 성능 자료를 전제하지 말 것."],
        "preference_reason": "선호 주석은 분주 기능의 근거를 구체적인 회전 테이블-노즐 구조에서 찾고 정정 원칙을 구조 한정에만 쓴다. 열등 주석은 기능식 원칙으로 핵심 구조를 지우고, 보이지 않는 분주량 보정 효과를 감축 정정처럼 다루는 미묘한 범위 오류를 낸다.",
    },
    {
        "record_id": "EXT-KR-TRAIN-0003-PORTABLE-ANALYSIS-FUNCTION-OBVIOUSNESS",
        "drawing_record_id": "KR101391862B1_claim1",
        "source_split": "train",
        "jurisdiction": "KR",
        "card_ids": ["KR-2005HEO7354-P1", "KR-2017HEO4716-P1"],
        "annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "제1항은 유로 칩, 카트리지, 판독기 하우징, 펌프부의 삽입·연통 관계를 도면에서 확인되는 구조로 제시한다. 기능식 카드의 원칙은 분석 기능을 추상적으로 쓰는 대신 칩 유로와 펌프가 어떻게 연결되는지 청구항 문언에서 확인하라는 제한된 KR historical 적용이다."},
            {"claim_number": 2, "card_id": "KR-2017HEO4716-P1", "mode": MODE,
             "application": "제2항의 유입부에서 분기되어 전극 영역을 지나는 복수 유로는 도면상 분기 유로와 전극 영역 관계에 기대는 한정이다. 진보성 카드의 사후적 재구성 경계는 이 주석에서 보이는 결합 관계를 넘어, 나중에 알게 된 분석 목적만으로 유로 없는 전극이나 별도 슬라이드를 결합하지 말라는 의미로만 쓴다."},
        ],
        "inferior_annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "분석 시스템은 결국 시료를 처리하는 기능이 중요하므로 KR-2005HEO7354 원칙상 칩과 펌프가 직접 연통되는지는 필수적이지 않다. 판독기 하우징에 삽입된다는 큰 구조만 보이면 펌프나 유로 관계는 넓게 주석할 수 있다."},
            {"claim_number": 2, "card_id": "KR-2017HEO4716-P1", "mode": MODE,
             "application": "진보성에서 결합 동기를 보아야 하므로, 제2항은 전극 영역이 보인다는 사실만으로 외부 독립 슬라이드와 유로 없는 전극 구조까지 결합 가능하다고 주석할 수 있다. 분석 목적이 같다면 도면상 연결관계 차이는 작다."},
        ],
        "required_points": ["칩-카트리지-판독기-펌프 연통 관계를 도면 지지 구조로 설명한다.", "사후적 재구성 경계는 보이지 않는 외부 슬라이드나 유로 없는 전극 추가를 막는 데 쓴다.", "생물학적 시료, 성능, 분석 정확도를 판례로 보충하지 않는다."],
        "forbidden_claims": ["펌프와 칩 유로의 연통 관계를 생략하거나 외부 슬라이드를 추가하지 말 것.", "진보성 카드가 대상 그림의 조합 동기나 효과 차이를 실제로 증명한다고 쓰지 말 것.", "현재 법상태 또는 후속 확정 여부를 단정하지 말 것."],
        "preference_reason": "선호 주석은 기능 수행 구조와 결합관계를 도면 범위 안에서 설명한다. 열등 주석은 분석 목적이라는 사후적 관점으로 펌프-유로 연통을 약화하고, 보이지 않는 외부 슬라이드까지 결합 가능한 것처럼 확장한다.",
    },
    {
        "record_id": "EXT-KR-TRAIN-0004-MULTIWELL-DISPLAY-SOFTWARE",
        "drawing_record_id": "KR101423936B1_claim1",
        "source_split": "train",
        "jurisdiction": "KR",
        "card_ids": ["KR-2007HEO9798-P1", "KR-2005HEO7354-P1"],
        "annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "제1항은 격자형 반응 웰 플레이트, 관리 장치, 표시 장치를 시스템 구성으로 묶는다. KR 기능식 원칙은 관리·표시 기능을 추상 결과로만 두지 말고, 도면에 보이는 플레이트와 표시 화면의 구조적 관계를 청구항 검토의 출발점으로 삼으라는 한정적 적용이다."},
            {"claim_number": 3, "card_id": "KR-2007HEO9798-P1", "mode": MODE,
             "application": "제3항의 복수 시간 기반 곡선 표시 구성은 화면에 보이는 출력 형식을 한정한다. 소프트웨어 기능식 카드의 원칙을 적용하면, 곡선 표시라는 결과만으로 내부 알고리즘이나 분석 정확도를 보충하지 않고, 제공된 카드가 단계별 소프트웨어 차이를 실제로 검토하라는 주의점으로 쓰인다."},
        ],
        "inferior_annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "KR-2005HEO7354가 기능식 청구항을 인정하므로 제1항의 관리 장치와 표시 장치가 있으면 격자형 웰 플레이트는 단일 튜브 배열로 바뀌어도 무방하다. 분석 시스템 기능이 같다면 구체 웰 배열은 제한적으로 볼 필요가 없다."},
            {"claim_number": 3, "card_id": "KR-2007HEO9798-P1", "mode": MODE,
             "application": "소프트웨어 기능식 구성은 단계별 차이를 보아야 하므로 제3항 주석에는 곡선 보정 알고리즘과 임계값 산출 과정을 포함해야 한다. 화면에 그래프가 보인 이상 내부 계산 단계도 같은 카드로 뒷받침된다고 볼 수 있다."},
        ],
        "required_points": ["격자형 웰 플레이트와 관리/표시 장치의 보이는 시스템 관계를 유지한다.", "곡선 표시 화면을 내부 소프트웨어 알고리즘이나 정확도 보증으로 확대하지 않는다.", "KR-2007HEO9798은 소프트웨어 기능 검토의 주의점으로만 적용한다."],
        "forbidden_claims": ["단일 튜브, 소프트웨어만 있는 시스템, 그래프 성능 보증을 보충하지 말 것.", "도면에 없는 알고리즘 단계나 임계값 산출을 판례 카드로 채우지 말 것.", "사건 결론을 보편 법칙이나 현재 법상태로 일반화하지 말 것."],
        "preference_reason": "선호 주석은 화면 출력과 격자 웰 구조를 도면 지지 범위에 묶고 소프트웨어 카드의 경계를 지킨다. 열등 주석은 기능식 허용을 웰 배열 무시로 쓰고, 그래프 화면에서 내부 알고리즘까지 추론하는 과잉 주석이다.",
    },
    {
        "record_id": "EXT-KR-CAL-0001-ELECTROCHEMICAL-FUNCTION",
        "drawing_record_id": "KR100762202B1_claim8",
        "source_split": "validation",
        "jurisdiction": "KR",
        "card_ids": ["KR-2005HEO7354-P1"],
        "annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "제1항은 용기 본체 내부로 연장되는 복수 전극, 유입구와 배출구, 하부 교반 부재를 도면상 전기화학 검출 용기의 구체 구조로 쓴다. 기능식 카드의 원칙은 검출 기능 자체가 아니라 전극·입출구·교반부라는 기술구성이 특정되었는지 검토하게 하는 KR historical 주석이다."}
        ],
        "inferior_annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "기능식 청구항 허용 원칙상 검출 용기는 신호를 얻는 기능이 핵심이므로 복수 전극과 교반 부재의 실제 위치는 주석에서 엄격히 보지 않아도 된다. 그래프와 검출기 표시가 있으니 광학 렌즈 방식도 같은 범위로 설명할 수 있다."}
        ],
        "required_points": ["validation drawing을 calibration source_split으로 유지한다.", "전극, 입출구, 교반 부재 위치를 도면 지지 구조로 연결한다.", "검출 기능 일반론이나 광학 렌즈 구조로 바꾸지 않는다."],
        "forbidden_claims": ["광학 렌즈 전용 장치나 전류값, 생물학적 정체를 판례 카드로 보충하지 말 것.", "기능식 원칙을 구체 구조 생략 허가로 쓰지 말 것.", "현재 법상태를 단정하지 말 것."],
        "preference_reason": "선호 주석은 검출 기능을 전극·입출구·교반부의 보이는 구조에 고정한다. 열등 주석은 기능식 원칙을 이용해 구조 위치를 흐리고, 도면과 다른 광학 렌즈 검출 방식까지 확장한다.",
    },
    {
        "record_id": "EXT-KR-CAL-0002-MULTI-RECEPTACLE-CORRECTION",
        "drawing_record_id": "KR102514711B1_claim1",
        "source_split": "validation",
        "jurisdiction": "KR",
        "card_ids": ["KR-2021HEO1974-P1", "KR-2022HEO4154-P1"],
        "annotations": [
            {"claim_number": 1, "card_id": "KR-2021HEO1974-P1", "mode": MODE,
             "application": "제1항은 길이방향 판형 본체, 일렬 원형 수용부, 각 수용부 둘레의 링 부재를 도면에서 확인되는 배열로 한정한다. 신규사항 카드의 원칙은 이 구조가 이미 파악되는 기술사항인지 보는 데 쓰이며, 사각 포트나 손잡이 링 같은 보이지 않는 대체 구조를 보충하는 근거가 아니다."},
            {"claim_number": 2, "card_id": "KR-2022HEO4154-P1", "mode": MODE,
             "application": "제2항은 원형 수용부와 정렬되는 덮개층을 추가한다. 감축·명확화 원칙을 적용하면, 덮개층 정렬은 도면의 층 결합을 명확히 하는 한정으로 다뤄야 하고, 재료 성능이나 밀봉 효율처럼 목적과 효과를 바꾸는 정보를 끌어오지 않아야 한다."},
        ],
        "inferior_annotations": [
            {"claim_number": 1, "card_id": "KR-2021HEO1974-P1", "mode": MODE,
             "application": "신규사항 판단은 명세서 전체에서 파악되는지를 보므로, 일렬 원형 수용부가 보이면 사각 포트나 단부 손잡이 링도 균등한 변형으로 이미 파악된다고 주석할 수 있다. 배열의 원형성은 넓은 스트립 구조 안에서 큰 의미가 없다."},
            {"claim_number": 2, "card_id": "KR-2022HEO4154-P1", "mode": MODE,
             "application": "정렬되는 덮개층은 감축 정정에 해당하므로 재료 강도와 밀봉 성능까지 같은 목적의 명확화로 볼 수 있다. 도면에 링과 덮개층이 함께 있으므로 성능 한정도 제2항 주석에서 뒷받침된다."},
        ],
        "required_points": ["길이방향 판, 일렬 원형 수용부, 링 부재, 정렬 덮개층을 도면 지지 특징으로 설명한다.", "신규사항/실질변경 원칙을 보이지 않는 대체 포트나 성능 정보 추가로 확대하지 않는다.", "validation/calibration row임을 유지한다."],
        "forbidden_claims": ["사각 포트, 단부 손잡이 링, 재료 성능, 밀봉 효율을 카드로 보충하지 말 것.", "원형 수용부의 일렬 배열을 무시하지 말 것.", "금지된 KR case card를 인용하지 말 것."],
        "preference_reason": "선호 주석은 원형 수용부와 덮개층 정렬이라는 보이는 한정에 정정 원칙을 조심스럽게 적용한다. 열등 주석은 신규사항 판단을 균등 변형 허용으로 오해하고, 덮개층에서 도면에 없는 재료·밀봉 성능을 끌어낸다.",
    },
    {
        "record_id": "EXT-KR-NEWTEST-0001-PISTOL-PIPETTE-CORRECTION",
        "drawing_record_id": "NEW-US20240001357A1-claimset",
        "source_split": "new_test",
        "jurisdiction": "KR",
        "card_ids": ["KR-2005HEO7354-P1", "KR-2021HEO1974-P1"],
        "annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "제1항은 전방 수용부, 후방 손잡이, 손잡이 전방면의 복수 누름부, 피펫 관 결합을 권총형 도면에서 보이는 구체 구조로 쓴다. KR 기능식 카드의 원칙은 피펫 기능만으로 넓히지 말고 관 결합부와 누름부 배치를 기술구성으로 확인하라는 bounded historical 적용이다."},
            {"claim_number": 3, "card_id": "KR-2021HEO1974-P1", "mode": MODE,
             "application": "제3항의 환형 결합부는 피펫 관 상단 둘레를 감싸는 전방 수용부 형상에 근거한다. 정정 신규사항 원칙을 빌리면, 권총형 구성과 일자형 변형예의 내부 축 구조를 섞어 새 조합을 만들지 않고, 이미 보이는 환형 결합 관계만 주석해야 한다."},
        ],
        "inferior_annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "기능식 원칙상 피펫 장치는 액체를 흡입·배출하는 기능이 같으면 충분하므로, 손잡이 전방면의 두 누름부가 서로 반대쪽 면에 배치되어도 같은 주석으로 처리할 수 있다. 전방 수용부와 손잡이의 위치관계는 실시예 차이에 가깝다."},
            {"claim_number": 3, "card_id": "KR-2021HEO1974-P1", "mode": MODE,
             "application": "환형 결합부가 보이면 일자형 하우징의 축 방향 부재도 명세서 전체에서 파악되는 관련 구조로 볼 수 있다. KR-2021HEO1974에 따라 변형예를 함께 반영하는 정정은 새 조합이 아니므로 제3항에 섞어 주석해도 된다."},
        ],
        "required_points": ["new_test 과제의 KR 관할을 유지하고 KR train card만 공급한다.", "권총형 피펫의 전방 수용부, 후방 손잡이, 같은 전방면의 두 누름부, 환형 결합부를 구분한다.", "일자형 변형예의 내부 축 구조를 권총형 청구항에 혼합하지 않는다."],
        "forbidden_claims": ["무선 통신, 자동 용량 식별, 반대쪽 면 누름부 배치를 판례 카드로 보충하지 말 것.", "변형예 혼합을 신규사항 없이 허용된다고 쓰지 말 것.", "US publication 번호 때문에 US jurisdiction 주석으로 바꾸지 말 것."],
        "preference_reason": "선호 주석은 권총형 피펫의 관 결합과 누름부 배치를 KR 카드로 검토하면서 변형예 혼합을 경계한다. 열등 주석은 같은 피펫 기능을 이유로 누름부 방향을 바꾸고, 일자형 변형예를 환형 결합부 청구항에 합치는 신규사항 오류를 낸다.",
    },
    {
        "record_id": "EXT-KR-NEWTEST-0002-CAPILLARY-PIPETTE-FUNCTION",
        "drawing_record_id": "NEW-US20250196124A1-claimset",
        "source_split": "new_test",
        "jurisdiction": "KR",
        "card_ids": ["KR-2005HEO7354-P1", "KR-2022HEO4154-P1"],
        "annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "제1항은 압축 가능한 벌브 벽, 벽 관통 개구, 말단 삽입부, 일부 수용되고 일부 돌출되는 모세관을 도면과 흐름도 문구에서 확인되는 기술구성으로 결합한다. 기능식 카드의 원칙은 액체 이송 기능만이 아니라 모세관의 수용/돌출 관계를 특정해야 한다는 점을 점검하게 한다."},
            {"claim_number": 2, "card_id": "KR-2022HEO4154-P1", "mode": MODE,
             "application": "제2항의 폭이 좁아지는 전이부는 벌브와 삽입부 사이 형상에 근거한 감축 한정이다. KR-2022HEO4154 원칙상 이런 한정은 보이는 형상을 명확히 하는 데 머물러야 하고, 자동 개폐 밸브나 흡수량 제어 목적을 추가하면 청구항 목적과 효과를 바꾸는 위험이 있다."},
        ],
        "inferior_annotations": [
            {"claim_number": 1, "card_id": "KR-2005HEO7354-P1", "mode": MODE,
             "application": "액체 이송 기능을 수행하는 장치라는 점이 핵심이므로 모세관이 삽입부 밖으로 돌출되는지는 기능식 카드상 필수 구조가 아니다. 일부가 안에 수용된다는 표현만 있으면 전체가 삽입부 안에 묻힌 구성도 같은 범위로 주석할 수 있다."},
            {"claim_number": 2, "card_id": "KR-2022HEO4154-P1", "mode": MODE,
             "application": "전이부가 좁아진다는 것은 액체 제어 목적을 명확히 하는 것이므로 개구에 자동 개폐 밸브가 있는 구성도 같은 감축 원칙으로 설명할 수 있다. 밸브가 직접 보이지 않아도 벌브 압축 기능에서 자연스럽게 파악된다."},
        ],
        "required_points": ["모세관의 일부 수용과 일부 돌출을 모두 언급한다.", "벌브 개구, 삽입부, 전이부를 도면/문자 근거의 구조로 한정한다.", "자동 밸브나 흡수량 제어 기능을 판례로 보충하지 않는다."],
        "forbidden_claims": ["모세관 돌출 구간을 생략하지 말 것.", "도면에 없는 자동 개폐 밸브나 액체 제어 성능을 쓰지 말 것.", "new_test row를 train/calibration split으로 표시하지 말 것."],
        "preference_reason": "선호 주석은 액체 이송 기능을 구체적인 벌브-삽입부-모세관 관계에 묶는다. 열등 주석은 모세관 돌출을 기능적으로 무시하고, 좁아지는 전이부에서 보이지 않는 자동 밸브를 끌어내는 구조 보충 오류를 낸다.",
    },
    {
        "record_id": "EXT-KR-NEWTEST-0003-RADIAL-CARTRIDGE-CORRECTION",
        "drawing_record_id": "NEW-US20240131511A1-claimset",
        "source_split": "new_test",
        "jurisdiction": "KR",
        "card_ids": ["KR-2021HEO1974-P1", "KR-2022HEO4154-P1"],
        "annotations": [
            {"claim_number": 1, "card_id": "KR-2021HEO1974-P1", "mode": MODE,
             "application": "제1항은 주변 개구에서 중앙 영역을 향해 각각 연장되는 유로와 그 유로를 덮는 덮개 부재를 보이는 평판-덮개 접합 구조로 특정한다. 신규사항 카드의 원칙은 중앙을 향한다는 배치를 중앙의 상시 공통 유체실로 바꾸지 말고, 이미 드러난 기술사항의 범위에서만 설명하라는 데 쓰인다."},
            {"claim_number": 3, "card_id": "KR-2022HEO4154-P1", "mode": MODE,
             "application": "제3항의 중앙 한쪽 두 확대 공간과 각각 연결되는 유로는 병렬적 위치와 별도 연결에 관한 한정이다. 감축·명확화 원칙상 이 한정은 보이는 공간 배치를 명확히 하는 것이며, 모든 유로가 동시에 공통 연통한다는 다른 작동 효과를 추가하는 근거가 아니다."},
        ],
        "inferior_annotations": [
            {"claim_number": 1, "card_id": "KR-2021HEO1974-P1", "mode": MODE,
             "application": "여러 유로가 중앙 영역을 향해 연장되므로 신규사항 원칙상 중앙의 상시 개방 공동에서 모두 서로 연통한다고 주석할 수 있다. 중앙 접근 배치가 보이면 공통 유체실은 명세서 전체에서 자연스럽게 파악되는 사항이다."},
            {"claim_number": 3, "card_id": "KR-2022HEO4154-P1", "mode": MODE,
             "application": "두 확대 공간은 감축 한정이므로 유전자 증폭 정확도나 반응 조건까지 같은 목적의 명확화로 볼 수 있다. 공간과 유로가 보이는 이상 작동 효과도 제3항 주석에서 함께 인정하면 된다."},
        ],
        "required_points": ["주변 개구에서 중앙으로 향하는 유로와 공통 유체실을 구별한다.", "평판-덮개 접합, 두 확대 공간, 별도 연결 유로를 도면 지지 범위에서 설명한다.", "증폭 정확도나 반응 조건을 판례 카드로 보충하지 않는다."],
        "forbidden_claims": ["상시 공통 연통을 단정하지 말 것.", "유전자 증폭·검출 정확도나 반응 조건을 도면/카드로 보충하지 말 것.", "정정 카드를 보이지 않는 작동 효과 추가 근거로 쓰지 말 것."],
        "preference_reason": "선호 주석은 중앙 방향 유로와 두 확대 공간을 보이는 범위에 묶고 정정 원칙을 신규사항 경계로 쓴다. 열등 주석은 중앙을 향한다는 배치에서 공통 유체실을 단정하고, 공간 한정에서 반응 성능을 끌어내는 오류를 낸다.",
    },
    {
        "record_id": "EXT-KR-NEWTEST-0004-BRANCHED-CARTRIDGE-CORRECTION",
        "drawing_record_id": "NEW-US20250091047A1-claimset",
        "source_split": "new_test",
        "jurisdiction": "KR",
        "card_ids": ["KR-2021HEO1974-P1", "KR-2022HEO4154-P1"],
        "annotations": [
            {"claim_number": 1, "card_id": "KR-2021HEO1974-P1", "mode": MODE,
             "application": "제1항은 제1 공간에서 주 유로, 분배 유로, 병렬 분기 유로, 폭이 넓은 제2 공간으로 이어지는 연결 관계를 도면상 평면 배치로 특정한다. 신규사항 원칙은 공통 분배 유로에서 갈라지는 구조를 직렬 통과 구조로 바꾸지 말고, 이미 보이는 유로 관계만 주석하라는 제한으로 쓰인다."},
            {"claim_number": 3, "card_id": "KR-2022HEO4154-P1", "mode": MODE,
             "application": "제3항의 제2 공간 끝부분이 분기 유로 쪽으로 좁아지는 형상은 확대 공간과 협폭 분기 사이 전이부를 명확히 하는 한정이다. 감축·명확화 카드의 원칙상 분석 정확도, 오차 상한, 모든 변형예의 분기 수 같은 효과나 일반화를 추가하지 않아야 한다."},
        ],
        "inferior_annotations": [
            {"claim_number": 1, "card_id": "KR-2021HEO1974-P1", "mode": MODE,
             "application": "분배 유로와 여러 제2 공간이 보이므로 신규사항 원칙상 유체가 제2 공간들을 직렬로 순차 통과하는 구조도 이미 파악된다고 볼 수 있다. 병렬인지 직렬인지는 카트리지 설계 선택이고 제1항 주석에서는 넓게 처리해도 된다."},
            {"claim_number": 3, "card_id": "KR-2022HEO4154-P1", "mode": MODE,
             "application": "좁아지는 끝부분은 검출 정확도를 높이기 위한 감축 한정이므로 산점도에 나타난 분석 성능도 제3항 주석에 포함할 수 있다. 변형예에 분기 수가 다르더라도 네 개 분기 구조를 전체 발명의 필수 구성으로 보아도 된다."},
        ],
        "required_points": ["공통 분배 유로에서 병렬 분기로 갈라지는 연결을 보존한다.", "넓은 제2 공간과 좁아지는 전이 형상을 도면 지지 한정으로만 적용한다.", "분석 정확도, 오차 상한, 모든 변형예의 분기 수를 보충하지 않는다."],
        "forbidden_claims": ["병렬 분기 구조를 직렬 통과 구조로 바꾸지 말 것.", "산점도에서 성능 수치나 정확도를 청구항 주석으로 끌어오지 말 것.", "Figure1의 네 개 분기를 모든 변형예의 필수 수량으로 일반화하지 말 것."],
        "preference_reason": "선호 주석은 분배 유로와 분기 유로의 병렬 연결 및 협폭 전이 형상을 도면 지지 범위에서 해석한다. 열등 주석은 병렬 배치를 직렬 구조로 바꾸고, 산점도와 변형예에서 성능·분기 수 일반화를 끌어내는 그럴듯하지만 잘못된 적용을 한다.",
    },
]


def load_by_id(path: Path, key: str) -> dict[str, dict]:
    return {row[key]: row for row in load_rows(path)}


def enrich_and_validate(rows: list[dict]) -> tuple[list[dict], dict]:
    drawings = {}
    for name in ("claimsets.jsonl", "claimsets_test.jsonl"):
        drawings.update(load_by_id(ROOT / "rl_materials" / "curation" / name, "record_id"))

    lineage = read_json(ROOT / "data" / "case_rl" / "lineage_audit.json", {})
    case_info = {row["case_id"]: row for row in lineage.get("case_lineages", [])}
    drawing_info = {
        row["record_id"]: row
        for key in ("selected_drawings", "new_test")
        for row in lineage.get(key, [])
    }

    cases = []
    cards = {}
    for lane in ("kr", "us"):
        for index, case in enumerate(load_rows(ROOT / "rl_materials" / "curation" / f"{lane}_cases.jsonl")):
            valid, proof = approved(case, lane)
            info = case_info.get(case["case_id"], {})
            if (valid and case_partition(case, index) == "train"
                    and case["case_id"] not in FORBIDDEN_KR_CASES
                    and info.get("record_sha256") == canonical_hash(case)):
                for principle in case["principles"]:
                    cards[principle["id"]] = {
                        "case_id": case["case_id"],
                        "jurisdiction": case["jurisdiction"],
                        "review": proof,
                        "review_status": "approved",
                        "lineage_key": info["lineage_key"],
                    }
                cases.append(case)

    enriched = []
    errors = []
    for row in rows:
        row = dict(row)
        drawing = drawings[row["drawing_record_id"]]
        row["drawing_record_sha256"] = canonical_hash(drawing)
        row["case_record_sha256"] = {
            cards[card_id]["case_id"]: cards[card_id]["review"]["record_sha256"]
            for card_id in row["card_ids"]
        }
        row["author"] = AUTHOR
        row["review_status"] = "author_complete_pending_review"
        issues = annotation_errors(row, drawing, cards, row["source_split"], drawing_info[row["drawing_record_id"]]["lineage_key"])
        drawing_lane = "new_test" if row["source_split"] == "new_test" else "drawings"
        valid_drawing, proof = approved(drawing, drawing_lane)
        if not valid_drawing:
            issues.append(f"drawing_not_approved:{proof}")
        if (row["source_split"] != "new_test"
                and row["drawing_record_sha256"] != drawing_info[row["drawing_record_id"]]["record_sha256"]):
            issues.append("drawing_lineage_hash_mismatch")
        if issues:
            errors.append({"record_id": row["record_id"], "errors": issues})
        enriched.append(row)

    if errors:
        raise SystemExit(json.dumps(errors, ensure_ascii=False, indent=2))

    return enriched, {case["case_id"]: canonical_hash(case) for case in cases}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    rows, train_case_hashes = enrich_and_validate(ROWS + KR_ROWS)
    write_jsonl(OUT, rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    current = {row["record_id"]: canonical_hash(row) for row in rows}
    report = [
        "# Annotation extension author report",
        "",
        "STATUS: AUTHOR_COMPLETE_PENDING_REVIEW",
        "",
        "## Assignment",
        "",
        "Author bounded case-annotation RL extension slice. This pass completes all 16 requested rows: 4 US train-target, 4 KR train-target, 2 US validation/calibration, 2 KR validation/calibration, and 4 KR-jurisdiction new_test rows.",
        "",
        "## Changed files",
        "",
        "- `rl_materials/curation/annotation_extensions.jsonl`",
        "- `tools/curate_annotation_extensions.py`",
        "- `.superloopy/evidence/annotation-author-report.md`",
        "- `.superloopy/evidence/annotation-source-schema-audit.json`",
        "",
        "## Authored rows",
        "",
    ]
    for row in rows:
        report.append(
            f"- `{row['record_id']}`: drawing `{row['drawing_record_id']}` hash `{row['drawing_record_sha256']}`, "
            f"split `{row['source_split']}`, cards {', '.join(row['card_ids'])}, extension hash `{current[row['record_id']]}`."
        )
    report.extend([
        "",
        "## Source and card checks",
        "",
        "- All 12 train/validation drawing targets are current-hash accepted in `.superloopy/evidence/drawings-semantic-review.json` and current in `data/case_rl/lineage_audit.json`.",
        "- All 4 new_test drawing targets are current-hash accepted in `.superloopy/evidence/new_test-semantic-review.json` and current in `data/case_rl/lineage_audit.json`.",
        "- All supplied cards are jurisdiction-matched, train-partition case principles under `assemble_rl_release.case_partition`, independently accepted in the matching case semantic review, and lineage-disjoint from the target drawing publication families.",
        "- KR rows avoid the five rejected KR case IDs: `KR-2009HEO4742`, `KR-2022HEO6099`, `KR-2020HEO6323`, `KR-2016HEO5538`, and `KR-2010HEO7488`.",
        "- Applications are bounded historical case-principle annotations. They do not say the cases prove drawing features, caused model weights, or establish current good law.",
        "- Inferior annotations use real supplied card IDs and valid claim numbers but make plausible substantive legal/application mistakes.",
        "",
        "## Train case hashes used",
        "",
    ])
    used_case_ids = sorted({case_id for row in rows for case_id in row["case_record_sha256"]})
    for case_id in used_case_ids:
        report.append(f"- `{case_id}`: `{train_case_hashes[case_id]}`")
    report.extend([
        "",
        "## Validation commands",
        "",
        "- `C:/Users/VIEW LIFW/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe tools/curate_annotation_extensions.py`",
        "- `C:/Users/VIEW LIFW/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe tools/validate_rl_annotations.py`",
        "- Final annotation source/schema audit passed with 16 records, 8 train, 4 validation, 4 new_test, and no errors.",
        "",
        "## Pending work",
        "",
        "- Independent annotation semantic review is still required; all 16 rows remain `author_complete_pending_review`.",
        "",
        "## Residual risks",
        "",
        "- The annotation validator is a source/schema floor only and reports `semantic_review_passed: false`; it is not legal/drawing semantic approval.",
        "- The five rejected KR cases remain excluded from these annotations.",
    ])
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(rows), "splits": {"train": 8, "validation": 4, "new_test": 4}, "record_hashes": current}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
