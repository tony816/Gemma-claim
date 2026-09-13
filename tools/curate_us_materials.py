"""Curate US case-grounded RL advisory material from local patent PDFs."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data" / "case_rl" / "inventory.jsonl"
OUT_JSONL = ROOT / "rl_materials" / "curation" / "us_cases.jsonl"
EVIDENCE_ROOT = ROOT / "data" / "case_rl" / "curation_us"
REPORT = ROOT / ".superloopy" / "evidence" / "us-author-report.md"
PRE_ARIAD_FIX_HASH = "acce4f9a7d4f62f4a1f0a7ff326b0ec3c4a518410230212c37e19419934c7dc2"
POST_ARIAD_FIX_HASH = "7d46b2a44795f45ce288e3d66806728dce2f8675a4b32a24b83bda66741c33b3"
PRESERVED_19_HASHES = {
    "US-MARKMAN-1996": "da56671ada4098767c3dc1a7983fb3abd6bfc61a273b1be962db78c7048b92e3",
    "US-PHILLIPS-2005": "d909d717064789095bad520a1ed55f9f505b01c80f998413349481afa3da0c56",
    "US-TEVA-SANDOZ-2015": "eae06d8399acb4ae965c03bd6419c887eba2cff34c1d281f15aa9899e680527e",
    "US-NAUTILUS-2014": "146fccdbefa9d73a6aa7f20a1f9202a492f5ce86ebd78ad53cd7c4c80700481a",
    "US-WILLIAMSON-2015": "a72bc70c81d115621d20d3bf29810550739f516712f337a78be0ec667301ec41",
    "US-BIOMEDINO-2007": "4fdc31f850a8ec614fcc8ba8dadc850564d2f686a5d89e9c1a1d8bde0ba2aba0",
    "US-ARISTOCRAT-2008": "629d1bdb50b6387e3776a6b853e2fcaae84f26abec2f6e2ce0e5bc18b03c169f",
    "US-NOAH-2012": "e7abd7d46a70299bfe2ec459538153683802c7f1a3203d5de05a66928124499d",
    "US-INTERVAL-2014": "a92be33417aad16950604c3f2c1ef30b7f331f8304a58b1a399fc4e158d8029c",
    "US-DATAMIZE-2005": "a842e09bc4e85f99675cb6b2de4cc0e0e1a7ab9c3c5c6692396095e8c834312c",
    "US-HALLIBURTON-2008": "ad31190e270fd4c8d22823a2d00871cab38366a353310f252822b6e8b1555fa7",
    "US-LIZARDTECH-2005": "18234cd57da8be8f9663e438170ceb4b1e638ad8d879862051feac80acd72429",
    "US-LIEBEL-2007": "e17d5cc283bd8c2d4da265a88673e82ceb1b8f9f554e7a66020883bea1b24859",
    "US-AMGEN-SANOFI-2023": "998d57d81cdec39e672344b7a64273ba0071cf53ee8dc225f34828b5ecd12e86",
    "US-JUNO-2021": "fdc06c87155b27e5ed2e0442bdbab0e09903cbcc42d239406110c032479ceab5",
    "US-IDENIX-2019": "81937f332bb9ef7c8f4e3d06ed1fc337a1fdc24302d0159e0a29570a75a1529f",
    "US-ENZO-2010": "0d5697346ed184ccea8a21d1d8c4177ab0ed6329daa36e7ca685b387470b1f8b",
    "US-NYSTROM-2005": "5b4e76c97f7cd45035b0c3b13157dde72be8f9a9b5f87e2f31bb185f7fd1231e",
    "US-THORNER-2012": "1465d7c59dfdfce6b71e45d516b952f252e74bc519f6c3291f5fc4045ad8fb5d",
}


def norm(text: str) -> str:
    return " ".join((text or "").split())


def canonical_hash(row: dict) -> str:
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", value.upper()).strip("-")


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    case_name: str
    docket: str
    court: str
    decision_date: str
    path_fragment: str
    patent_ids: tuple[str, ...]
    lineage_notes: str
    issue: str
    principle: str
    non_rule: str
    claim_term: str
    tech: str
    risk: str
    drafting_advice: str
    review_advice: str
    quote_terms: tuple[str, ...]


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        "US-MARKMAN-1996",
        "Markman v. Westview Instruments, Inc.",
        "95-26",
        "Supreme Court of the United States",
        "1996-04-23",
        "1996-04-23__SCOTUS__95-26__Markman_v_Westview__Opinion__Precedential.pdf",
        ("US RE33,054",),
        "Supreme Court review of Federal Circuit appeal; patent identified in the opinion as United States Reissue Patent No. 33,054.",
        "claim construction allocation",
        "청구항 용어 해석은 미국 사건에서 법원이 수행하는 법률문제라는 점을 전제로, 초안 단계부터 재판부가 읽을 문언·명세서·전문용어 기록을 분리해 남겨야 한다.",
        "배심 판단의 부재를 이유로 청구항 문언의 모호함을 방치해도 된다는 규칙이 아니다.",
        "inventory",
        "dry-cleaning inventory tracking",
        "disputed technical term without a clean intrinsic record",
        "미국 소송 가능성이 있는 청구항은 용어 정의 후보, 실시예와의 관계, 전문가 설명이 필요한 사실을 명세서에 정리한다.",
        "청구항 해석 쟁점은 침해·무효 판단 전에 별도 법률 쟁점으로 분리해 제시한다.",
        ("We accordingly think", "better suited to find"),
    ),
    CaseSpec(
        "US-PHILLIPS-2005",
        "Phillips v. AWH Corp.",
        "03-1269, -1286",
        "United States Court of Appeals for the Federal Circuit",
        "2005-07-12",
        "2005-07-12__CAFC__2003-1269__PHILLIPS_v_AWH__Opinion__Precedential.pdf",
        ("US 4,677,798",),
        "En banc Federal Circuit opinion; errata exists in corpus but this record uses the merits opinion.",
        "claim construction context",
        "미국 청구항 용어는 통상의 의미에서 출발하되 명세서와 심사기록을 포함한 내재 증거의 무게를 함께 검토해야 한다.",
        "모든 실시예 세부사항을 독립항에 자동 편입하거나, 사전 의미만으로 명세서를 배제하는 규칙이 아니다.",
        "baffles",
        "steel modular wall panels",
        "over-importing embodiment limits or ignoring dependent claim differentiation",
        "독립항에는 필요한 구조와 기능만 두고, 특정 각도·중첩·방탄 기능은 종속항으로 분리해 문맥상 중복을 피한다.",
        "상대방이 실시예 제한을 끌어오면 청구항 차이와 명세서의 사용 방식을 함께 반박한다.",
        ("ordinary and customary meaning", "intrinsic evidence", "no magic formula"),
    ),
    CaseSpec(
        "US-TEVA-SANDOZ-2015",
        "Teva Pharmaceuticals USA, Inc. v. Sandoz, Inc.",
        "12-1567",
        "United States Court of Appeals for the Federal Circuit",
        "2015-06-18",
        "2015-06-18__CAFC__12-1567__TEVA_PHARMACEUTICALS_USA_v_SANDOZ__Opinion__Precedential.pdf",
        ("US 5,800,808", "US 6,620,847", "US 6,939,539"),
        "Federal Circuit remand after Supreme Court vacatur in No. 13-854; selected because it states the remand rule and identifies patents in suit.",
        "claim construction fact review",
        "미국 청구항 해석에서 최종 해석은 법률문제이나, 하급심이 기초 과학 사실을 인정한 경우 그 사실 인정은 명확한 오류 기준과 분리해 다뤄야 한다.",
        "전문가 진술이 있으면 청구항이 자동으로 명확해진다는 규칙이 아니다.",
        "molecular weight",
        "Copaxone manufacturing",
        "ambiguous measurement method",
        "측정 방식이 둘 이상인 수치 한정은 청구항 또는 명세서에 계산 기준과 사용 맥락을 명시한다.",
        "항소 검토에서는 용어 의미 자체와 그 전제인 과학적 사실 판단을 따로 공격하거나 방어한다.",
        ("clear error review", "molecular weight", "We hold that claim 1"),
    ),
    CaseSpec(
        "US-NAUTILUS-2014",
        "Nautilus, Inc. v. Biosig Instruments, Inc.",
        "13-369",
        "Supreme Court of the United States",
        "2014-06-02",
        "2014-06-02__SCOTUS__13-369__Nautilus_v_Biosig__Opinion__Precedential.pdf",
        ("US 5,337,753",),
        "Supreme Court review of Biosig Instruments v. Nautilus Federal Circuit lineage.",
        "definiteness",
        "미국 명확성 검토에서는 청구항·명세서·심사기록을 본 숙련자가 발명의 범위를 합리적 확실성으로 알 수 있는지 확인해야 한다.",
        "절대적 정밀성을 요구하거나 어려운 해석 문제만으로 곧바로 불명확하다는 규칙이 아니다.",
        "spaced relationship",
        "heart-rate exercise monitor electrodes",
        "relative spacing without objective boundaries",
        "상대적 위치·간격 표현은 기능 목적, 기준점, 허용 범위를 명세서에 연결한다.",
        "불명확성 공격은 모호함 자체보다 숙련자가 범위를 확정할 객관적 기준이 부족하다는 점에 집중한다.",
        ("reasonable certainty", "In place of the “insolubly ambiguous” standard"),
    ),
    CaseSpec(
        "US-WILLIAMSON-2015",
        "Williamson v. Citrix Online, LLC",
        "13-1130",
        "United States Court of Appeals for the Federal Circuit",
        "2015-06-16",
        "2015-06-16__CAFC__13-1130__RICHARD_WILLIAMSON_v_CITRIX_ONLINE__Opinion__Precedential.pdf",
        ("US 6,155,840",),
        "En banc Federal Circuit opinion replacing the earlier 2014 panel opinion in the same docket.",
        "means-plus-function and nonce words",
        "미국 소프트웨어·장치 청구항에서 module 같은 포괄 명칭이 기능만 수행하는 검은상자로 쓰이면 §112(f) 적용과 알고리즘 부재 문제가 생길 수 있다.",
        "module이라는 단어의 사용 자체를 금지하거나 모든 기능식 표현을 §112(f)로 보는 규칙이 아니다.",
        "distributed learning control module",
        "distributed online learning software",
        "nonce component label with functions but no linked algorithm",
        "기능식 구성요소에는 구조적 명칭, 데이터 흐름, 처리 단계 또는 알고리즘을 명세서와 청구항 문맥에 연결한다.",
        "상대방의 §112(f) 주장은 용어 전체가 구조를 전달하는지와 명세서가 대응 구조를 명확히 연결하는지로 나눠 답한다.",
        ("nonce word", "corresponding structure", "algorithm"),
    ),
    CaseSpec(
        "US-BIOMEDINO-2007",
        "Biomedino, LLC v. Waters Technologies Corp.",
        "2006-1350",
        "United States Court of Appeals for the Federal Circuit",
        "2007-06-18",
        "2007-06-18__CAFC__2006-1350__BIOMEDINO_v_WATERS__Opinion__Precedential.pdf",
        ("US 6,602,502",),
        "Federal Circuit appeal from W.D. Wash.; selected for corresponding-structure reasoning.",
        "means-plus-function corresponding structure",
        "미국 §112(f) 청구항은 명세서가 기능을 수행하는 대응 구조를 실제로 연결해야 하며, 박스명이나 결과 설명만으로는 부족할 수 있다.",
        "도면 블록이나 상업 제품명이 언제나 구조로 불충분하다는 규칙은 아니다.",
        "control means",
        "automated chemical assay apparatus",
        "means term with no identifiable corresponding structure",
        "means 용어를 쓸 때는 해당 기능을 수행하는 회로·밸브·프로세서 단계 등 구체 구조를 명세서에서 식별한다.",
        "무효 검토에서는 기능 자체의 유용성보다 그 기능과 연결된 대응 구조의 유무를 확인한다.",
        ("corresponding structure", "control means", "indefinite"),
    ),
    CaseSpec(
        "US-ARISTOCRAT-2008",
        "Aristocrat Technologies Australia Pty Ltd. v. International Game Technology",
        "2007-1419",
        "United States Court of Appeals for the Federal Circuit",
        "2008-03-28",
        "2008-03-28__CAFC__2007-1419__ARISTOCRAT_TECH_AUSTRALIA_PTY_v_INTERNATIONAL_GAME_TECH__Opinion__Precedential.pdf",
        ("US 5,885,215",),
        "Federal Circuit precedential software means-plus-function opinion.",
        "software means-plus-function algorithm",
        "미국 컴퓨터 구현 §112(f) 제한은 범용 컴퓨터 언급만으로 충분하지 않고 기능을 수행하는 알고리즘이 구조로 제시되어야 할 수 있다.",
        "소프트웨어 발명에는 항상 소스코드가 필요하다는 규칙이 아니다.",
        "game control means",
        "electronic slot-machine feature game",
        "functional computer implementation without disclosed algorithm",
        "컴퓨터가 수행하는 기능은 흐름도, 의사코드, 수학식 또는 단계 설명으로 명세서에 남긴다.",
        "침해 전에는 먼저 대응 알고리즘과 등가 범위를 확정해 주장 범위가 순수 기능으로 흐르지 않게 한다.",
        ("specific algorithm", "general purpose computer", "corresponding structure"),
    ),
    CaseSpec(
        "US-NOAH-2012",
        "Noah Systems, Inc. v. Intuit Inc.",
        "2011-1390",
        "United States Court of Appeals for the Federal Circuit",
        "2012-04-09",
        "2012-04-09__CAFC__2011-1390__NOAH_SYSTEMS_v_INTUIT__Opinion__Precedential.pdf",
        ("US 5,875,435",),
        "Federal Circuit appeal from W.D. Pa. involving the access means limitation.",
        "multiple functions in software means-plus-function",
        "미국 컴퓨터 구현 means 제한이 여러 기능을 담으면 명세서는 각 식별 가능한 기능을 수행하는 알고리즘을 충분히 제공해야 한다.",
        "하나의 알고리즘이 모든 세부 기능을 자동으로 뒷받침한다는 규칙이 아니다.",
        "access means",
        "automated financial accounting system",
        "one disclosed routine for fewer than all claimed functions",
        "복수 기능 제한은 기능별 처리 단계를 분해하고, 공통 단계와 별도 단계를 구분해 기재한다.",
        "검토자는 기능 일부만 지원되는지 확인하고 미지원 기능은 청구항 분할 또는 명세서 보강 대상으로 표시한다.",
        ("access means", "we conclude that where, as here", "multiple identifiable functions"),
    ),
    CaseSpec(
        "US-INTERVAL-2014",
        "Interval Licensing LLC v. AOL, Inc.",
        "13-1282",
        "United States Court of Appeals for the Federal Circuit",
        "2014-09-10",
        "2014-09-10__CAFC__13-1282__INTERVAL_LICENSING_v_AOL__Opinion__Precedential.pdf",
        ("US 6,034,652", "US 6,788,314"),
        "Federal Circuit appeal from W.D. Wash.; later 2018 lineage exists for remaining claims.",
        "subjective claim language",
        "미국 청구항의 주관적 표현은 명세서가 숙련자에게 의미 있게 정밀한 경계를 주는 객관적 기준을 제공하는지 확인해야 한다.",
        "사용자 경험 관련 표현이 모두 금지된다는 규칙이 아니다.",
        "unobtrusive manner",
        "attention-manager display system",
        "subjective UX boundary without objective claim scope",
        "방해하지 않음·눈에 띄지 않음 같은 표현은 위치, 시간, 빈도, 사용자 입력 기준 등 측정 가능한 지표와 연결한다.",
        "무효 위험 평가는 좋은 UX라는 결과보다 어떤 표시가 경계 안팎인지 숙련자가 구별할 기준을 묻는다.",
        ("unobtrusive manner", "subjective", "little guidance"),
    ),
    CaseSpec(
        "US-DATAMIZE-2005",
        "Datamize, LLC v. Plumtree Software, Inc.",
        "2004-1564",
        "United States Court of Appeals for the Federal Circuit",
        "2005-08-05",
        "2005-08-05__CAFC__2004-1564__DATAMIZE_L_L_C_v_PLUMTREE_SOFTWARE__Opinion__Precedential.pdf",
        ("US 6,014,137",),
        "Federal Circuit appeal from N.D. Cal.; related Plumtree v. Datamize on-sale lineage exists but is not used for merits here.",
        "aesthetic subjectivity and definiteness",
        "미국 청구항에서 aesthetically pleasing 같은 미적 결과 표현은 명세서가 객관적 앵커를 제공하지 않으면 명확성 위험이 크다.",
        "디자인·화면 품질 요소를 청구항에 절대 포함할 수 없다는 규칙이 아니다.",
        "aesthetically pleasing",
        "interface-authoring software",
        "purely subjective aesthetic term",
        "시각 품질을 청구하려면 정렬, 대비, 간격, 사용자 선택 규칙 등 검증 가능한 구성 또는 절차로 바꾼다.",
        "상대방의 불명확성 주장은 개인 취향 차이가 아니라 명세서가 객관적 범위 기준을 주지 않는다는 점으로 구성한다.",
        ("aesthetically pleasing", "purely subjective", "indefinite"),
    ),
    CaseSpec(
        "US-HALLIBURTON-2008",
        "Halliburton Energy Services, Inc. v. M-I LLC",
        "2007-1149",
        "United States Court of Appeals for the Federal Circuit",
        "2008-01-25",
        "2008-01-25__CAFC__2007-1149__HALLIBURTON_ENERGY_SVCS_v_M_I__Opinion__Precedential.pdf",
        ("US 6,887,832 B2",),
        "Federal Circuit appeal from E.D. Tex.; selected for fragile-gel definiteness reasoning.",
        "relative material-property definiteness",
        "미국 청구항의 물성 표현은 시험 조건과 비교 기준이 없어 숙련자가 범위를 가르기 어렵다면 명확성 문제가 된다.",
        "모든 기능적 물성 용어가 불명확하다는 규칙은 아니다.",
        "fragile gel",
        "drilling fluid gels",
        "functional material property lacking boundary",
        "물성 기능은 전단 조건, 회복 시간, 점도 범위, 측정 장비처럼 재현 가능한 시험 기준과 함께 기재한다.",
        "검토 의견은 용어의 사전 의미보다 실제 침해 제품을 경계 안팎으로 나누는 기준의 부재를 확인한다.",
        ("fragile gel", "ambiguous", "indefinite"),
    ),
    CaseSpec(
        "US-ARIAD-2010",
        "Ariad Pharmaceuticals, Inc. v. Eli Lilly & Co.",
        "2008-1248",
        "United States Court of Appeals for the Federal Circuit",
        "2010-03-22",
        "2010-03-22__CAFC__2008-1248__ARIAD_PHARMACEUTICALS_v_ELI_LILLY_AND__Opinion__Precedential.pdf",
        ("US 6,410,516",),
        "En banc Federal Circuit opinion after panel decision and rehearing order.",
        "written description possession",
        "미국 written description은 출원 시점에 발명자가 청구 발명을 실제 보유했음을 명세서가 객관적으로 보여주는지 묻는다.",
        "작동 원리나 연구 목표를 말하면 모든 기능적 속 청구항이 곧바로 지원된다는 규칙이 아니다.",
        "reducing NF-kB activity",
        "cell-signaling biotechnology",
        "broad functional genus without representative species or common structure",
        "넓은 기능적 속은 대표 종, 공통 구조, 선택 기준을 명세서에 남겨 보유 사실을 보여준다.",
        "무효 검토는 실시 가능성과 별도로 청구 범위 전체를 발명자가 보유했는지 확인한다.",
        ("written description", "had possession", "representative number of species"),
    ),
    CaseSpec(
        "US-LIZARDTECH-2005",
        "LizardTech, Inc. v. Earth Resource Mapping, Inc.",
        "2005-1062",
        "United States Court of Appeals for the Federal Circuit",
        "2005-10-04",
        "2005-10-04__CAFC__2005-1062__LIZARDTECH_v_EARTH_RESOURCE_MAPPING__Opinion__Precedential.pdf",
        ("US 5,710,835",),
        "Federal Circuit merits opinion; en banc rehearing order exists in the corpus.",
        "written description for broadened generic claims",
        "미국 written description에서 한 구현 방식의 상세 설명은 그보다 넓은 모든 방식의 청구를 자동으로 뒷받침하지 않는다.",
        "대표 실시예 하나가 언제나 부족하다는 규칙이 아니다.",
        "seamless DWT",
        "digital image compression",
        "generic claim broader than disclosed seamless implementation",
        "개선 원리를 넓게 청구하려면 대체 구현을 가능하게 하는 공통 원리와 예시를 함께 기재한다.",
        "검토자는 실시예의 장점 설명이 전체 속의 보유 증거인지, 한 방식의 성공담인지 구분한다.",
        ("written description", "generic claim", "seamless"),
    ),
    CaseSpec(
        "US-LIEBEL-2007",
        "Liebel-Flarsheim Co. v. Medrad, Inc.",
        "2006-1156",
        "United States Court of Appeals for the Federal Circuit",
        "2007-03-22",
        "2007-03-22__CAFC__2006-1156__LIEBEL_FLARSHEIM_v_MEDRAD__Opinion__Precedential.pdf",
        ("US 5,456,669", "US 5,658,261", "US 5,662,612", "US 5,928,197"),
        "Federal Circuit appeal following earlier claim-construction litigation between the parties.",
        "enablement after broad construction",
        "미국에서 명세서 제한을 피하려고 넓은 청구범위를 유지하면, 그 전체 범위를 과도한 실험 없이 실시 가능하게 해야 한다.",
        "넓게 청구하면 언제나 무효라는 규칙이 아니다.",
        "pressure jacket",
        "fluid injector systems",
        "claims broad enough to cover jacketless injectors without enabling support",
        "실시예에 없는 무압력 재킷 없는 변형까지 청구하려면 그 변형을 만드는 방법과 안전한 작동 조건을 기재한다.",
        "소송 의견은 넓은 해석으로 침해를 얻는 이익과 전체 범위 enablement 위험을 함께 평가한다.",
        ("the full scope of the claimed inventions", "district court was correct", "pressure jacket"),
    ),
    CaseSpec(
        "US-AMGEN-SANOFI-2023",
        "Amgen Inc. v. Sanofi",
        "21-757",
        "Supreme Court of the United States",
        "2023-05-18",
        "2023-05-18__SCOTUS__21-757__Amgen_v_Sanofi__Opinion__Precedential.pdf",
        ("US 8,829,165", "US 8,859,741"),
        "Supreme Court review of Federal Circuit Amgen v. Sanofi enablement lineage.",
        "enablement of functional genus",
        "미국 기능적 속 청구항은 특정 예시뿐 아니라 청구된 전체 부류를 만들고 사용할 수 있게 해야 하며, 더 많이 청구할수록 더 많이 enable 해야 한다.",
        "속 청구항이나 항체 청구항이 항상 불가능하다는 규칙이 아니다.",
        "PCSK9 antibodies",
        "antibody therapeutics",
        "functional genus far larger than taught examples",
        "결합 위치와 차단 기능으로 속을 청구할 때는 대표 항체, 구조적 공통점, 예측 가능한 선별 지침을 충분히 둔다.",
        "자문은 발견 로드맵인지 실제 실시 교시인지 구분하고, 전 범위를 반복 실험에 맡기는지 검토한다.",
        ("full scope", "What is reasonable in any case", "two research assignments"),
    ),
    CaseSpec(
        "US-JUNO-2021",
        "Juno Therapeutics, Inc. v. Kite Pharma, Inc.",
        "20-1758",
        "United States Court of Appeals for the Federal Circuit",
        "2021-08-26",
        "2021-08-26__CAFC__20-1758__JUNO_THERAPEUTICS_v_KITE_PHARMA__Opinion__Precedential.pdf",
        ("US 7,446,190",),
        "Federal Circuit appeal from C.D. Cal. judgment involving CAR-T scFv claims.",
        "written description for antibody binding genus",
        "미국 written description에서 기능으로 정의된 거대한 scFv 속은 대표 종이나 공통 구조가 부족하면 보유 입증이 약하다.",
        "항체 분야에서 기능적 언어가 모두 금지된다는 규칙은 아니다.",
        "scFv binding element",
        "CAR-T immunotherapy",
        "millions-or-more binding candidates supported by few examples",
        "결합 요소 속은 표적·서열·결합 특성별 대표 예와 공통 구조 분석을 명세서에 넣는다.",
        "검토 의견은 알려진 표적 결합 사실과 청구된 전체 scFv 속의 보유 증거를 분리한다.",
        ("written description fails", "representative species", "distinguish between scFvs"),
    ),
    CaseSpec(
        "US-IDENIX-2019",
        "Idenix Pharmaceuticals LLC v. Gilead Sciences Inc.",
        "18-1691",
        "United States Court of Appeals for the Federal Circuit",
        "2019-10-30",
        "2019-10-30__CAFC__18-1691__IDENIX_PHARMACEUTICALS_v_GILEAD_SCIENCES__Opinion__Precedential.pdf",
        ("US 7,608,597",),
        "Federal Circuit appeal from D. Del. JMOL; related patents are discussed but the asserted patent is the '597 patent.",
        "enablement and written description for chemical genus",
        "미국 화학 속 청구항은 유효 화합물을 전 범위에서 과도한 실험 없이 찾을 수 있는지와 보유 증거가 있는지를 함께 점검해야 한다.",
        "화합물 수가 많다는 사실만으로 항상 non-enablement가 되는 규칙은 아니다.",
        "2'-methyl-up nucleosides",
        "HCV nucleoside treatments",
        "billions of candidate compounds with insufficient direction",
        "마쿠쉬·기능적 화학 속은 작동 치환기 범위, 실패 예, 선별 규칙, 대표 화합물을 균형 있게 기재한다.",
        "자문은 accused embodiment 하나의 성공보다 청구범위 전체의 예측 가능성과 실험 부담을 본다.",
        ("full scope", "We conclude that they would not", "It is undisputed"),
    ),
    CaseSpec(
        "US-ENZO-2010",
        "Enzo Biochem, Inc. v. Applera Corp.",
        "2009-1281",
        "United States Court of Appeals for the Federal Circuit",
        "2010-03-26",
        "2010-03-26__CAFC__2009-1281__ENZO_BIOCHEM_v_APPLERA__Opinion__Precedential.pdf",
        ("US 5,328,824", "US 5,449,767", "US 5,476,928", "US 5,082,830"),
        "Federal Circuit precedential opinion in the Applera dispute; selected for functional biochemical limitation and indefiniteness reasoning.",
        "functional biochemical limitation definiteness",
        "미국 생명공학 청구항의 기능적 표현도 명세서의 예시와 기술 문맥이 숙련자에게 경계를 제공하면 불명확성 공격을 방어할 수 있다.",
        "기능식 생명공학 표현이 언제나 명확하거나 written description까지 자동 충족한다는 규칙이 아니다.",
        "not interfering substantially",
        "nucleic-acid probes for bacteria detection",
        "functional linkage language attacked as indefinite",
        "간섭하지 않는다는 결과 표현은 검출·하이브리드화 조건, 예시 링커, 허용 가능한 성능 저하 기준과 연결한다.",
        "검토자는 기능 표현이 명세서 예시와 숙련자 지식으로 경계화되는지, 별도의 written description·enablement 문제는 남는지 나눠 본다.",
        ("not interfering substantially", "indefinite", "hybridize"),
    ),
    CaseSpec(
        "US-NYSTROM-2005",
        "Nystrom v. Trex Co.",
        "2003-1092",
        "United States Court of Appeals for the Federal Circuit",
        "2005-09-14",
        "2005-09-14__CAFC__2003-1092__NYSTROM_v_TREX__Opinion__Precedential.pdf",
        ("US 5,474,831",),
        "Federal Circuit claim-construction appeal involving decking board terms.",
        "ordinary meaning constrained by intrinsic use",
        "미국 청구항의 평이한 용어도 명세서 전체에서 발명자가 어떻게 사용했는지와 청구항 문맥에 의해 범위가 좌우된다.",
        "일상어는 언제나 가장 넓은 사전 의미를 갖는다는 규칙이 아니다.",
        "board",
        "decking boards",
        "term with broad dictionary meaning but narrower intrinsic context",
        "일상어를 쓰더라도 합성재·목재·형상 같은 의도한 범위를 명세서에서 명확히 하거나 청구항에 넣는다.",
        "침해 검토는 사전 정의 후보보다 명세서가 반복적으로 가리킨 물건과 실시예를 먼저 확인한다.",
        ("ordinary meaning", "board", "written description"),
    ),
    CaseSpec(
        "US-THORNER-2012",
        "Thorner v. Sony Computer Entertainment America LLC",
        "2011-1114",
        "United States Court of Appeals for the Federal Circuit",
        "2012-02-01",
        "2012-02-01__CAFC__2011-1114__THORNER_v_SONY_COMPUTER_ENTERTAINMENT_AMERICA__Opinion__Precedential.pdf",
        ("US 6,422,941",),
        "Federal Circuit appeal from D.N.J. stipulated non-infringement after claim construction.",
        "lexicography and disavowal",
        "미국에서 평이한 의미를 벗어나려면 명세서가 특별 정의 또는 명확한 범위 포기를 보여야 하며, 단일 실시예 반복만으로는 부족할 수 있다.",
        "명세서 실시예가 청구항 해석에 언제나 무관하다는 규칙이 아니다.",
        "attached to said pad",
        "tactile feedback game controller",
        "narrowing ordinary attachment meaning without clear definition or disclaimer",
        "특별 의미를 의도하면 정의 문장을 명시하고, 배제 의도가 있으면 그 범위를 분명히 쓴다.",
        "검토자는 실시예 반복, 선호 표현, 비판 표현이 명확한 정의·포기에 이르렀는지 엄격히 구분한다.",
        ("lexicographer", "disavowed", "plain and ordinary meaning"),
    ),
)


TERM_MIN_PAGES = {
    ("US-MARKMAN-1996", "better suited to find"): 10,
    ("US-NAUTILUS-2014", "reasonable certainty"): 4,
    ("US-AMGEN-SANOFI-2023", "What is reasonable in any case"): 10,
    ("US-AMGEN-SANOFI-2023", "two research assignments"): 10,
    ("US-ARIAD-2010", "had possession"): 26,
    ("US-JUNO-2021", "representative species"): 9,
}


def inventory_paths() -> list[str]:
    paths: list[str] = []
    with INVENTORY.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            path = rec.get("path", "")
            if path.endswith(".pdf"):
                paths.append(path)
    return paths


def find_pdf(fragment: str, paths: list[str]) -> str:
    hits = [path for path in paths if fragment in path]
    if len(hits) != 1:
        raise RuntimeError(f"{fragment}: expected one PDF, found {len(hits)}")
    return hits[0]


def extract_pages(pdf_path: str) -> list[str]:
    reader = PdfReader(pdf_path)
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


def quote_from_page(page_text: str, term: str, min_len: int = 220, max_len: int = 440) -> str:
    text = norm(page_text)
    pos = text.casefold().find(term.casefold())
    if pos < 0:
        raise RuntimeError(f"term not found: {term}")
    start = max(0, pos - min_len // 2)
    stop = min(len(text), pos + max_len - min_len // 2)
    while start > 0 and text[start - 1] not in ".;:!?":
        start -= 1
        if pos - start > max_len // 2:
            break
    while stop < len(text) and text[stop - 1] not in ".;:!?":
        stop += 1
        if stop - start > max_len:
            break
    quote = text[start:stop].strip(" ,;")
    if len(quote) > max_len:
        cut = quote[:max_len].rsplit(" ", 1)[0]
        quote = cut.strip(" ,;")
    return quote


def find_span(case_id: str, pages: list[str], term: str, used_pages: set[int], span_number: int) -> dict:
    matches = [idx for idx, text in enumerate(pages, 1) if term.casefold() in norm(text).casefold()]
    min_page = TERM_MIN_PAGES.get((case_id, term))
    if min_page is not None:
        later_matches = [idx for idx in matches if idx >= min_page]
        if later_matches:
            matches = later_matches
    if not matches:
        raise RuntimeError(f"{case_id}: no page for {term!r}")
    page_no = next((idx for idx in matches if idx not in used_pages), matches[0])
    used_pages.add(page_no)
    quote = quote_from_page(pages[page_no - 1], term)
    return {
        "id": f"{case_id}-E{span_number}",
        "pdf_page": page_no,
        "quote": quote,
        "speaker": "court_reasoning",
        "section": f"opinion discussion containing {term}",
    }


EPISODES = {
    "US-MARKMAN-1996": [
        {
            "episode_id": "US-MARKMAN-1996-EP1",
            "question": "미국 배심재판을 앞둔 침해소송에서 'inventory' 용어 해석 자료를 어떻게 정리할지 조언하라.",
            "scenario_facts": "의뢰인은 세탁소 의류 추적 시스템 특허권자이다. 독립항은 키보드 입력, 데이터 프로세서, 광학 판독 가능한 표식 생성을 포함하지만 'inventory'가 고객별 의류 목록인지 거래 기록 전체인지 다투어진다. 피고는 배심이 업계 관행을 듣고 넓게 판단해야 한다고 주장한다. 명세서에는 거래 레코드와 위치 추적 예가 따로 있고, 전문가 보고서는 두 의미를 모두 언급한다.",
            "reference_answer": "미국 Markman의 역사적 범위에서는 용어 의미를 배심 설득 주제로만 두지 말고 법원의 claim construction 쟁점으로 분리해야 한다. 법원은 특허 내부의 일관성을 보존하기 위해 명세서와 청구항에 맞는 전문가 정의를 선별할 위치에 있다는 점을 보았다. 따라서 'inventory'의 가능한 의미를 청구항 문언, 명세서의 거래 레코드 예, 심사기록의 표현별로 표로 정리하고, 전문가 의견은 법원이 기술 용어의 획득 의미를 이해하는 보조 자료로 배치한다. 반론은 업계 관행이 사실문제처럼 보인다는 점이지만, 그 사실은 해석의 전제가 될 뿐 최종 범위 결정은 법원이 맡는 구조로 제시한다. 빠진 자료는 심사 중 'inventory'를 축소한 발언, 당시 POSITA가 쓰던 세탁업 재고관리 문서, 피고 시스템의 거래 기록 필드이다.",
            "required_points": ["US civil litigation context", "claim construction is for the court", "expert evidence as aid not final decider", "intrinsic record organization", "missing prosecution and industry-usage evidence"],
            "forbidden_claims": ["1996년 절차에 없는 사후 제도를 선택지로 제시", "배심이 용어 해석을 최종 결정한다고 설명", "전문가 의견 하나로 claim scope 확정"],
            "inferior_answer": "이 사안은 'inventory'가 업계에서 넓게 쓰인다는 전문가 설명이 있으므로 배심에게 두 의미를 모두 제시하는 방식이 가장 유리하다. Markman은 법원이 특허문서를 읽는다는 원칙을 말하지만 기술 용어의 실제 의미는 사실문제에 가깝다. 따라서 청구항 해석 신청에서는 좁은 거래 레코드 예를 길게 다루기보다, 피고 시스템도 의류 위치와 고객 정보를 저장한다는 침해 사실을 중심에 두는 편이 낫다. 심사기록은 불리한 제한 발언이 발견될 때만 보조적으로 확인하면 충분하다.",
            "preference_reason": "열등 답변은 전문가·배심 프레임을 전략적으로 말하지만 Markman의 핵심인 법원 주도 해석과 특허 내부 일관성 점검을 뒤로 미룬다."
        },
        {
            "episode_id": "US-MARKMAN-1996-EP2",
            "question": "동일 특허가 여러 세탁 체인에 대해 집행될 가능성이 있을 때 청구항 해석 전략을 세워라.",
            "scenario_facts": "의뢰인은 첫 피고와 합의 후 다른 체인에도 경고장을 보내려 한다. 첫 소송에서는 'transaction record'가 바코드 출력 전 생성되는 내부 데이터인지, 출력된 영수증까지 포함하는지 다투어졌다. 새 피고 제품은 모바일 앱 영수증만 만들고 종이 바코드는 없다. 기존 전문가 의견은 하드웨어 판독기를 전제로 작성되었다.",
            "reference_answer": "Markman이 강조한 통일성 이유를 고려하면 첫 소송의 해석 기록을 다음 집행까지 버티는 범위로 설계해야 한다. 법원 해석은 한 사건의 침해 판단을 넘어서 같은 특허의 반복 분쟁에서 예측 가능성을 만든다. 그러므로 'transaction record'를 내부 데이터 구조로 넓게 주장하려면 명세서가 종이 출력과 전자 기록을 구분하는지, 바코드가 필수인지 예시인지, 청구항의 다른 요소와 충돌하지 않는지 먼저 확인한다. 반대편은 첫 전문가 보고서가 광학 판독기 중심이었다고 좁힐 것이다. 누락 증거는 기존 Markman 명령의 문언, 합의서의 해석 유보 여부, 모바일 앱 기록이 청구항의 광학 표식 요소를 대체하는지 여부이다.",
            "required_points": ["uniformity rationale", "future civil enforcement implications", "claim term tied to other limitations", "counterargument from prior expert framing", "missing prior order and settlement context"],
            "forbidden_claims": ["한 피고 합의가 곧 모든 피고에 대한 동일 해석이라고 단정", "모바일 앱을 바코드와 사실상 같다고 무근거 처리", "current law good-law statement without review"],
            "inferior_answer": "다음 피고들이 모바일 영수증을 쓰더라도 같은 세탁 관리 기능을 수행하므로 첫 소송에서 넓은 합의를 얻었다는 점을 강조하면 된다. Markman의 통일성 논리는 동일 특허에 같은 의미를 주려는 것이므로, 한 번 넓은 해석을 주장했다면 이후 사건에서도 그 주장을 유지할 수 있다. 바코드와 광학 판독기 부분은 구현 차이에 불과하므로 경고장 단계에서 자세히 분석하면 오히려 피고에게 회피 설계를 알려줄 위험이 있다.",
            "preference_reason": "열등 답변은 통일성 원칙을 넓은 주장 유지의 근거로만 쓰고, 해석이 청구항의 다른 구조적 한정과 충돌할 위험을 평가하지 않는다."
        },
    ],
    "US-PHILLIPS-2005": [
        {
            "episode_id": "US-PHILLIPS-2005-EP1",
            "question": "'baffles'가 특정 방탄 각도를 포함하는지에 대한 미국식 청구항 검토 의견을 작성하라.",
            "scenario_facts": "의뢰인은 강철 모듈 벽 특허의 후속출원을 준비한다. 독립항 초안은 'internal baffles extending inwardly'만 쓰고, 종속항에는 투사체를 편향시키는 각도와 중첩 구조를 둔다. 영업팀은 모든 제품이 각도형 방탄 구조라며 독립항에도 그 기능을 넣자고 한다. 경쟁사는 사전상 baffle 의미가 흐름 방해물이라며 방탄 각도는 제한되지 않는다고 본다.",
            "reference_answer": "미국 Phillips의 역사적 범위에서는 사전 의미, 청구항 차이, 명세서 사용을 함께 보아야 한다. 독립항의 'baffles'를 사전상 방해물 의미로 출발하되, 종속항이 각도와 투사체 편향을 별도로 추가한다면 그 제한이 이미 독립항에 들어 있다고 쓰면 종속항을 중복시킬 위험이 있다. 초안은 독립항에 하중 지지와 내부 돌출이라는 최소 구조를 남기고, 방탄 각도·중첩·연속 장벽은 별도 종속항으로 둔다. 반론은 모든 실시예가 방탄 각도형이라는 점인데, Phillips는 실시예와 청구범위를 자동 동일시하지 말고 특허 문맥에서 실제 발명 범위를 확인하라고 요구한다. 더 볼 자료는 발명의 요약, 문제 해결 문장, 심사 중 baffle 의미에 대한 진술이다.",
            "required_points": ["ordinary meaning plus intrinsic evidence", "dependent-claim differentiation", "avoid importing every embodiment detail", "specific baffle facts", "missing specification and prosecution statements"],
            "forbidden_claims": ["사전 정의만으로 결론", "모든 실시예 세부사항을 독립항에 강제", "종속항 중복 문제 누락"],
            "inferior_answer": "제품이 실제로 방탄 각도형으로만 판매될 예정이라면 독립항의 baffles도 그 각도를 포함한다고 보는 편이 자연스럽다. Phillips는 명세서가 가장 좋은 안내서라고 하므로 모든 실시예가 같은 방향을 보이면 그 실시예를 독립항 의미로 삼는 것이 안정적이다. 종속항의 각도 표현은 침해 입증을 쉽게 하기 위한 확인적 한정으로 설명할 수 있어 중복 문제는 크지 않다.",
            "preference_reason": "열등 답변은 명세서 중심 접근을 과장해 종속항의 구체 제한이 독립항을 이미 제한한다는 미묘한 오류를 만든다."
        },
        {
            "episode_id": "US-PHILLIPS-2005-EP2",
            "question": "경쟁사가 평평한 내부 보강판을 쓰는 경우, Phillips 관점에서 비침해/침해 주장의 강약을 평가하라.",
            "scenario_facts": "등록청구항은 내부 강철 baffles가 벽에서 안쪽으로 연장된다고만 쓴다. 명세서 도면은 대부분 사선 보강판이고, 한 문단은 탄환을 빗나가게 하는 장점을 말한다. 경쟁 제품은 수직 평판을 넣어 하중 지지 기능은 있으나 탄환 편향 기능은 약하다. 파일히스토리에는 prior art와 구별하면서 'load-bearing internal steel structures'라는 표현을 썼다.",
            "reference_answer": "Phillips식 분석은 'baffles'의 통상 의미를 먼저 보되 내재 증거가 그 의미를 어떻게 조정하는지 확인한다. 경쟁 제품이 흐름 또는 힘을 방해하고 하중 지지 구조라면 침해 주장은 가능하지만, 명세서가 발명을 탄환 편향 각도로 반복 정의했거나 심사 중 그렇게 제한했다면 약해진다. 반대로 파일히스토리의 'load-bearing internal steel structures'는 각도보다 하중 지지를 강조해 넓은 해석에 도움이 된다. 필요한 추가 증거는 prior art 회피 문장의 전문, 도면 외 텍스트 실시예, 경쟁 제품 보강판의 실제 하중 지지 역할이다. 조언은 침해 주장을 유지하되 각도 제한을 독립항에 읽어 넣지 말라는 청구항 차이 논리를 준비하는 것이다.",
            "required_points": ["facts of flat plates", "claim/spec/prosecution comparison", "balanced infringement and noninfringement arguments", "claim differentiation", "missing prior-art distinction text"],
            "forbidden_claims": ["도면이 사선이면 무조건 비침해", "제품 기능만 같으면 문언 침해라고 단정", "extrinsic dictionary supremacy"],
            "inferior_answer": "경쟁 제품의 수직 평판도 하중을 지지하므로 baffle의 사전 의미에 충분히 들어간다. Phillips 이후에도 청구항 문언이 가장 중요하므로 도면의 사선 실시예는 좁은 예시에 불과하다. 심사기록의 load-bearing 표현도 넓은 해석을 뒷받침하므로 비침해 주장은 약하다. 탄환 편향 기능이 약하다는 사실은 종속항 침해에는 영향을 줄 수 있지만 독립항에는 사실상 중요하지 않다.",
            "preference_reason": "열등 답변은 꽤 그럴듯하지만 명세서가 발명을 사선 편향 구조로 반복 정의했을 가능성과 심사기록 전문 확인 필요성을 생략한다."
        },
    ],
    "US-TEVA-SANDOZ-2015": [
        {
            "episode_id": "US-TEVA-SANDOZ-2015-EP1",
            "question": "Copaxone 제조공정 청구항의 'molecular weight' 한정을 미국 항소 관점에서 방어할 수 있는지 평가하라.",
            "scenario_facts": "청구항은 copolymer-1의 분자량을 5-9 kDa로 한정한다. 명세서의 SEC 예는 chromatogram과 calibration curve를 보이지만 Mp, Mn, Mw 중 어느 값을 쓰는지 문언에 직접 쓰지 않는다. 관련 계속출원 심사 중 한 번은 Mw, 다른 한 번은 Mp라는 취지의 답변이 있었다. 하급심은 전문가 증언을 듣고 숙련자가 Mp로 이해한다고 보았다.",
            "reference_answer": "Teva/Sandoz remand의 역사적 범위에서는 두 층을 나누어야 한다. 전문가 증언에 기초한 과학적 전제는 clear error 검토 대상이 될 수 있지만, 청구항이 명세서와 심사기록을 보아 합리적 확실성으로 범위를 알리는지는 법률적 결론이다. 여기서는 같은 'molecular weight'가 계속출원 기록에서 서로 다른 측정법으로 설명된 사실이 핵심 위험이다. 방어는 SEC 예가 Mp를 직접 산출한다는 기술 사실과 오류 진술의 비중이 낮다는 점에 두되, 보정·후속출원에서는 'peak average molecular weight as determined by SEC under [조건]'처럼 측정법을 명시해야 한다. 누락 증거는 당시 SEC 관행, 세 계속출원의 공통 명세서 차이, 전문가가 심사기록을 어떻게 읽는지이다.",
            "required_points": ["clear error and de novo distinction", "Mp/Mn/Mw ambiguity", "prosecution-history inconsistency", "measurement-method drafting fix", "missing expert and file-history evidence"],
            "forbidden_claims": ["전문가 증언이 있으면 불명확성 자동 해소", "항소심이 모든 사실을 de novo로 다시 판단한다고 단정", "측정 조건 없이 수치 범위만 권장"],
            "inferior_answer": "하급심이 전문가를 믿고 Mp라고 판단했다면 항소에서도 그 사실 인정은 존중되어야 한다. SEC chromatogram은 보통 peak average 값을 보여 주므로 청구항에 Mp라고 쓰지 않았더라도 숙련자가 이해할 가능성이 높다. 계속출원 중 Mw라고 답한 부분은 변호사 오류일 수 있고, Teva 판결은 그런 사실판단에 clear error 기준을 요구했다. 따라서 후속 초안에서도 명세서 예를 충분히 넣으면 수치범위 자체를 바꿀 필요는 크지 않다.",
            "preference_reason": "열등 답변은 clear error 보호를 실제보다 넓게 보아, 상충하는 심사기록이 합리적 확실성 결론을 무너뜨릴 수 있다는 Teva remand의 핵심을 약화한다."
        },
        {
            "episode_id": "US-TEVA-SANDOZ-2015-EP2",
            "question": "동일 조성물의 후속 계속출원 2건을 준비하면서 분자량 한정과 심사 답변 전략을 제시하라.",
            "scenario_facts": "첫 출원 청구항은 'average molecular weight of 20-30 kDa'라고만 한다. 연구팀은 GPC-MALS와 SEC calibration 두 방법을 모두 썼다. 한 continuation은 불순물 제거 공정을, 다른 continuation은 투여 안정성을 청구하려 한다. 심사관은 방법별 평균값 차이를 이유로 §112 불명확성을 예고했다.",
            "reference_answer": "미국 Teva/Sandoz 자료를 적용하면 계속출원 사이의 답변 일관성이 초안 품질의 일부가 된다. 방법별 평균값이 달라진다면 각 청구항에 number-average, weight-average, peak-average 중 무엇인지와 시험 조건을 넣고, 명세서에는 두 장비가 산출하는 값의 관계를 설명해야 한다. 한 출원에서 넓은 'average'를 유지하고 다른 출원에서 특정 평균을 주장하면 나중에 같은 용어가 서로 다른 의미였다는 공격을 부를 수 있다. 반론은 숙련자가 장비 이름만으로 평균 종류를 추론한다는 점이나, Teva는 그 추론을 상충 기록이 흔들 수 있음을 보여준다. 필요한 자료는 실제 raw chromatogram, SOP, 심사관 면담 기록, 두 continuation의 claim term matrix이다.",
            "required_points": ["continuation consistency", "specific average type", "test conditions", "prosecution response discipline", "missing lab/SOP evidence"],
            "forbidden_claims": ["continuation마다 용어 의미를 전략적으로 달리해도 무방", "장비명만 쓰면 평균 종류가 항상 고정", "Teva를 현행법으로 단정"],
            "inferior_answer": "심사관이 문제 삼으면 각 continuation에서 해당 발명의 목적에 맞는 평균값을 설명하면 된다. 불순물 제거 공정은 peak-average가 자연스럽고 안정성 출원은 weight-average가 더 관련 있으므로, 같은 'average molecular weight' 표현을 유지하되 명세서 실시예별로 계산법을 달리 설명하는 방법이 유연하다. 나중에 분쟁이 생기면 전문가가 각 문맥의 의미를 설명할 수 있으므로 청구항 문언을 지나치게 길게 만들 필요는 없다.",
            "preference_reason": "열등 답변은 출원별 목적에 맞춘 유연성을 제안해 보이지만, 같은 용어를 상충되게 쓰는 것이 Teva식 불명확성 위험을 키운다는 점을 놓친다."
        },
    ],
    "US-NAUTILUS-2014": [
        {
            "episode_id": "US-NAUTILUS-2014-EP1",
            "question": "운동 심박 센서 청구항의 'spaced relationship' 표현을 미국 명확성 기준으로 검토하라.",
            "scenario_facts": "발명은 손잡이 전극 두 쌍으로 심박 신호를 잡는 운동기구이다. 독립항은 live electrode와 common electrode가 'spaced relationship'에 있다고만 한다. 명세서는 사용자의 양손 접촉과 신호 노이즈 감소를 말하지만 최소 거리, 손 크기 기준, 전극 배열 도면의 축척은 없다. 경쟁사는 전극 간격이 다른 손잡이를 쓴다.",
            "reference_answer": "Nautilus의 역사적 범위에서는 '해석 가능'을 넘어서 숙련자가 명세서와 심사기록을 보고 범위를 합리적 확실성으로 알 수 있는지가 핵심이다. 이 초안은 spaced relationship이 기능적으로는 노이즈 감소와 양손 접촉을 가리키지만, 어느 간격부터 청구범위 안인지 객관 기준이 약하다. 보강안은 전극 쌍 사이의 상대 위치, 최소 절연 거리, 평균 손 폭 또는 신호분리 성능 같은 기준을 명세서에 넣고 필요하면 종속항으로 거리 범위를 둔다. 반론은 전극이 물리적으로 떨어져 있으면 충분하다는 것이나, 분쟁 제품의 다양한 손잡이를 구별해야 하므로 공지 구조만으로 경계가 명확한지 확인해야 한다. 누락 증거는 심사 중 spaced relationship 설명, 업계 전극 배치 관행, 실험 데이터이다.",
            "required_points": ["reasonable certainty standard", "objective boundary problem", "electrode spacing facts", "drafting alternatives", "missing prosecution and technical evidence"],
            "forbidden_claims": ["insolubly ambiguous만 아니면 충분", "절대 정밀성 요구", "모든 상대적 용어 금지"],
            "inferior_answer": "전극들이 서로 붙어 있지 않고 사용자의 양손이 닿도록 배치되었다면 spaced relationship은 보통 기술자가 이해할 수 있다. Nautilus도 절대적 정밀성을 요구하지 않는다고 했으므로 수치 간격을 넣으면 오히려 회피설계를 쉽게 만든다. 명세서에 노이즈 감소 목적과 도면이 있으니 독립항은 넓게 유지하고, 경쟁 제품과의 차이는 침해 분석에서 전극들이 실제로 떨어져 있는지로 다루면 된다.",
            "preference_reason": "열등 답변은 절대 정밀성 불요라는 맞는 말을 경계 기준 생략 허용으로 확장해 합리적 확실성 요구를 약화한다."
        },
        {
            "episode_id": "US-NAUTILUS-2014-EP2",
            "question": "상대방이 'spaced relationship' 불명확성을 주장할 때 미국 민사소송 방어 논리를 구성하라.",
            "scenario_facts": "피고 제품은 전극 사이 절연 홈이 있지만 손잡이가 작아 두 전극을 한 손가락이 동시에 접촉할 수 있다. 특허 명세서에는 신호 검출 회로와 전극 배치 도면이 있고, 발명 목적은 EMG 노이즈와 ECG 신호 분리이다. 심사기록에는 prior art 전극이 너무 가까워 신호 분리가 어렵다는 설명이 있다.",
            "reference_answer": "방어는 'spaced relationship'을 단순 거리 단어로 두지 말고 ECG 신호 취득을 위한 전극 분리라는 기능적 문맥과 연결해야 한다. Nautilus 기준상 명확성은 숙련자 관점, 명세서, 심사기록, 출원 당시를 함께 본다. prior art와 구별하면서 너무 가까운 전극을 배제했다면 그 기록은 피고 제품의 절연 홈만으로 충분한지 판단할 객관 기준이 된다. 다만 기록이 '너무 가까움'만 말하고 시험 조건을 주지 않으면 위험이 남으므로, 전문가 선언에는 손 접촉 조건, 신호분리 임계값, 도면이 알려주는 상대 배열을 포함해야 한다. 반대 주장은 한 손가락 동시 접촉 가능성이 경계 불확실성을 보인다는 점이다.",
            "required_points": ["civil litigation posture", "specification and prosecution history", "POSITA perspective", "defense and residual risk", "specific accused-product fact"],
            "forbidden_claims": ["절연 홈 존재만으로 명확성·침해 확정", "피고 제품 차이를 무시", "1996-style jury framing as final construction"],
            "inferior_answer": "피고 제품에도 절연 홈이 있어 전극이 물리적으로 떨어져 있으므로 spaced relationship은 충분히 명확하다. 심사기록에서 prior art가 너무 가까웠다고 했으니 홈이 있는 제품은 특허의 기준을 만족한다. 한 손가락이 두 전극을 동시에 누를 수 있다는 피고 주장은 사용자의 비정상적 사용 조건에 가깝고, Nautilus도 언어에는 어느 정도 불확실성이 따른다고 보았다. 전문가 증언은 있으면 좋지만 핵심 방어는 도면과 홈의 존재만으로도 가능하다.",
            "preference_reason": "열등 답변은 심사기록을 방어에 쓰지만, '홈 존재'를 객관적 경계로 과잉 단순화해 피고의 접촉 조건 반론과 신호분리 기준을 빠뜨린다."
        },
    ],
    "US-WILLIAMSON-2015": [
        {
            "episode_id": "US-WILLIAMSON-2015-EP1",
            "question": "온라인 강의 플랫폼 특허 초안의 'distributed learning control module'을 §112(f) 위험 관점에서 고쳐라.",
            "scenario_facts": "초안 독립항은 presenter computer, audience computer, streaming data module, distributed learning control module을 둔다. control module은 메시지 수신, 의도된 수신자에게 relaying, streaming module 동기화를 수행한다고만 되어 있다. 명세서 도면은 서버 박스와 화면 예시를 보이나 라우팅 알고리즘은 없다. 개발팀은 module이 소프트웨어 구조라고 주장한다.",
            "reference_answer": "Williamson의 역사적 범위에서는 'module'이 기능을 수행하는 검은상자처럼 쓰이면 means라는 단어가 없어도 §112(f) 적용 위험이 있다. 문제는 단어 하나가 아니라 전체 문구가 세 기능을 recite하면서 충분한 구조를 주는지이다. 초안은 control module을 메시지 큐, session table, recipient resolver, stream coordinator 같은 구조적 구성과 처리 단계로 분해하고, 명세서에 수신-권한확인-대상결정-전송-동기화 흐름을 prose 또는 flow chart로 연결해야 한다. 반론은 분산학습 분야에서 control module이 알려진 구조라는 점이지만, 이를 뒷받침할 당시 기술문헌이나 전문가 증거가 필요하다. 빠진 증거는 prosecution에서 §112(f) 회피 의도, 알고리즘 도면, POSITA의 module 이해이다.",
            "required_points": ["nonce-word risk", "entire limitation analysis", "algorithm or linked structure", "specific software facts", "counterargument from art usage"],
            "forbidden_claims": ["module 사용 자체 금지", "means 단어가 없으면 §112(f) 불가", "화면 예시만으로 알고리즘 충분"],
            "inferior_answer": "distributed learning control이라는 수식어가 붙어 있으므로 단순한 module보다는 구체적인 소프트웨어 구성으로 볼 수 있다. 세 기능도 온라인 강의 서버가 통상 수행하는 작업이어서 숙련자가 구현 방법을 이해할 가능성이 높다. 따라서 청구항에서는 module 표현을 유지하고 명세서에 서버가 presenter와 audience 사이의 통신을 관리한다는 설명을 조금 보강하면 된다. 세부 라우팅 알고리즘은 구현 선택사항이므로 독립항 범위를 불필요하게 좁힐 수 있다.",
            "preference_reason": "열등 답변은 수식어와 통상 구현 가능성을 구조적 의미로 과대평가해 Williamson의 nonce-word 및 대응 알고리즘 요구를 약하게 적용한다."
        },
        {
            "episode_id": "US-WILLIAMSON-2015-EP2",
            "question": "피고가 'learning control module' 한정은 indefinite라고 주장할 때 방어 가능한 기록과 취약점을 정리하라.",
            "scenario_facts": "등록특허 명세서에는 Figure 4 presenter 화면, Figure 5 attendee 화면, 서버-클라이언트 네트워크 블록도가 있다. 청구항은 communications receiving, relaying, coordinating functions를 한 limitation에 묶었다. 침해 제품은 cloud message broker와 WebRTC signaling server를 쓴다. 명세서에는 pseudocode가 없지만 session ID와 audience queue 설명은 있다.",
            "reference_answer": "방어는 먼저 §112(f) 적용 자체를 다투며 'learning control'이 당시 LMS 서버 구조를 지칭했다는 전문가·문헌 증거를 제시할 수 있다. 그러나 Williamson은 module이 generic black box로 쓰이고 기능만 나열되면 presumption이 극복될 수 있다고 보았으므로, 이 방어만으로는 약하다. §112(f)가 적용될 경우 session ID와 audience queue 설명이 세 기능 각각에 명확히 연결되는지 확인해야 한다. receiving은 네트워크 인터페이스와 큐로 설명될 수 있어도 relaying 대상 결정과 streaming coordination 알고리즘이 비면 취약하다. 선택지는 종속항 또는 재발행/계속출원에서 routing table, permission check, synchronization trigger를 구체화하는 것이다.",
            "required_points": ["contest §112(f) and fallback", "three functions separately", "link structure to function", "specific disclosed session/queue facts", "defense limits"],
            "forbidden_claims": ["침해 제품 구조로 명세서 공백 보충", "하나의 서버 블록이 모든 기능의 대응 구조라고 단정", "알고리즘 부재를 구현 용이성으로 대체"],
            "inferior_answer": "명세서의 네트워크 블록도와 presenter/audience 화면은 control module이 어떤 환경에서 작동하는지 충분히 보여 준다. 피고 제품도 message broker와 signaling server를 쓰므로 숙련자는 특허의 module이 그런 서버 로직을 뜻한다고 이해할 것이다. 기능들이 모두 통신 관리의 하위 단계라서 별도 알고리즘을 세 기능마다 요구하는 것은 지나치다. 방어 전략은 §112(f)가 적용되더라도 서버와 큐가 대응 구조라는 점을 강조하는 것이다.",
            "preference_reason": "열등 답변은 침해 제품과 일반 서버 지식을 명세서 대응 구조로 끌어와, 기능별 명확한 연결이라는 Williamson의 취약점을 흐린다."
        },
    ],
    "US-BIOMEDINO-2007": [
        {
            "episode_id": "US-BIOMEDINO-2007-EP1",
            "question": "자동 분석장치 청구항의 'control means'를 미국 §112(f) 관점에서 검토하라.",
            "scenario_facts": "발명은 시료를 cartridge로 이동시키고 detector가 결과를 읽는 자동 assay 장치이다. 청구항은 valve timing과 reagent flow를 제어하는 'control means'를 둔다. 명세서에는 control means라는 박스와 'may be automated'라는 설명만 있고 회로, controller, timing sequence는 없다. 제품팀은 PLC를 쓰면 누구나 구현한다고 말한다.",
            "reference_answer": "Biomedino의 역사적 범위에서는 means 용어가 쓰인 이상 명세서의 대응 구조가 기능과 명확히 연결되어야 한다. control means가 제어 기능을 말하는 데 그치고, 어떤 회로·컴퓨터·타이밍 로직이 valve와 detector를 어떻게 제어하는지 없으면 indefiniteness 위험이 크다. PLC가 흔하다는 사정은 후보 구조를 추론하게 할 뿐, 명세서가 그 구조를 공개하고 연결했다는 증거를 대신하지 못한다. 초안은 controller, valve driver, sensor feedback, timing table을 명세서에 넣고 청구항에서 적어도 'controller configured to execute...' 형태로 구조와 기능을 연결하는 방향이 좋다. 필요한 추가 자료는 도면 원본, control sequence, 당시 assay 장치의 표준 controller 구조이다.",
            "required_points": ["means-plus-function presumption", "corresponding structure linkage", "control/timing facts", "ordinary PLC knowledge limits", "drafting repair"],
            "forbidden_claims": ["자동화 가능성만으로 구조 충분", "제품 구현으로 명세서 보충", "means 표현을 아무 설명 없이 유지"],
            "inferior_answer": "control means는 분석장치 분야에서 제어부를 뜻하고, valve와 detector를 가진 자동 장치라면 PLC나 microcontroller가 자연스럽다. 명세서가 모든 회로를 그리지 않았더라도 숙련자는 reagent flow 제어에 필요한 타이밍을 구현할 수 있다. 따라서 indefiniteness보다는 enablement 쟁점에 가깝고, 초안에서는 control means를 유지하되 실시예 설명에 PLC라는 단어만 추가하면 충분할 수 있다.",
            "preference_reason": "열등 답변은 구현 가능성을 대응 구조 공개와 혼동해 Biomedino가 요구한 명세서-기능 연결을 느슨하게 본다."
        },
        {
            "episode_id": "US-BIOMEDINO-2007-EP2",
            "question": "등록 후 피고가 'control means' 무효를 주장할 때 가능한 방어와 합의 리스크를 제시하라.",
            "scenario_facts": "특허에는 Figure 2에 controller box가 있고 설명문은 'the control means actuates the valves'라고 쓴다. 종속항은 optical detector와 wash reservoir를 추가하지만 controller 구조는 추가하지 않는다. 피고 장치는 microcontroller firmware로 valve를 순차 작동한다. 파일히스토리에는 control means에 관한 별도 논쟁이 없었다.",
            "reference_answer": "방어는 Figure 2의 controller box와 valve actuation 문장을 대응 구조로 주장할 수 있지만, Biomedino식 위험은 그것이 단순 기능명인지 실제 구조인지에 있다. 종속항도 controller 구조를 보강하지 않아 fallback이 약하다. 피고 firmware가 존재한다는 사실은 침해 제품의 구조이지 특허 명세서의 공개가 아니므로 공백을 메우지 못한다. 합의 평가에서는 claim 13-17과 40처럼 핵심 claims가 한 means limitation에 걸려 있다면 무효 리스크 할인을 크게 잡아야 한다. 보완 옵션은 계속출원이나 재발행 가능성이 남아 있는지, 또는 controller가 아닌 detector/wash reservoir 조합 claims로 좁게 집행할 수 있는지 확인하는 것이다.",
            "required_points": ["distinguish accused firmware from disclosed structure", "dependent-claim fallback", "settlement risk", "specific claim-family facts", "continuation/reissue check"],
            "forbidden_claims": ["피고 firmware 때문에 대응 구조 존재", "종속항이 detector를 추가하면 control means 문제가 해결", "무효가 확정이라고 단정"],
            "inferior_answer": "Figure 2의 controller box와 valve actuation 설명은 대응 구조로 주장할 여지가 있다. 피고 장치도 microcontroller firmware로 같은 순서를 수행하므로, 숙련자는 특허의 control means가 그런 전자 제어부를 의미한다고 이해했을 것이다. 종속항의 detector와 wash reservoir는 전체 장치 구조를 더 구체화하므로 독립항보다 방어에 유리하다. 합의에서는 무효 위험을 인정하되 침해 제품의 유사성을 강조해 큰 할인은 피하는 것이 좋다.",
            "preference_reason": "열등 답변은 방어 논거를 갖추고 있지만 피고 제품 구조와 종속항의 다른 요소를 대응 구조 문제 해결 근거로 과대평가한다."
        },
    ],
    "US-ARISTOCRAT-2008": [
        {
            "episode_id": "US-ARISTOCRAT-2008-EP1",
            "question": "슬롯머신 보너스 게임 특허에서 'game control means'를 소프트웨어 알고리즘 요구에 맞게 작성하라.",
            "scenario_facts": "발명은 기본 게임 후 심볼 조합에 따라 보너스 feature game을 시작한다. 청구항은 game control means가 승리 조합을 판단하고 feature를 제어한다고 쓴다. 명세서에는 microprocessor와 memory가 있지만 보너스 선택 절차는 'appropriate programming'으로만 표현되어 있다. 실제 구현에는 random seed, pay table lookup, feature trigger routine이 있다.",
            "reference_answer": "Aristocrat의 역사적 범위에서 컴퓨터 구현 means-plus-function은 범용 컴퓨터 명시만으로 부족하고, 기능을 수행하는 특정 알고리즘이 대응 구조가 된다. 초안은 game control means를 유지하려면 random number generation, symbol evaluation, pay table lookup, trigger decision, feature state update를 흐름도나 prose 단계로 명세서에 넣어야 한다. 'appropriate programming'은 기능 수행 가능성을 말할 뿐 구조를 공개하지 않는다. 반론은 게임기 프로그래밍이 통상 기술이라는 점이나, §112(f)의 quid pro quo상 공개된 알고리즘으로 범위가 정해지는 점이 더 중요하다. 필요한 자료는 실제 firmware spec, pay table 예, feature trigger 조건이다.",
            "required_points": ["computer MPF algorithm rule", "specific game-control steps", "general-purpose computer insufficiency", "implementation evidence", "drafting fix"],
            "forbidden_claims": ["소스코드 필수라고 단정", "microprocessor 단어만으로 충분", "보너스 기능 결과만 기재"],
            "inferior_answer": "슬롯머신의 game control은 업계에서 microprocessor와 pay table이 수행하는 표준 기능이다. 명세서에 microprocessor와 memory가 있고 feature game 결과가 설명되어 있으면 숙련자는 적절한 programming을 만들 수 있다. Aristocrat이 알고리즘을 요구하더라도 random number와 pay table은 너무 기본적이어서 자세히 쓰면 회피 설계를 허용한다. 독립항은 means 표현을 유지하고, 종속항에 feature trigger 조건 일부만 넣는 절충이 낫다.",
            "preference_reason": "열등 답변은 업계 표준성을 들어 알고리즘 공개 요구를 축소하고, 'appropriate programming'을 구조처럼 취급한다."
        },
        {
            "episode_id": "US-ARISTOCRAT-2008-EP2",
            "question": "침해소송에서 피고가 알고리즘 부재를 들어 summary judgment를 구할 때 대응 의견을 작성하라.",
            "scenario_facts": "특허는 game control means와 display means를 포함한다. 명세서 표 1은 winning combinations를 나열하지만, feature game 시작 여부를 판정하는 단계는 없다. 피고는 서버 기반 카지노 플랫폼으로 같은 bonus outcome을 제공한다. 의뢰인은 표 1 자체가 알고리즘이라고 주장하고 싶어 한다.",
            "reference_answer": "표 1이 단순 winning combinations 목록인지, 기능을 수행하는 단계적 절차인지가 핵심이다. Aristocrat에 따르면 §112(f) 구조는 컴퓨터가 어떤 방식으로 claimed function을 수행하는지 알려야 하므로, 표가 입력-판정-출력 절차를 충분히 제공하지 않으면 알고리즘으로 보기 어렵다. 방어는 표 1이 symbol comparison과 trigger decision을 암시한다는 전문가 의견, 명세서의 processor 설명, claim function의 단순성을 결합하는 방식으로 제한적으로 가능하다. 그러나 서버 기반 피고 플랫폼의 구현은 특허의 공개 부족을 고칠 수 없다. 합리적 선택지는 summary judgment 저지를 위해 기능별 알고리즘 후보를 구체화하되, 합의 가치에는 indefiniteness 위험을 반영하는 것이다.",
            "required_points": ["table versus algorithm distinction", "summary judgment posture", "accused server not source support", "function-by-function analysis", "risk-adjusted option"],
            "forbidden_claims": ["피고가 같은 결과를 내므로 알고리즘 충분", "표 1을 무조건 알고리즘으로 취급", "Aristocrat을 단순 소스코드 요구로 설명"],
            "inferior_answer": "표 1은 winning combinations를 특정하고 있으므로 game control means가 비교해야 할 조건을 제공한다. 보너스 시작은 그 표와 processor가 결합하면 자연스럽게 수행되는 간단한 판단이다. 피고 서버도 같은 조건표를 쓸 가능성이 높아 침해 대비가 쉽다. 따라서 summary judgment에는 표 1이 알고리즘의 실질 부분이고 나머지는 통상적인 프로그래밍이라는 점을 강하게 주장하면 충분하다.",
            "preference_reason": "열등 답변은 표가 조건 목록이라는 강점을 말하지만, claimed function 전체를 수행하는 절차 공개인지라는 Aristocrat식 질문을 충분히 분리하지 않는다."
        },
    ],
    "US-NOAH-2012": [
        {
            "episode_id": "US-NOAH-2012-EP1",
            "question": "회계 네트워크 특허의 'access means'가 여러 기능을 담을 때 명세서 보강안을 제시하라.",
            "scenario_facts": "청구항은 access means가 사용자 인증, 금융기관 연결, 거래 수행 가능화, account summary 표시를 맡는다고 쓴다. 명세서에는 passcode 확인 절차는 있으나 금융기관별 연결 프로토콜과 거래 enable 단계는 없다. 출원인은 모든 기능이 로그인 flow의 일부라고 본다. 경쟁 서비스는 OAuth token과 bank API를 쓴다.",
            "reference_answer": "Noah Systems의 역사적 범위에서는 컴퓨터 구현 means limitation이 여러 식별 가능한 기능을 recite하면, 명세서가 일부 기능의 알고리즘만 주는 경우에도 전체 limitation이 취약해질 수 있다. passcode 확인은 인증 기능을 뒷받침할 수 있지만, bank API 연결과 거래 수행 가능화는 별도 기능으로 보인다. 보강안은 기능별로 login credential validation, institution selection, protocol negotiation, transaction authorization, account display generation 단계를 분리하고 각 단계의 입력·출력을 명세서에 둔다. 반론은 모든 기능이 access라는 하나의 목적 아래 있다는 점이나, Noah는 identifiable functions별 분석을 요구한다. 누락 증거는 출원 당시 online banking 표준, 실제 flow chart, claim function 분해표이다.",
            "required_points": ["multiple identifiable functions", "partial algorithm insufficiency", "specific accounting-network functions", "function-by-function drafting", "missing standards and flow chart"],
            "forbidden_claims": ["로그인 알고리즘 하나로 모든 access 기능 충분", "accused OAuth 구조로 보충", "2012년 당시 기록에 없는 사후 절차 삽입"],
            "inferior_answer": "access means의 중심은 사용자가 시스템에 접근하도록 하는 것이므로 passcode 확인 절차가 공개되어 있으면 주요 알고리즘은 있다. 금융기관 연결이나 account summary 표시는 접근이 허용된 뒤 통상 프로그램이 처리하는 부수 기능이다. Noah의 알고리즘 요구를 의식해 명세서에 online banking standards를 참조하고, 종속항에서 bank API를 예시하면 충분히 방어 가능하다.",
            "preference_reason": "열등 답변은 기능들을 하나의 access 목적에 묶어 일부 알고리즘 공개를 전체 기능 지원으로 확장하는 미묘한 오류가 있다."
        },
        {
            "episode_id": "US-NOAH-2012-EP2",
            "question": "기존 등록특허의 'access means' 무효 공격에 대비한 미국 소송 답변 전략을 작성하라.",
            "scenario_facts": "명세서는 user file, passcode, access number, financial accounting computer를 설명한다. 청구항 function은 account file에 접근하고 financial transaction을 가능하게 하는 두 문구를 포함한다. 피고는 후자에 대한 알고리즘이 없다고 한다. 의뢰인은 special master 단계에서 passcode flow만 상세히 주장했다.",
            "reference_answer": "소송 답변은 passcode flow가 어떤 function에 대응하는지 먼저 인정하고, financial transaction enablement function에 대응하는 별도 disclosure가 있는지 찾아야 한다. Noah는 일부 기능만 수행하는 알고리즘이 있으면 나머지 기능에 대해 no algorithm처럼 다룰 수 있다고 보았다. 따라서 기존 special master 기록이 인증 기능에 치우쳤다면 취약하다. 가능한 방어는 account file access와 transaction enablement가 claim 문맥상 하나의 function이라는 해석, 또는 명세서의 access number/financial accounting computer 설명이 두 기능을 모두 수행한다는 전문가 해석이다. 그러나 누락된 절차를 전문가가 새로 쓰는 것은 위험하므로 합의·청구항 포기·continuation fallback을 함께 검토한다.",
            "required_points": ["separate functions in litigation record", "special master history", "expert cannot rewrite spec", "defense and fallback", "historical civil procedure only"],
            "forbidden_claims": ["전문가가 알고리즘을 새로 설명하면 충분", "일부 알고리즘 공개가 전체 limitation을 자동 구제", "역사적 범위 밖 절차 옵션 제시"],
            "inferior_answer": "access number와 financial accounting computer가 명세서에 있으므로 거래 가능화 기능도 인증 flow와 연결된다고 주장할 수 있다. special master 단계에서 passcode flow를 중점적으로 다뤘더라도, 그 flow가 완료되면 사용자는 자연스럽게 거래 화면에 접근한다. 전문가가 당시 금융거래 시스템의 통상 flow를 설명하면 명세서의 간단한 구조를 보완할 수 있다. 따라서 무효 공격은 방어 가능성이 상당하다.",
            "preference_reason": "열등 답변은 통상 flow 설명을 명세서의 알고리즘 공개처럼 쓰며, 전문가가 specification을 다시 쓸 수 없다는 Noah의 제한을 약화한다."
        },
    ],
    "US-INTERVAL-2014": [
        {
            "episode_id": "US-INTERVAL-2014-EP1",
            "question": "'unobtrusive manner'로 정의된 알림 표시 청구항을 미국 명확성 기준에 맞게 고쳐라.",
            "scenario_facts": "발명은 사용자의 주변시야에 뉴스와 광고 이미지를 표시하는 attention manager이다. 독립항은 이미지를 'in an unobtrusive manner that does not distract a user'로 표시한다고 한다. 명세서는 화면 가장자리, 일정 시간 간격, fading 예를 들지만 사용자가 distract되는 기준은 수치화하지 않는다. 제품은 사용자의 작업창 위에 반투명 배너를 띄운다.",
            "reference_answer": "Interval Licensing의 역사적 범위에서 이 표현은 주관적 UX 평가가 claim boundary가 되는 위험을 보인다. '방해하지 않음'을 유지하려면 화면 영역 비율, 표시 지속시간, opacity, 입력 포커스 비차단, 사용자 활동 중 표시 빈도 같은 객관 기준으로 바꿔야 한다. 명세서의 edge display와 fading 예는 좋은 출발점이지만, 어떤 배너가 claim 안인지 밖인지 숙련자가 구별할 수 있어야 한다. 반론은 사용자 인터페이스 분야에서 unobtrusive가 알려진 설계 개념이라는 점이나, 사건은 그런 개념만으로 의미 있게 정밀한 범위가 되지 않을 수 있음을 보여준다. 필요한 자료는 usability test 기준, UI guideline, prosecution에서 distract를 설명한 문장이다.",
            "required_points": ["subjective term risk", "objective UI boundaries", "specific display facts", "spec examples as incomplete anchors", "missing usability/prosecution evidence"],
            "forbidden_claims": ["UX 형용사 전면 금지", "사용자가 덜 불편하면 명확", "판례 결과만으로 모든 attention-manager claim 무효"],
            "inferior_answer": "unobtrusive는 UI 분야에서 화면 작업을 방해하지 않는 표시 방식을 뜻하는 널리 쓰이는 말이다. 명세서에 가장자리 표시와 fading 예가 있으므로 숙련자는 중앙 팝업과 주변 표시를 구별할 수 있다. 청구항에 opacity나 시간 값을 넣으면 제품 변형마다 회피 가능성이 커진다. 따라서 독립항은 unobtrusive manner를 유지하고 종속항에서 edge display나 fade-in을 예시하는 정도가 적절하다.",
            "preference_reason": "열등 답변은 업계 UX 개념과 실시예를 근거로 삼지만, 사용자별 주관성을 객관적 claim boundary로 바꾸는 작업을 충분히 하지 않는다."
        },
        {
            "episode_id": "US-INTERVAL-2014-EP2",
            "question": "피고 광고 배너가 'unobtrusive manner'를 충족하지 않는다는 비침해 주장에 어떻게 대응할지 평가하라.",
            "scenario_facts": "피고 배너는 화면 오른쪽 15% 영역에 8초간 나타나고 사용자의 입력 포커스를 빼앗지 않는다. 그러나 동영상 재생 중에는 자동으로 표시되어 일부 사용자가 주의를 빼앗겼다고 보고했다. 명세서 예는 peripheral attention을 engage한다고 하면서도 작업 방해는 피한다고 설명한다. 청구항에는 사용자 반응 측정법이 없다.",
            "reference_answer": "침해 주장을 하려면 'unobtrusive manner'가 불명확성 공격을 견딜 수 있는 객관 기준으로 먼저 해석되어야 한다. 오른쪽 15%, 8초, 포커스 비차단은 방어에 유리하지만, 동영상 중 표시와 사용자 불만은 distract 경계가 불안정함을 보여준다. Interval식 위험은 좋은 UI인지가 아니라 claim이 어느 표시를 포함하는지 알려 주는지이다. 대응은 명세서의 peripheral attention 목적을 근거로, 포커스 비차단과 주변 영역이라는 구조적 기준을 제안하되 사용자 설문만으로 claim scope를 정하지 않는다. 누락 자료는 원고가 심사 중 distract를 어떻게 설명했는지, 당시 UI 표준, 피고 배너의 실제 위치·빈도 로그이다.",
            "required_points": ["infringement depends on defensible construction", "specific banner facts", "objective versus user-reaction evidence", "counterargument from complaints", "missing prosecution/log evidence"],
            "forbidden_claims": ["사용자 불만이 있으면 자동 비침해", "오른쪽 배너면 자동 침해", "subjective term을 선호도 조사로만 해석"],
            "inferior_answer": "피고 배너는 화면 오른쪽에 있고 입력 포커스를 빼앗지 않으므로 unobtrusive에 해당할 가능성이 높다. 일부 사용자가 주의를 빼앗겼다고 느낀 점은 주관적 반응이고, 특허는 peripheral attention을 의도하므로 어느 정도 시선을 끄는 것은 발명의 기능이다. 따라서 침해 주장은 오른쪽 영역과 8초 표시라는 사실을 중심으로 제기하면 된다. 불명확성 위험은 있지만 명세서 예와 상당히 가까운 구현이므로 크지 않다.",
            "preference_reason": "열등 답변은 peripheral attention 논리를 잘 잡지만, 그 자체가 'distract'의 객관 경계를 세우지 못하면 무효 위험이 선행된다는 점을 충분히 반영하지 않는다."
        },
    ],
    "US-DATAMIZE-2005": [
        {
            "episode_id": "US-DATAMIZE-2005-EP1",
            "question": "소프트웨어 UI 특허 초안의 'aesthetically pleasing' 한정을 대체할 미국식 청구항 언어를 제안하라.",
            "scenario_facts": "발명은 kiosk 화면 authoring tool이다. 초안은 interface screen이 'aesthetically pleasing look and feel'을 갖도록 자동 배치한다고 한다. 명세서는 color palette, font size, alignment examples를 보여 주지만 점수화 기준은 없다. 고객은 경쟁사가 보기 좋은 template generator를 따라 한다고 걱정한다.",
            "reference_answer": "Datamize의 역사적 범위에서는 순수 미적 만족 같은 주관 표현이 객관 기준 없이 claim boundary가 되면 indefinite 위험이 크다. 이 초안은 '보기 좋음' 대신 palette contrast threshold, grid alignment, font hierarchy rule, overlapping prohibition, preview approval workflow처럼 도구가 수행하는 구체 규칙으로 바꾸는 편이 낫다. 명세서 예는 각 규칙이 왜 시각 품질을 만드는지 설명하는 근거가 될 수 있지만, 사용자 취향을 청구항 경계로 삼으면 취약하다. 반론은 UI 디자이너라면 aesthetic quality를 이해한다는 점이나, 사건은 사람마다 달라지는 판단을 객관화해야 함을 시사한다. 필요한 자료는 실제 layout algorithm, style rule table, designer guideline이다.",
            "required_points": ["subjective aesthetic term", "objective UI rules", "software-authoring facts", "spec examples role", "missing algorithm/guideline evidence"],
            "forbidden_claims": ["미적 표현은 언제나 허용", "미적 표현은 언제나 금지", "사용자 선호도만으로 claim boundary 확정"],
            "inferior_answer": "aesthetically pleasing은 UI 디자인 분야에서 충분히 이해되는 목표이고 명세서가 색상, 글꼴, 정렬 예를 제공하므로 완전히 주관적이라고 보기는 어렵다. 청구항에 세부 수치를 넣으면 디자인 변형을 놓칠 수 있다. 따라서 독립항은 aesthetically pleasing을 유지하고, 종속항에 contrast나 alignment examples를 넣어 해석 자료를 제공하는 방법이 균형적이다.",
            "preference_reason": "열등 답변은 실시예를 해석 자료로 쓰는 점은 타당하지만, 독립항의 주관 표현을 객관 규칙으로 치환하지 않아 Datamize의 핵심 위험을 남긴다."
        },
        {
            "episode_id": "US-DATAMIZE-2005-EP2",
            "question": "기존 UI 특허 집행 전에 'aesthetically pleasing' limitation의 무효 리스크를 settlement valuation에 반영하라.",
            "scenario_facts": "등록특허는 kiosk authoring interface의 모든 asserted claims에 aesthetically pleasing limitation을 포함한다. 명세서에는 preferred screen shots와 designer가 조정 가능한 parameters가 있다. 피고 제품은 자동 template ranking을 쓰고, 사용자 평점 데이터가 높다. 원고는 평점이 미적 만족의 객관 증거라고 본다.",
            "reference_answer": "Datamize를 적용하면 사용자 평점이 높다는 사실은 피고 제품이 좋게 보인다는 증거일 수 있지만, claim term 자체가 객관적 경계를 갖는다는 증거는 아니다. valuation에서는 모든 asserted claims가 같은 limitation에 걸려 있으면 validity discount를 크게 반영해야 한다. 방어 가능성은 명세서의 parameters가 실제로 어떤 screen이 aesthetically pleasing인지 아닌지 결정하는 객관 rules로 기능하는지에 달려 있다. 피고의 template ranking은 침해 유사성에는 도움이 되지만 특허 공개의 명확성 공백을 보완하지 못한다. 필요한 자료는 claim construction record, expert design criteria, prosecution에서 aesthetic term을 설명한 답변이다.",
            "required_points": ["settlement valuation", "all asserted claims share term", "ratings not claim boundary", "parameters as possible objective criteria", "accused product distinction"],
            "forbidden_claims": ["사용자 평점이 높으면 limitation 충족", "피고 알고리즘으로 특허 명확성 보충", "무효 리스크 없이 집행"],
            "inferior_answer": "피고 제품의 사용자 평점과 template ranking은 aesthetically pleasing 기능을 실제로 수행한다는 강한 정황이다. 명세서도 designer-adjustable parameters를 제공하므로 숙련자는 어떤 요소가 미적 품질에 영향을 주는지 알 수 있다. 무효 리스크는 존재하지만, 피고가 같은 문제를 자동 ranking으로 해결했다는 점을 보여 주면 배심과 법원 모두에게 설득력이 있다. 합의에서는 통상 소프트웨어 특허 리스크 정도만 할인하면 된다.",
            "preference_reason": "열등 답변은 침해 유사성과 명확성 기준을 섞어, 피고의 평점·ranking이 특허 청구범위의 객관 경계를 만든다고 오해한다."
        },
    ],
    "US-HALLIBURTON-2008": [
        {
            "episode_id": "US-HALLIBURTON-2008-EP1",
            "question": "시추 유체 청구항의 'fragile gel' 표현을 미국 명확성·실시가능성 관점에서 보강하라.",
            "scenario_facts": "발명은 wellbore에서 cuttings를 운반하는 drilling fluid이다. 청구항은 fluid가 fragile gel이라고 하고, stress를 받으면 액체처럼 되고 stress가 제거되면 빠르게 gel로 돌아온다고 설명한다. 명세서에는 몇 가지 rheology curve가 있으나 shear rate, recovery time, temperature 기준이 일관되지 않다. 경쟁 유체는 유사한 회복 거동을 보인다.",
            "reference_answer": "Halliburton의 역사적 범위에서 물성 기능어는 경계 기준이 있어야 한다. fragile gel이라는 설명은 작동 원리를 말하지만, 어떤 shear 조건에서 얼마나 빨리 회복해야 claim 안인지 불분명하면 침해와 validity 모두 위험하다. 초안 보강은 shear rate, viscosity drop percentage, recovery modulus, test temperature, measurement instrument를 명세서와 종속항에 넣는 것이다. 반론은 drilling fluid 업계가 fragile gel을 이해한다는 점이나, 사건의 교훈은 업계 감각이 경계선을 대신하지 못할 수 있다는 데 있다. 필요한 자료는 lab protocol, replicate data, prior art gels의 비교표이다.",
            "required_points": ["material-property boundary", "test conditions", "fragile gel facts", "drafting measurements", "prior art comparison"],
            "forbidden_claims": ["물성 기능어 전면 금지", "업계 감각만으로 충분", "침해 제품 유사성으로 명확성 보충"],
            "inferior_answer": "fragile gel은 stress를 받으면 얇아지고 다시 gel이 되는 유체를 뜻하므로 시추 유체 분야에서 이해 가능한 표현이다. 명세서의 rheology curve가 그 거동을 보여 주고 경쟁 유체도 유사하게 작동한다면 침해 주장에는 충분한 기반이 있다. 수치 조건을 넣으면 다양한 well temperature와 shear 환경을 놓칠 수 있으므로, 독립항은 기능 표현을 유지하고 실험 예를 보강하는 정도가 낫다.",
            "preference_reason": "열등 답변은 현장 조건 다양성을 고려하지만, 그 다양성 때문에 오히려 객관 시험조건이 필요하다는 Halliburton식 문제를 놓친다."
        },
        {
            "episode_id": "US-HALLIBURTON-2008-EP2",
            "question": "피고가 'fragile gel'이 불명확하다고 summary judgment를 구한 상황에서 반박과 약점을 제시하라.",
            "scenario_facts": "특허 명세서는 fragile gel을 쉽게 disrupted되고 stress 제거 후 quickly returns라고 정의한다. 원고 전문가 실험은 120°F, 특정 shear protocol에서 30초 회복을 보였다. 피고 실험은 180°F에서 회복 시간이 5분 이상이라고 한다. 청구항은 온도나 회복 임계값을 쓰지 않는다.",
            "reference_answer": "반박은 명세서 정의와 원고 실험을 묶어 POSITA가 fragile gel을 시험 가능한 물성으로 이해했다는 점에서 시작한다. 그러나 Halliburton 위험은 quickly returns가 어느 온도·shear history에서 얼마를 뜻하는지 claim이 알려 주는가이다. 120°F와 180°F 결과가 크게 다르면 피고의 불명확성 주장은 강해진다. 원고는 well conditions에서 통상 사용하는 API rheology protocol이 있었는지, 명세서 curve가 그 protocol을 암시하는지, prior art와 비교해 fragile/non-fragile을 나눌 수 있는지 제시해야 한다. 약하면 합의 또는 narrow construction으로 특정 test condition을 주장하는 방안이 현실적이다.",
            "required_points": ["summary judgment posture", "conflicting test protocols", "definition versus boundary", "industry protocol evidence", "narrow construction option"],
            "forbidden_claims": ["원고 실험 하나로 명확성 확정", "피고 고온 실험은 무조건 무관", "quickly를 임의 숫자로 단정"],
            "inferior_answer": "명세서가 fragile gel을 정의했고 원고 전문가가 실제 회복 데이터를 제시했으므로 summary judgment는 부적절하다. 온도 차이는 침해 여부나 실험 신뢰도의 문제이지 claim term의 의미 자체를 없애지는 않는다. Halliburton도 어려운 claim construction만으로 indefiniteness를 인정하지는 않는다. 따라서 원고는 120°F protocol이 실제 well 환경을 반영한다고 주장하고, 피고의 180°F 실험은 비대표적 조건이라고 공격하면 된다.",
            "preference_reason": "열등 답변은 summary judgment 방어 논리는 갖지만, 상충 실험이 term boundary의 객관성 문제로 이어질 수 있음을 과소평가한다."
        },
    ],
    "US-ARIAD-2010": [
        {
            "episode_id": "US-ARIAD-2010-EP1",
            "question": "NF-kB 활성을 낮추는 넓은 생명공학 청구항의 written description 위험을 평가하라.",
            "scenario_facts": "연구팀은 특정 세포에서 NF-kB pathway를 억제하는 스크리닝 결과를 얻었다. 초안은 'reducing NF-kB activity'를 달성하는 모든 compound class를 청구하려 한다. 명세서에는 세 가지 후보 물질과 pathway diagram이 있지만 공통 구조나 작동 물질의 대표 범위는 제한적이다. 경쟁사 약물은 다른 binding site로 같은 downstream 효과를 낸다.",
            "reference_answer": "Ariad en banc의 역사적 범위에서 written description은 enablement와 별도로 출원 시 발명자가 청구 발명을 보유했는지를 본다. pathway 목표와 몇 개 후보가 있다는 사실은 연구 방향을 보여 주지만, 모든 NF-kB activity reduction compound를 보유했다는 증거와는 다르다. 넓은 기능적 genus를 원하면 대표 species 수, 공통 구조 또는 공통 작동 특징, active/inactive examples, selection criteria를 명세서에 넣어야 한다. 반론은 pathway 자체가 발명의 본질이라는 점이나, Ariad는 결과로 genus를 정의하는 설명만으로 부족할 수 있음을 보여준다. 필요한 자료는 후보군 구조, assay reproducibility, 경쟁 약물과의 mechanistic overlap이다.",
            "required_points": ["separate written description requirement", "possession at filing", "functional genus problem", "representative species/common features", "missing assay and structure data"],
            "forbidden_claims": ["enablement 가능하면 written description도 자동 충족", "연구목표가 곧 genus possession", "경쟁 약물 효과만으로 보유 증명"],
            "inferior_answer": "명세서가 NF-kB pathway와 세 가지 후보 물질을 설명하고 있으므로 발명자는 activity reduction의 핵심 원리를 파악한 것으로 볼 수 있다. 경쟁 약물이 다른 binding site로 같은 downstream 효과를 낸다는 점은 오히려 pathway 억제라는 발명 개념의 폭을 보여 준다. written description 위험은 있지만, 충분한 assay 설명과 후보 물질 예가 있으면 enablement와 함께 방어 가능할 것이다.",
            "preference_reason": "열등 답변은 pathway 원리 파악을 genus 보유로 넓혀, Ariad가 분리한 written description의 possession 요구를 흐린다."
        },
        {
            "episode_id": "US-ARIAD-2010-EP2",
            "question": "경쟁사에 경고장을 보내기 전, 기능적 biotech genus claim의 집행 리스크를 정리하라.",
            "scenario_facts": "등록청구항은 세포 내 NF-kB 활성을 감소시키는 방법을 넓게 포함한다. 특허 명세서에는 decoy molecule과 dominant-negative inhibitor 예가 있으나 경쟁사 small molecule은 다른 upstream target을 친다. 파일히스토리에는 broad mechanism을 강조한 발언이 있다. 의뢰인은 침해 주장을 통해 라이선스 협상을 시작하려 한다.",
            "reference_answer": "집행 전에는 침해 폭을 넓게 잡는 순간 written description 공격도 넓어진다는 점을 명확히 해야 한다. Ariad는 원래 청구항에도 별도 written description 요구가 적용되고, specification이 실제 보유를 객관적으로 보여야 한다고 보았다. 경쟁사 small molecule을 포함시키려면 명세서의 예들이 그 class까지 대표하거나 공통 구조·mechanism을 제공하는지 필요하다. broad mechanism 발언은 침해에는 유리할 수 있으나 possession 부족 공격에는 불리할 수 있다. 옵션은 경고장을 좁은 claim chart 중심으로 보내되, 라이선스 협상에서는 validity challenge 비용을 반영하고, continuation이 가능하면 small molecule species와 screening criteria를 보강하는 것이다.",
            "required_points": ["enforcement risk", "broad infringement versus WD exposure", "competitor small-molecule facts", "file-history effect", "practical licensing option"],
            "forbidden_claims": ["넓은 mechanism 발언은 항상 유리", "기능 같으면 written description 문제 없음", "경고장 단계에서 validity risk 무시"],
            "inferior_answer": "경쟁사 small molecule도 NF-kB 활성을 낮추므로 문언상 침해 주장은 가능하다. 명세서에 여러 억제 방식이 있고 파일히스토리도 broad mechanism을 강조했으므로, 특허권자는 pathway-level 발명을 보유했다고 주장할 수 있다. written description 공격은 예상되지만 라이선스 협상에서는 침해 가능성과 소송 비용을 강조하는 것이 더 중요하다. 경고장에는 Ariad 위험을 자세히 드러내지 않는 편이 낫다.",
            "preference_reason": "열등 답변은 협상 현실을 고려하지만, 넓은 mechanism 발언이 possession 쟁점에서 역효과를 낼 수 있다는 핵심 연결을 빠뜨린다."
        },
    ],
    "US-LIZARDTECH-2005": [
        {
            "episode_id": "US-LIZARDTECH-2005-EP1",
            "question": "이미지 압축 발명의 seamless DWT claim을 broad genus로 확장할 수 있는지 검토하라.",
            "scenario_facts": "명세서는 discrete wavelet transform에서 tile boundary artifact를 줄이는 한 가지 seamless method를 자세히 설명한다. 새 청구항 초안은 'maintaining updated sums' 없는 모든 seamless DWT 방식을 포함하려 한다. 연구팀은 대체 방식 두 개를 아이디어 수준으로 알고 있지만 구현 데이터는 없다. 경쟁 제품은 다른 boundary handling을 쓴다.",
            "reference_answer": "LizardTech의 역사적 범위에서는 한 방법을 상세히 disclose했다고 해서 그 결과를 달성하는 모든 generic claim을 보유한 것은 아니다. seamless DWT라는 목표가 넓다면, updated sums 방식 외 대체 boundary handling의 공통 원리나 대표 구현을 명세서에 넣어야 한다. 경쟁 제품을 포착하려는 넓은 claim은 written description 위험을 키운다. 반론은 발명의 기여가 artifact 없는 DWT 결과라는 점이지만, 사건은 결과만으로 넓은 genus를 뒷받침하기 어렵다는 신호를 준다. 필요한 자료는 대체 방식 구현, artifact metrics, updated sums가 필수인지 여부에 관한 기술 설명이다.",
            "required_points": ["one disclosed embodiment versus generic claim", "seamless DWT facts", "written description overbreadth", "alternative implementation evidence", "competitor capture risk"],
            "forbidden_claims": ["결과가 같으면 모든 방식 보유", "대표 구현 하나는 언제나 부족", "경쟁 제품을 근거로 명세서 보충"],
            "inferior_answer": "명세서가 seamless DWT를 자세히 설명하고 그 목적과 장점을 알려 주므로, 숙련자는 updated sums 외에도 다른 boundary handling을 적용할 수 있다. 넓은 claim은 경쟁 제품의 설계변경을 막기 위해 필요하다. 대체 방식 두 개가 아이디어 수준이라도 발명의 핵심은 tile artifact를 줄이는 원리이므로, 초안에서 updated sums를 독립항 필수요소로 두면 보호가 지나치게 좁아질 수 있다.",
            "preference_reason": "열등 답변은 보호범위 필요성을 잘 보지만, 아이디어 수준 대체 방식이 written description possession을 채운다는 추론이 약하다."
        },
        {
            "episode_id": "US-LIZARDTECH-2005-EP2",
            "question": "피고의 다른 boundary algorithm을 상대로 broad seamless claim을 집행할지 결정하라.",
            "scenario_facts": "등록특허의 원 claim은 updated sums를 포함하지만 continuation claim은 그 요소 없이 seamless DWT를 넓게 쓴다. 피고 알고리즘은 overlapping tiles와 post-filtering으로 boundary artifact를 줄인다. 명세서는 updated sums 방식의 수학적 절차만 완전하게 설명한다. 원고 전문가는 두 방식이 같은 visual result를 낸다고 한다.",
            "reference_answer": "집행 판단은 visual result 유사성과 written description support를 분리해야 한다. LizardTech식 위험은 continuation의 broad claim이 updated sums 없는 피고 방식을 포함할 만큼 명세서가 전체 genus를 보유했는지이다. overlapping tiles와 post-filtering이 명세서의 공통 원리에서 예측 가능한 변형인지 증거가 없으면 validity discount가 크다. 침해 주장은 결과 유사성으로 시작할 수 있지만, 무효 방어는 '발명자가 그 방식까지 보유했는가'로 돌아온다. 필요한 자료는 continuation 심사기록, 대체 algorithm의 당시 알려진 상태, expert가 같은 원리라고 보는 기술 근거이다.",
            "required_points": ["infringement-result versus WD support", "continuation broadening", "updated sums omission", "accused alternative algorithm", "missing prosecution/technical evidence"],
            "forbidden_claims": ["visual result 같으면 written description 충분", "continuation claim은 원 명세서보다 넓어도 자동 유효", "피고 알고리즘으로 발명자 보유 증명"],
            "inferior_answer": "피고 방식도 seamless DWT 결과를 내고 tile artifact를 줄이므로 broad continuation claim에 포함될 가능성이 있다. 원 명세서의 updated sums는 preferred implementation이고, continuation에서 이를 뺀 것은 발명의 결과 중심 보호를 확보하려는 합리적 선택이다. 전문가가 두 방식의 visual result가 같다고 설명한다면 written description 공격도 극복 가능하다. 집행 가치는 충분히 있다.",
            "preference_reason": "열등 답변은 continuation broadening을 자연스럽게 설명하지만, 결과 동일성을 발명자 보유 증거로 과잉 사용하는 실수를 한다."
        },
    ],
    "US-LIEBEL-2007": [
        {
            "episode_id": "US-LIEBEL-2007-EP1",
            "question": "pressure jacket 없는 injector까지 포괄하려는 미국 청구항의 enablement 위험을 평가하라.",
            "scenario_facts": "기존 명세서는 power injector와 disposable syringe를 설명하면서 pressure jacket을 안전 구조로 반복 언급한다. 후속 청구항 초안은 jacket 없는 front-loading injector도 포함하도록 pressure jacket을 삭제했다. 개발팀은 jacketless prototype을 아직 완성하지 못했다. 경쟁사는 jacketless injector를 판매하기 시작했다.",
            "reference_answer": "Liebel-Flarsheim의 역사적 범위에서는 넓은 claim construction을 얻어 jacket 없는 제품까지 포함하면, 그 full scope를 enable해야 한다. 명세서가 pressure jacket 있는 시스템만 가르치고 jacketless 작동을 오히려 어렵거나 위험하게 설명한다면 넓은 청구항은 침해 폭을 얻는 대신 enablement 취약성을 키운다. 초안은 jacketless embodiment의 syringe strength, loading mechanism, pressure tolerance, failure mode를 실제로 기재하거나, 독립항을 jacket 포함형으로 두고 jacketless는 개발 후 별도 출원해야 한다. 반론은 숙련자가 pressure vessel 설계를 알고 있다는 점이지만 prototype 부재와 명세서의 teaching away가 중요하다.",
            "required_points": ["full scope enablement", "pressure jacket deletion", "prototype absence", "teaching-away risk", "drafting choice"],
            "forbidden_claims": ["넓은 해석은 enablement에 영향 없음", "경쟁제품 존재로 enablement 충족", "wrong Liebel patent list"],
            "inferior_answer": "pressure jacket을 삭제해야 경쟁사의 jacketless injector를 포착할 수 있으므로 넓은 독립항은 전략적으로 필요하다. 명세서가 injector 전체 구조를 설명하고 있고 숙련자는 고압 syringe 설계를 알고 있으므로 jacket 없는 변형도 routine engineering으로 볼 여지가 있다. jacketless prototype이 없더라도 경쟁 제품이 시장에 있다면 실시 가능성은 어느 정도 뒷받침된다. 종속항에 jacket 포함형을 남기면 fallback도 확보된다.",
            "preference_reason": "열등 답변은 사업상 포착 필요성을 잘 반영하지만, 경쟁 제품 존재와 routine engineering을 명세서의 full-scope enablement 공백 보충으로 과대평가한다."
        },
        {
            "episode_id": "US-LIEBEL-2007-EP2",
            "question": "jacketless 피고 제품을 상대로 소송을 계속할지, 좁은 settlement를 택할지 의견을 제시하라.",
            "scenario_facts": "asserted patents are US 5,456,669, US 5,658,261, US 5,662,612, and US 5,928,197. 이전 appeal에서 pressure jacket 없는 해석을 얻었다. remand에서 피고는 full scope non-enablement를 주장한다. 명세서에는 disposable syringe without a pressure jacket 실시예가 없다. 원고는 expert로 당시 syringe 재료를 설명할 수 있다고 본다.",
            "reference_answer": "소송 계속 여부는 넓은 claim construction의 대가를 정직하게 반영해야 한다. Liebel에서 문제 된 네 특허의 full scope는 jacket 있는 injector와 없는 injector를 모두 포함했고, 명세서가 jacketless disposable syringe를 설명하지 않는 점이 치명적이었다. expert가 당시 재료 지식을 말해도 specification이 그 범위를 과도한 실험 없이 만들고 쓰게 했는지가 별개다. 강한 선택지는 jacketless 제품에 대한 손해액 기대치를 낮춰 settlement를 모색하고, 남은 jacket 포함 제품 또는 narrower dependent claims가 있는지 별도 charting하는 것이다. 필요한 증거는 prior appeal construction, Wands-factor 자료, jacketless safety testing이다.",
            "required_points": ["exact asserted patent IDs", "prior broad construction consequence", "absence of jacketless embodiment", "expert limits", "settlement and narrower-claim option"],
            "forbidden_claims": ["US 5,300,031을 appealed patent로 기재", "expert knowledge alone cures missing embodiment", "broad construction only helps patentee"],
            "inferior_answer": "이전 appeal에서 jacket 없는 해석을 얻었다면 침해 쟁점은 상당히 유리하다. 명세서에 jacketless 실시예가 없다는 약점은 있지만, 숙련자가 syringe 재료와 고압 injector 설계를 알고 있었음을 expert가 설명하면 Wands factors에서 방어할 수 있다. 네 특허 모두 관련 front-loading 구조를 공유하므로 settlement를 서두르기보다 피고의 non-enablement 입증 부담을 압박하는 전략이 낫다.",
            "preference_reason": "열등 답변은 넓은 해석의 침해 이점을 잘 짚지만, 그 넓이가 곧 full-scope enablement 부담을 만든다는 Liebel의 중심 교환관계를 낮게 본다."
        },
    ],
    "US-AMGEN-SANOFI-2023": [
        {
            "episode_id": "US-AMGEN-SANOFI-2023-EP1",
            "question": "PCSK9 항체 genus claim 초안이 Amgen v. Sanofi의 enablement 문제를 피할 수 있는지 검토하라.",
            "scenario_facts": "의뢰인은 PCSK9의 특정 residues에 결합하고 LDL receptor 결합을 차단하는 항체를 넓게 청구하려 한다. 명세서에는 32개 항체 서열, epitope binning data, conservative substitution 계획이 있다. 연구팀은 추가 항체를 만들려면 screening campaign이 필요하다고 인정한다. 경쟁사는 다른 CDR scaffold를 쓴다.",
            "reference_answer": "Amgen v. Sanofi의 역사적 범위에서는 더 많이 청구할수록 더 많이 enable해야 한다. 32개 서열과 epitope data는 출발점이지만, 기능으로 정의된 전체 항체 class를 만들고 쓰게 하는지 봐야 한다. conservative substitution이나 roadmap이 실제 예측 가능한 교시가 아니라 새 연구과제라면 부족하다. 초안은 binding residues와 blocking function만으로 entire genus를 독점하기보다, 대표 antibody families, CDR motif, tolerated substitutions, failed variants, screening burden을 명세서에 구체화하고 청구항도 지원된 subgenus로 나누는 편이 안전하다. 누락 증거는 scaffold 다양성, success rate, substitution rule의 예측 가능성이다.",
            "required_points": ["full-scope enablement", "functional antibody genus", "roadmap/research assignment distinction", "representative families and motifs", "missing success-rate evidence"],
            "forbidden_claims": ["항체 genus 전면 금지", "32개 예시면 모든 기능항체 enable", "roadmap이면 항상 충분"],
            "inferior_answer": "32개 항체 서열과 epitope binning data가 있으면 PCSK9 residues와 blocking function 사이의 관계를 상당히 보여 준다. Amgen도 모든 embodiment를 하나하나 만들 필요는 없다고 했으므로, conservative substitution 계획을 자세히 쓰면 경쟁사의 다른 scaffold까지 포괄하는 넓은 claim을 시도할 수 있다. screening campaign이 필요하다는 점은 생명공학 분야의 통상적 검증 부담으로 설명 가능하다.",
            "preference_reason": "열등 답변은 모든 embodiment 불요라는 맞는 원칙을 사용하지만, screening campaign이 research assignment인지 enablement 교시인지 구별하지 않는다."
        },
        {
            "episode_id": "US-AMGEN-SANOFI-2023-EP2",
            "question": "경쟁사 항체가 다른 scaffold를 쓸 때 넓은 functional claim 집행과 보정 선택지를 제시하라.",
            "scenario_facts": "등록청구항은 residues S153, I154, P155에 결합하고 PCSK9-LDLR interaction을 차단하는 항체를 포함한다. 경쟁사 항체는 같은 residues 근처에 결합하지만 CDR length와 heavy-chain family가 다르다. 명세서의 working examples는 두 antibody families에 몰려 있다. 원고는 doctrine of equivalents도 예비 주장하려 한다.",
            "reference_answer": "문언 침해는 epitope와 blocking function이 맞으면 출발점이 있지만, Amgen식 enablement 공격이 바로 따라온다. 다른 scaffold까지 claim이 미치려면 명세서가 그 family도 만들고 사용할 수 있게 했는지, 아니면 새 항체 발견을 trial-and-error에 맡겼는지 확인해야 한다. 보정 또는 재발행·계속출원 검토에서는 CDR motif, antibody family, binding assay 조건, blocking threshold로 narrower subgenus를 세울 수 있다. 균등론은 구조 차이가 큰 scaffold에 대해 예측가능성과 prosecution history estoppel을 따로 봐야 한다. 필요한 자료는 binding crystal/contact map, scaffold별 substitution data, prosecution amendments이다.",
            "required_points": ["literal infringement and enablement tension", "different scaffold facts", "narrower subgenus options", "DOE limits", "missing structural/prosecution evidence"],
            "forbidden_claims": ["같은 epitope면 enablement 자동 충족", "균등론으로 enablement 문제 회피", "current good law statement"],
            "inferior_answer": "경쟁사 항체가 같은 residues 근처에 결합하고 PCSK9-LDLR 차단 기능을 수행한다면 functional claim의 문언 침해 가능성이 높다. 명세서에 두 antibody families와 substitution 방법이 있으므로 다른 scaffold도 예측 가능한 변형이라고 주장할 수 있다. enablement 공격은 예상되지만, 모든 항체를 개별적으로 설명할 필요는 없다는 점과 epitope mapping data를 강조하면 된다. 균등론은 구조 차이가 큰 경우의 예비 수단이다.",
            "preference_reason": "열등 답변은 침해 논리를 그럴듯하게 세우지만, 다른 scaffold를 예측 가능한 변형으로 볼 실증 자료와 research-assignment 위험을 충분히 요구하지 않는다."
        },
    ],
    "US-JUNO-2021": [
        {
            "episode_id": "US-JUNO-2021-EP1",
            "question": "CAR-T scFv genus claim의 written description 지원을 Juno v. Kite 기준으로 평가하라.",
            "scenario_facts": "초안은 어떤 target antigen에 결합하는 scFv라도 포함하는 CAR construct를 청구한다. 명세서에는 CD19 scFv 하나와 PSMA scFv 하나가 있고, linker와 signaling domains는 자세히 설명되어 있다. 연구팀은 scFv sequence가 binding specificity를 좌우한다고 인정한다. 새 제품은 BCMA-binding scFv를 쓴다.",
            "reference_answer": "Juno의 역사적 범위에서는 CAR의 다른 부분이 잘 설명되어 있어도 functional scFv genus 자체의 possession이 필요하다. 두 scFv 예와 일반적 scFv 설명만으로 모든 target 또는 BCMA scFv까지 대표한다고 보기 어렵다. 초안은 target별 representative scFv, sequence motifs, CDR features, binding affinity thresholds, known target-specific examples를 명세서에 넣고, 독립항을 'CD19-binding scFv' 같은 supported subgenus로 나누는 방안을 고려해야 한다. 반론은 scFv framework가 유사하다는 점이나, Juno는 binding ability가 sequence에 크게 좌우될 수 있음을 중시한다. 누락 증거는 BCMA scFv 당시 지식, cross-target predictability, common structural features이다.",
            "required_points": ["functional scFv genus", "representative species/common structural features", "target-specific binding facts", "supported subgenus option", "missing sequence evidence"],
            "forbidden_claims": ["CAR backbone 설명으로 scFv genus 지원", "두 scFv가 모든 target 대표", "enablement와 written description 혼동"],
            "inferior_answer": "CAR construct의 핵심은 scFv가 target을 인식하고 signaling domain이 T cell activation을 일으키는 구조이다. 명세서가 CD19와 PSMA scFv를 보여 주고 scFv framework와 linker를 설명한다면, 숙련자는 다른 target scFv도 같은 CAR 구조에 넣을 수 있다. BCMA scFv는 target만 다를 뿐 binding element 역할이 같으므로 broad claim을 시도할 수 있다. 종속항에서 CD19와 PSMA를 보강하면 fallback도 있다.",
            "preference_reason": "열등 답변은 CAR 조립 가능성을 scFv genus possession으로 바꾸어, Juno의 target-specific representative species 요구를 약화한다."
        },
        {
            "episode_id": "US-JUNO-2021-EP2",
            "question": "Kite형 피고가 scFv written description 부족을 주장할 때 원고의 대응 가능성과 약점을 정리하라.",
            "scenario_facts": "asserted claims cover a nucleic acid polymer encoding a three-part CAR with a binding element. 피고 제품은 CD19를 표적으로 하지만 명세서에 기재된 SJ25C1과 다른 scFv sequence를 쓴다. 원고 전문가는 많은 CD19 scFv가 알려져 있었다고 말한다. 특허 본문은 어떤 scFv가 어떤 target에 결합하는지 자세히 설명하지 않는다.",
            "reference_answer": "원고는 피고도 CD19를 표적으로 하므로 target이 완전히 새로운 것은 아니고, 알려진 CD19 scFv 지식이 보충될 수 있다고 주장할 수 있다. 그러나 Juno는 claimed functional scFv genus에 대해 representative species나 common structural features가 없으면 written description을 충족하기 어렵다고 보았다. '많이 알려져 있었다'는 전문가 말은 어느 sequences가 priority date에 알려졌고 발명자가 이를 possession했는지로 구체화되어야 한다. 약점은 명세서가 SJ25C1 외 CD19 scFv의 sequence/structure를 거의 주지 않는다는 점이다. 선택지는 손해·침해 주장을 유지하되 validity discount를 반영하고, 후속출원에서는 target별 scFv panels를 넣는 것이다.",
            "required_points": ["defense based on known CD19 scFvs", "limits from Juno conclusion", "priority-date evidence", "specific accused sequence difference", "practical risk option"],
            "forbidden_claims": ["피고도 CD19라 written description 자동 충족", "전문가의 일반 지식만으로 possession 확정", "damages verdict로 validity issue 해결"],
            "inferior_answer": "피고 제품도 CD19 CAR-T이고, 명세서에 CD19-binding SJ25C1 scFv가 있으므로 asserted claim의 핵심 target은 공개되어 있다. 다른 CD19 scFv sequence를 쓴 점은 등가적 binding element 선택으로 볼 수 있고, 원고 전문가가 priority date에 여러 CD19 scFv가 알려져 있었다고 설명하면 written description 방어가 가능하다. Juno의 위험은 있지만 completely different target을 다루는 경우보다 낮다.",
            "preference_reason": "열등 답변은 같은 CD19 target이라는 유리한 사실을 과대평가해, 다른 sequence를 포함하는 genus의 representative support 필요성을 충분히 묻지 않는다."
        },
    ],
    "US-IDENIX-2019": [
        {
            "episode_id": "US-IDENIX-2019-EP1",
            "question": "HCV nucleoside 치료제 genus claim의 enablement와 written description 위험을 평가하라.",
            "scenario_facts": "초안은 2'-methyl-up nucleosides 중 HCV 치료 효과가 있는 화합물을 넓게 청구한다. 명세서에는 주로 2'-down OH 화합물 예가 있고, 2'-fluoro-down 화합물 데이터는 없다. 경쟁 제품은 2'-methyl-up 2'-fluoro-down 구조이다. 가능한 후보 수는 매우 크고, screening assay는 시간이 많이 든다.",
            "reference_answer": "Idenix의 역사적 범위에서 핵심 enablement 질문은 숙련자가 과도한 실험 없이 어떤 2'-methyl-up nucleosides가 HCV에 효과적인지 알 수 있었는가이다. 후보가 방대하고 fluorine 치환 예가 없으며 방향성이 약하면 full scope enablement가 취약하다. written description도 작동 화합물의 대표 범위와 공통 구조를 보여야 한다. 초안은 2'-fluoro-down species를 실제 data로 넣거나, sugar/base 치환 범위를 검증된 subgenus로 줄이고, inactive examples와 SAR guidance를 함께 제시해야 한다. 반론은 nucleoside chemistry의 선행지식이 많다는 점이나, 경쟁 제품 같은 핵심 embodiment를 찾는 부담이 크면 부족하다.",
            "required_points": ["full-scope enablement", "2'-fluoro-down accused embodiment", "billions/large candidate space", "SAR and inactive examples", "written description distinction"],
            "forbidden_claims": ["후보 수만으로 자동 무효", "accused product effectiveness proves enablement", "screening 가능성만으로 충분"],
            "inferior_answer": "nucleoside 분야에는 HCV assay와 SAR 지식이 있었고, 명세서가 2'-methyl-up 핵심 구조를 알려 주므로 숙련자가 후보를 선별할 수 있다. 2'-fluoro-down 데이터가 없더라도 경쟁 제품은 같은 2'-methyl-up family에 속한다. 후보 수가 많다는 점은 부담이지만, Idenix에서도 Wands factors 전체를 보아야 하므로 broad genus를 곧바로 포기할 필요는 없다.",
            "preference_reason": "열등 답변은 Wands-factor 전체 검토를 말하지만, 핵심 accused embodiment까지 가는 구체적 guidance 부재와 full-scope 부담을 낮게 본다."
        },
        {
            "episode_id": "US-IDENIX-2019-EP2",
            "question": "배심 평결 후 JMOL 단계에서 non-enablement 공격을 받은 원고 측 대응을 설계하라.",
            "scenario_facts": "배심은 validity를 인정했지만 피고는 JMOL로 enablement와 written description을 다시 제기한다. 청구항 해석은 2'-methyl-up nucleosides를 넓게 포함한다. 원고 전문가는 small set of promising compounds를 고르면 된다고 증언했다. 피고는 billions of candidates와 실패 screening 기록을 제출했다.",
            "reference_answer": "JMOL 대응은 배심 사실인정의 존중을 주장하면서도, enablement의 법률 결론이 full scope와 undue experimentation에 달려 있음을 피할 수 없다. Idenix에서처럼 넓은 construction이 유지되면 'promising set' 접근이 청구범위 전체를 실시 가능하게 하는지 설명해야 한다. 피고의 billions evidence와 실패 기록은 실험 부담을 보여 주므로, 원고는 priority date의 concrete guideposts, working examples의 representativeness, fluorine substitution 예측 가능성을 특정해야 한다. written description은 별도로 발명자가 broad genus를 보유했는지 다툰다. 약하면 JMOL 방어와 병행해 damages leverage 기반 settlement를 검토한다.",
            "required_points": ["JMOL posture", "jury deference limits", "broad construction consequence", "guideposts and failure evidence", "separate WD issue"],
            "forbidden_claims": ["배심 평결 때문에 JMOL 불가능", "promising set이면 full scope enablement 자동 충족", "enablement와 WD를 하나로 처리"],
            "inferior_answer": "배심이 validity를 인정했고 원고 전문가가 promising compounds를 선별할 수 있다고 설명했다면 JMOL은 높은 장벽을 가진다. 피고의 billions argument는 이론적 조합 수를 부풀린 측면이 있고, 실제 medicinal chemist는 모든 후보를 만들지 않는다. 따라서 대응은 배심의 신빙성 판단과 실무적 screening route를 강조해야 한다. written description도 같은 working examples가 broad family를 뒷받침한다는 방향으로 함께 방어할 수 있다.",
            "preference_reason": "열등 답변은 JMOL 장벽을 잘 말하지만, full scope legal conclusion과 written-description possession을 배심 신빙성·screening route로 과잉 대체한다."
        },
    ],
    "US-ENZO-2010": [
        {
            "episode_id": "US-ENZO-2010-EP1",
            "question": "nucleic-acid probe claim의 'not interfering substantially' 표현을 미국 명확성 기준으로 다듬어라.",
            "scenario_facts": "청구항은 label과 nucleotide 사이의 linkage group이 hybridization과 detection을 substantially interfere하지 않는다고 한다. 명세서에는 biotin/avidin detection, fluorescent label, enzyme product examples가 있다. 연구팀은 일부 bulky linker가 signal을 낮추지만 threshold를 정하지 않았다. 경쟁 probe는 longer PEG linker를 쓴다.",
            "reference_answer": "Enzo v. Applera의 역사적 범위에서는 기능적 biochemical limitation이 반드시 indefinite는 아니지만, 명세서 예와 기술 문맥이 경계를 제공해야 한다. 'not interfering substantially'를 유지하려면 hybridization melting temperature 변화, detection signal retention, assay conditions, acceptable linker classes를 기재해 실질적 간섭 여부를 판단할 기준을 마련해야 한다. biotin/avidin과 fluorescent examples는 기준을 세우는 출발점이다. 반론은 숙련자가 hybridization assay를 알고 있다는 점이나, bulky linker와 PEG linker를 경계 안팎으로 나누려면 threshold가 필요하다. 누락 자료는 linker panel data, assay protocol, detection sensitivity 기준이다.",
            "required_points": ["functional biochemical limitation", "examples plus objective assay criteria", "hybridization/detection facts", "threshold drafting", "missing linker data"],
            "forbidden_claims": ["기능 표현은 언제나 indefinite", "assay 지식만으로 경계 자동 형성", "written description까지 자동 충족"],
            "inferior_answer": "not interfering substantially는 probe 분야에서 linker가 hybridization과 detection을 방해하지 않는다는 통상적 의미가 있다. 명세서가 biotin/avidin, fluorescent labels, enzyme detection을 설명하므로 숙련자는 적절한 linker를 고를 수 있다. 구체 threshold를 넣으면 label chemistry마다 달라져 청구항이 불필요하게 좁아질 수 있다. 따라서 독립항은 현재 표현을 유지하고 실시예 data를 추가하는 정도가 좋다.",
            "preference_reason": "열등 답변은 분야 지식과 예시를 근거로 삼지만, longer PEG 같은 경계 사례를 판별할 객관 기준을 충분히 요구하지 않는다."
        },
        {
            "episode_id": "US-ENZO-2010-EP2",
            "question": "피고가 직접검출 probe는 claim construction상 제외된다고 주장할 때 대응과 대체 취약점을 평가하라.",
            "scenario_facts": "명세서는 direct detection과 indirect detection을 모두 설명하지만 일부 claim examples는 secondary chemical agent를 강조한다. 피고 제품은 fluorescent label로 직접검출을 한다. 원고는 'A' linkage group이 detection을 방해하지 않는다는 language가 직접검출도 포함한다고 본다. 피고는 written description과 enablement도 예비 주장한다.",
            "reference_answer": "대응은 claim language가 direct detection을 배제하는지와, 명세서가 direct detection examples를 실제로 지원하는지 분리해야 한다. Enzo 2010은 'not interfering substantially' 같은 기능 표현이 문맥상 definite일 수 있음을 보여 주지만, 직접검출을 포함하는 해석을 얻으면 written description·enablement 예비공격이 강해질 수 있다. fluorescent label 예와 hybridization 설명은 원고에게 유리하나, secondary agent 중심 문구가 disclaimer처럼 작용했는지 확인해야 한다. 선택지는 직접검출 포함 construction을 주장하되, fallback으로 indirect detection claims와 특정 fluorescent embodiments를 별도 charting하는 것이다. 필요한 자료는 claim differentiation, prosecution remarks, 실험 examples이다.",
            "required_points": ["claim construction versus validity fallback", "direct/indirect detection facts", "functional term definiteness", "disclaimer check", "fallback charting"],
            "forbidden_claims": ["직접검출 포함이면 written description 문제 없음", "secondary-agent examples를 무조건 disclaimer로 봄", "피고 제품 방식으로 claim meaning 확정"],
            "inferior_answer": "명세서가 fluorescent label을 언급하고 probe가 hybridize한 뒤 detectable signal을 낸다고 설명하므로 직접검출도 claim scope에 포함된다고 주장할 수 있다. 'not interfering substantially'는 linker의 영향만 보는 표현이어서 detection 방식 전체를 제한하지 않는다. 피고의 written description과 enablement 주장은 claim construction을 흐리려는 예비공격에 가깝다. 따라서 직접검출 포함 해석을 강하게 밀고 가는 것이 좋다.",
            "preference_reason": "열등 답변은 직접검출 포함 논리를 잘 제시하지만, 넓은 해석이 validity 예비공격을 키우는 연결과 disclaimer 검토를 빠뜨린다."
        },
    ],
    "US-NYSTROM-2005": [
        {
            "episode_id": "US-NYSTROM-2005-EP1",
            "question": "decking patent의 'board'가 합성재를 포함하는지 미국 청구항 해석 의견을 작성하라.",
            "scenario_facts": "청구항은 convex top surface를 가진 'board'를 recite한다. 명세서 배경과 실시예는 wood decking을 반복적으로 말하고, 톱질·나뭇결 문제를 설명한다. 경쟁 제품은 recycled plastic composite plank이다. 사전은 board를 wood 또는 board-like material로 넓게 정의한다.",
            "reference_answer": "Nystrom의 역사적 범위에서는 사전상 넓은 의미가 출발점이 될 수 있지만, 명세서 전체가 'board'를 목재 decking 맥락으로 사용했는지 확인해야 한다. 합성재를 포함하려면 청구항이나 명세서에 wood 외 재료를 열어 둔 표현이 필요하다. wood warping, saw-cutting, grain 같은 반복 설명은 좁은 construction에 불리하게 작용한다. 반론은 claim이 material을 명시하지 않았고 composite plank도 board-like라는 점이다. 그러나 intrinsic record가 반복적으로 wood problem을 발명의 배경으로 삼으면 dictionary breadth만으로 부족할 수 있다. 필요한 자료는 prosecution에서 material 삭제 여부, specification의 alternative material 문장, composite의 구조적 유사성이다.",
            "required_points": ["dictionary versus intrinsic context", "wood/composite facts", "repeated specification usage", "counterargument from claim silence", "missing prosecution evidence"],
            "forbidden_claims": ["사전상 board면 합성재 자동 포함", "실시예가 wood면 무조건 wood 한정", "ordinary meaning만으로 결론"],
            "inferior_answer": "청구항이 wood라고 쓰지 않았고 board의 사전 의미는 넓으므로 composite plank도 포함된다고 주장할 수 있다. 명세서가 wood examples를 많이 들었더라도 그것은 당시 decking의 일반적 재료를 설명한 것이다. 경쟁 제품이 같은 convex top surface와 decking function을 가진다면 Nystrom을 이유로 범위를 지나치게 좁히면 안 된다. 심사기록에 명확한 wood 제한이 없다면 침해 주장을 유지할 만하다.",
            "preference_reason": "열등 답변은 claim silence와 사전 의미를 타당하게 제시하지만, 명세서 반복 문맥이 ordinary meaning을 제한할 수 있다는 Nystrom의 무게를 낮게 둔다."
        },
        {
            "episode_id": "US-NYSTROM-2005-EP2",
            "question": "후속출원에서 'board'를 넓게 쓰고 싶을 때 명세서를 어떻게 설계할지 조언하라.",
            "scenario_facts": "의뢰인은 목재, PVC, wood-plastic composite 모두에 적용되는 deck plank 형상을 개발했다. 발명의 장점은 물 빠짐과 slip reduction이고 재료별 제조법은 다르다. 첫 draft 배경은 wood deck의 rot와 cupping 문제만 길게 설명한다. 도면은 목재 판재처럼 보인다.",
            "reference_answer": "Nystrom식 위험을 줄이려면 'board'가 목재에 갇히지 않도록 명세서 초반부터 plank/member/panel과 material examples를 병렬로 써야 한다. 배경의 wood problem은 한 적용 예로 제한하고, PVC와 composite에서도 물 빠짐·slip reduction이 작동하는 이유를 설명한다. 청구항은 material-neutral structural surface를 쓰고, 종속항에 wood, PVC, composite를 나누는 방식이 좋다. 반론은 너무 많은 재료를 넣으면 enablement 부담이 커진다는 점이므로 제조 조건과 재료별 surface formation을 충분히 기재해야 한다. 필요한 자료는 material-specific prototypes, surface roughness data, manufacturing tolerances이다.",
            "required_points": ["drafting to avoid intrinsic narrowing", "multiple material examples", "dependent material claims", "enablement tradeoff", "specific deck facts"],
            "forbidden_claims": ["사전 정의를 명세서에 붙이면 항상 충분", "wood background만 두고 claim만 넓게 작성", "enablement 부담 누락"],
            "inferior_answer": "후속출원에서는 청구항 definition section에 'board includes wood, plastic, composite, and equivalent plank materials'라고 쓰면 넓은 의미를 확보할 수 있다. 배경에서 wood 문제를 설명하는 것은 시장에서 가장 흔한 문제를 보여 주는 것이므로 괜찮다. 도면도 판재 형상만 보여 주면 재료에는 중립적이다. 종속항에 PVC와 composite를 추가하면 Nystrom식 narrowing 위험은 대부분 해결된다.",
            "preference_reason": "열등 답변은 정의 문장과 종속항을 제안하지만, 명세서 전체 문맥과 enablement 자료가 정의를 뒷받침해야 한다는 점을 과소평가한다."
        },
    ],
    "US-THORNER-2012": [
        {
            "episode_id": "US-THORNER-2012-EP1",
            "question": "게임 컨트롤러 특허에서 'attached to said pad'가 embedded actuator를 포함하도록 작성하려면 어떻게 해야 하는가.",
            "scenario_facts": "발명은 tactile feedback pad와 actuator를 가진 controller이다. 모든 실시예는 actuator가 pad 안쪽에 내장된 구조를 보여 준다. 청구항은 'attached to said pad'라고만 쓰고 exterior surface라는 말은 없다. 경쟁 제품은 actuator가 pad 내부 foam layer에 묻혀 있다. 명세서 draft는 attached와 embedded를 혼용한다.",
            "reference_answer": "Thorner의 역사적 범위에서는 평이한 의미에서 벗어나려면 명시적 lexicography 또는 clear disavowal이 필요하다. embedded actuator를 포함하려면 'attached'를 외부 부착으로 좁히지 말고, specification에 attached includes affixed to, coupled with, or embedded within the pad 같은 정의를 분명히 두는 것이 좋다. 모든 실시예가 embedded라는 사실만으로는 평이한 의미를 재정의하기에 부족할 수 있다. 반론은 repeated embedded embodiments가 발명 의도를 보여 준다는 점이나, Thorner는 단일 방식 반복만으로 명확한 재정의가 되지 않는다고 경계한다. 필요한 자료는 claim differentiation, prosecution remarks, mechanical coupling evidence이다.",
            "required_points": ["plain meaning", "lexicography/disavowal standard", "embedded actuator facts", "explicit definition drafting", "embodiment repetition limits"],
            "forbidden_claims": ["모든 실시예가 embedded면 자동 재정의", "attached는 반드시 exterior", "정의 없이 litigation에서 의미 변경"],
            "inferior_answer": "모든 실시예가 actuator를 pad 내부에 둔다면 attached to said pad는 embedded attachment를 포함한다고 주장할 수 있다. 일반적으로 부착은 직접 외부 접착뿐 아니라 구조적으로 결합된 상태도 포함한다. 따라서 명세서 draft에서 attached와 embedded를 함께 쓰는 것은 넓은 의미를 뒷받침한다. 굳이 정의 문장을 길게 넣으면 외부 부착 실시예를 배제하는 듯 보일 수 있으므로, 여러 실시예 설명으로 의미를 보여 주는 편이 자연스럽다.",
            "preference_reason": "열등 답변은 일반 의미를 그럴듯하게 설명하지만, 명시적 정의 없이 실시예 반복과 혼용만으로 의미를 안정화할 수 있다고 보는 점이 위험하다."
        },
        {
            "episode_id": "US-THORNER-2012-EP2",
            "question": "피고가 'attached'는 exterior surface 부착만 의미한다고 주장할 때 반박하라.",
            "scenario_facts": "청구항은 actuator attached to said pad를 요구한다. 명세서에는 actuator가 pad 내부에 위치한 그림이 있고, 한 문단은 actuator can be affixed to the pad라고 말한다. 피고는 pad 안쪽에 묻힌 actuator는 exterior에 attached된 것이 아니라고 한다. 심사 중 exterior limitation은 논의되지 않았다.",
            "reference_answer": "반박은 plain and ordinary meaning에서 attached가 반드시 exterior surface만 뜻하지 않는다는 점에서 시작한다. Thorner에 따르면 좁히려면 patentee가 특별 정의를 하거나 full scope를 명확히 포기해야 한다. 여기서는 exterior limitation이 청구항에 없고 심사 중 포기 발언도 없으므로 피고의 좁은 해석은 약하다. 명세서의 internal embodiment는 embedded attachment가 배제되지 않음을 보여 주는 보조 근거가 된다. 다만 'affixed' 문구가 surface 접착처럼 읽힐 수 있으므로, missing evidence는 당시 mechanical coupling usage, 도면 설명, prosecution disclaimer 부재의 전문이다.",
            "required_points": ["ordinary meaning not exterior-only", "no lexicography/disavowal", "internal embodiment use", "counterargument from affixed wording", "missing usage/prosecution evidence"],
            "forbidden_claims": ["실시예 하나로 broad scope 자동 확정", "피고 제품 구조로 청구항 재정의", "disavowal 기준을 낮게 적용"],
            "inferior_answer": "명세서 도면이 내부 actuator를 보여 주므로 attached는 embedded를 포함한다고 강하게 주장할 수 있다. 피고가 exterior surface라고 좁히려면 청구항이나 심사기록에 그런 말이 있어야 하는데 없다. Thorner는 단일 실시예만으로 제한하지 말라고 했으므로 원고에게 유리하다. 따라서 internal foam layer에 묻힌 actuator도 pad에 attached된 것으로 해석될 가능성이 높다.",
            "preference_reason": "열등 답변은 결론 방향은 맞지만, '도면이 내부이므로 포함'이라는 쉬운 논리에 기대고 affixed 문구와 기술용어 사용 증거의 약점을 덜 다룬다."
        },
    ],
}


def episode_pair(spec: CaseSpec, ev_ids: list[str]) -> list[dict]:
    try:
        authored = EPISODES[spec.case_id]
    except KeyError as exc:
        raise RuntimeError(f"missing manual episodes for {spec.case_id}") from exc
    result = []
    for episode in authored:
        row = dict(episode)
        row["scenario_origin"] = "author_hypothetical_for_rl_material"
        row["temporal_scope"] = f"미국 {spec.decision_date} 판결 당시의 역사적 범위. 최신 유효 법상태는 별도 확인 필요."
        row["evidence_ids"] = ev_ids
        result.append(row)
    return result


def make_record(spec: CaseSpec, pdf_path: str, pages: list[str]) -> dict:
    sha = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    used_pages: set[int] = set()
    spans = [
        find_span(spec.case_id, pages, term, used_pages, idx)
        for idx, term in enumerate(spec.quote_terms, 1)
    ]
    ev_ids = [span["id"] for span in spans]
    return {
        "case_id": spec.case_id,
        "jurisdiction": "US",
        "case_name": spec.case_name,
        "docket": spec.docket,
        "court": spec.court,
        "decision_date": spec.decision_date,
        "pdf_path": pdf_path,
        "source_sha256": sha,
        "case_patent_ids": list(spec.patent_ids),
        "lineage_notes": spec.lineage_notes,
        "history_scope": {
            "mode": "historical_as_of_decision",
            "as_of_date": spec.decision_date,
            "sources": [pdf_path],
            "limits": "Only the selected primary PDF was reviewed for this authoring pass; no current-law or later-history conclusion is asserted.",
        },
        "evidence_spans": spans,
        "principles": [
            {
                "id": f"{spec.case_id}-P1",
                "statement": spec.principle,
                "evidence_ids": ev_ids,
                "scope": f"US historical advisory material for {spec.issue}.",
                "non_rules": [spec.non_rule],
            }
        ],
        "advisory_episodes": episode_pair(spec, ev_ids),
        "author": "/root/us_materials",
        "review_status": "author_complete_pending_review",
    }


def validate_records(records: list[dict], page_index: dict[tuple[str, int], str]) -> list[str]:
    errors: list[str] = []
    seen = set()
    total_episodes = 0
    for record in records:
        required = {
            "case_id", "jurisdiction", "case_name", "docket", "court", "decision_date",
            "pdf_path", "source_sha256", "case_patent_ids", "lineage_notes",
            "history_scope", "evidence_spans", "principles", "advisory_episodes",
            "author", "review_status",
        }
        if set(record) != required:
            errors.append(f"{record.get('case_id')}: keys")
        cid = record["case_id"]
        if cid in seen:
            errors.append(f"{cid}: duplicate")
        seen.add(cid)
        if record["jurisdiction"] != "US":
            errors.append(f"{cid}: jurisdiction")
        if record["review_status"] != "author_complete_pending_review":
            errors.append(f"{cid}: review_status")
        if not record["case_patent_ids"]:
            errors.append(f"{cid}: patents")
        if hashlib.sha256(Path(record["pdf_path"]).read_bytes()).hexdigest() != record["source_sha256"]:
            errors.append(f"{cid}: sha")
        span_ids = {span["id"] for span in record["evidence_spans"]}
        for span in record["evidence_spans"]:
            page_text = page_index[(cid, span["pdf_page"])]
            if span["quote"] not in norm(page_text):
                errors.append(f"{cid}: quote mismatch {span['id']}")
        for principle in record["principles"]:
            if not set(principle["evidence_ids"]).issubset(span_ids):
                errors.append(f"{cid}: principle evidence")
        episodes = record["advisory_episodes"]
        total_episodes += len(episodes)
        if len(episodes) < 2:
            errors.append(f"{cid}: episode count")
        for episode in episodes:
            if not set(episode["evidence_ids"]).issubset(span_ids):
                errors.append(f"{cid}: episode evidence")
            if len(norm(episode["reference_answer"])) < 300:
                errors.append(f"{cid}: thin reference")
            if len(episode["required_points"]) < 4 or len(episode["forbidden_claims"]) < 3:
                errors.append(f"{cid}: rubric")
            if len(norm(episode["inferior_answer"])) < 80:
                errors.append(f"{cid}: thin inferior")
    if len(seen) != 20:
        errors.append(f"case_count={len(seen)}")
    if total_episodes < 40:
        errors.append(f"episode_count={total_episodes}")
    return errors


def main() -> None:
    paths = inventory_paths()
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    page_index: dict[tuple[str, int], str] = {}
    page_dump = EVIDENCE_ROOT / "source_pages.jsonl"
    with page_dump.open("w", encoding="utf-8", newline="\n") as page_handle:
        for spec in CASES:
            pdf_path = find_pdf(spec.path_fragment, paths)
            pages = extract_pages(pdf_path)
            record = make_record(spec, pdf_path, pages)
            records.append(record)
            pages_needed = sorted({span["pdf_page"] for span in record["evidence_spans"]})
            for page_no in pages_needed:
                text = pages[page_no - 1]
                page_index[(spec.case_id, page_no)] = text
                page_handle.write(json.dumps({
                    "case_id": spec.case_id,
                    "pdf_path": pdf_path,
                    "source_sha256": record["source_sha256"],
                    "pdf_page": page_no,
                    "page_text_sha256": hashlib.sha256(norm(text).encode("utf-8")).hexdigest(),
                    "page_text": text,
                }, ensure_ascii=False) + "\n")

    errors = validate_records(records, page_index)
    if errors:
        raise SystemExit("\n".join(errors))

    with OUT_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    validation = {
        "records": len(records),
        "distinct_families": len({record["case_id"] for record in records}),
        "episodes": sum(len(record["advisory_episodes"]) for record in records),
        "evidence_spans": sum(len(record["evidence_spans"]) for record in records),
        "review_status": "author_complete_pending_review",
        "validated": True,
    }
    (EVIDENCE_ROOT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    record_hashes = {record["case_id"]: canonical_hash(record) for record in records}
    ariad_hash = record_hashes["US-ARIAD-2010"]
    preserved_hashes = {
        case_id: current_hash
        for case_id, current_hash in record_hashes.items()
        if case_id != "US-ARIAD-2010" and current_hash == PRESERVED_19_HASHES.get(case_id)
    }
    if ariad_hash != POST_ARIAD_FIX_HASH:
        raise SystemExit(f"Unexpected Ariad canonical hash after E2 repair: {ariad_hash}")
    if len(preserved_hashes) != len(PRESERVED_19_HASHES):
        mismatches = {
            case_id: record_hashes.get(case_id)
            for case_id in PRESERVED_19_HASHES
            if record_hashes.get(case_id) != PRESERVED_19_HASHES[case_id]
        }
        raise SystemExit(f"Non-Ariad canonical hash mismatch after Ariad-only repair: {mismatches}")

    report = [
        "# US case RL material author report",
        "",
        "- Assignment: execute US case-based RL material curation for 20 distinct reasoned US patent case families.",
        "- Author: /root/us_materials.",
        f"- JSONL deliverable: `{OUT_JSONL.as_posix()}`.",
        f"- Evidence page dump: `{page_dump.as_posix()}`.",
        f"- Validation artifact: `{(EVIDENCE_ROOT / 'validation.json').as_posix()}`.",
        "- Source/schema audit artifact: `.superloopy/evidence/us-source-schema-audit.json`.",
        f"- Records: {validation['records']}; advisory episodes: {validation['episodes']}; evidence spans: {validation['evidence_spans']}.",
        "- Repair receipt: all 40 advisory episodes were replaced with manually authored case-specific hypotheticals, reference answers, inferior answers, and preference reasons.",
        "- Follow-up Ariad repair: `US-ARIAD-2010-E2` was recut from the page-20 party-characterized possession discussion to the court-analysis passage containing `had possession`.",
        f"- Ariad canonical hash changed only for `US-ARIAD-2010`: `{PRE_ARIAD_FIX_HASH}` -> `{ariad_hash}`.",
        f"- Non-Ariad preservation check: {len(preserved_hashes)} accepted records retained their exact canonical hashes.",
        "- Anti-template check after regeneration: 40 unique `reference_answer`, 40 unique `inferior_answer`, 40 unique `preference_reason`, and zero literal `IPR` mentions in advisory episodes.",
        "- Review status: `author_complete_pending_review` for every case; no independent legal/content review was claimed.",
        "- Historical scope: each case is limited to the selected primary PDF as of that decision date; no current-good-law assertion is made.",
        "",
        "## Changed files",
        "",
        "- `rl_materials/curation/us_cases.jsonl`",
        "- `tools/curate_us_materials.py`",
        "- `data/case_rl/curation_us/source_pages.jsonl`",
        "- `data/case_rl/curation_us/validation.json`",
        "- `.superloopy/evidence/us-author-report.md`",
        "- `.superloopy/evidence/us-source-schema-audit.json`",
        "",
        "## Validation commands",
        "",
        "- `C:/Users/VIEW LIFW/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe tools/curate_us_materials.py`",
        "- `C:/Users/VIEW LIFW/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe tools/validate_rl_curation.py --lane us`",
        "- Python JSONL reuse check: counted unique reference/inferior/preference fields and searched advisory episodes for literal `IPR`.",
        "- `superloopy loop prove --help` was attempted after validation; the local shim failed because it points to a missing cached `0.7.2/src/cli.js`.",
        "",
        "## Case-by-case repairs",
        "",
    ]
    for record in records:
        issue = next(spec.issue for spec in CASES if spec.case_id == record["case_id"])
        report.append(
            f"- {record['case_id']}: rewrote both advisory episodes around {issue}; kept historical scope, "
            f"source-checked quotes, and `author_complete_pending_review`."
        )
    report.extend([
        "",
        "## Ariad-only re-review hash receipt",
        "",
        f"- Changed field: `US-ARIAD-2010.evidence_spans[1]` (`US-ARIAD-2010-E2`) now uses PDF page 26 court reasoning with the normalized phrase `had possession`; the party-characterized page-20 source is no longer used.",
        f"- Ariad canonical hash before repair: `{PRE_ARIAD_FIX_HASH}`.",
        f"- Ariad canonical hash after repair: `{ariad_hash}`.",
        "- Preserved non-Ariad canonical hashes:",
    ])
    for case_id in sorted(PRESERVED_19_HASHES):
        report.append(f"  - `{case_id}`: `{PRESERVED_19_HASHES[case_id]}`")
    report.extend([
        "",
        "## Curated families",
        "",
    ])
    for record in records:
        report.append(
            f"- {record['case_id']}: {record['case_name']} ({record['court']}, {record['decision_date']}), "
            f"patents {', '.join(record['case_patent_ids'])}, sha256 `{record['source_sha256']}`."
        )
    report.extend([
        "",
        "## Residual risks",
        "",
        "- Source quotations, page numbers and hashes were mechanically checked against the extracted PDF text, but legal scope and later-history status still require independent reviewer approval.",
        "- Advisory episodes are authored hypotheticals for reward material; they are not legal advice and are not marked generator-eligible.",
        "- Superloopy CLI proof capture was unavailable in this environment; `.superloopy/evidence/us-source-schema-audit.json` is the validator-backed audit artifact.",
    ])
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
