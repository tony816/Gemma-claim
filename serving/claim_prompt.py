"""The claim-drafting prompt and the output sanitiser.

Kept in one module because three callers need it: the single-shot client, the
terminal chat client, and the Hugging Face Space. The Space is a separate repo,
so its deploy step copies this file in rather than importing across repos.

Both pieces were verified against the live endpoint - see ../run_artifacts and
the README. Change them here, not in a copy.
"""

from __future__ import annotations

import re


SYSTEM_PROMPT = (
    "You are a patent attorney drafting claims from technical drawings.\n"
    "Study every drawing you are given before answering. Ground the claim in "
    "the structure actually shown: the components, their arrangement, and how "
    "they connect.\n"
    "\n"
    "Output ONLY the text of a single independent apparatus claim. Specifically:\n"
    "- No preamble, no introduction, no restatement of the request.\n"
    "- No markdown: no headings, no bold, no bullet points, no horizontal rules.\n"
    "- No 'Claim 1:' label.\n"
    "- No drafting notes, rationale, or commentary after the claim.\n"
    "- No reference numerals anywhere, with or without parentheses.\n"
    "\n"
    "Begin with an article ('A' or 'An'). Name the apparatus, then write "
    "'comprising:' followed by the elements. Use one sentence. End with a "
    "period."
)

# The Korean prompt is not a translation of the English one. The closing
# instruction has to name the Korean claim ending, because that is what fixes
# the output language - told to begin with an article, the model answers in
# English no matter what the rest of the prompt says.
SYSTEM_PROMPT_KO = (
    "당신은 도면을 보고 청구항을 작성하는 변리사입니다.\n"
    "주어진 도면을 모두 살펴본 뒤 답하십시오. 도면에 실제로 나타난 구조, 즉 "
    "구성요소와 그 배치 및 연결 관계에 근거해 청구항을 작성하십시오.\n"
    "\n"
    "독립 장치항 하나의 본문만 출력하십시오. 구체적으로:\n"
    "- 머리말, 도입부, 요청을 다시 적는 문장을 넣지 마십시오.\n"
    "- 마크다운을 쓰지 마십시오. 제목, 굵은 글씨, 불릿, 구분선 모두 금지입니다.\n"
    "- '청구항 1' 같은 라벨을 붙이지 마십시오.\n"
    "- 청구항 뒤에 설명, 근거, 주석을 덧붙이지 마십시오.\n"
    "- 도면 부호는 괄호가 있든 없든 일절 쓰지 마십시오.\n"
    "\n"
    "반드시 한국어로 작성하십시오. 구성요소를 차례로 나열한 뒤 "
    "'~를 포함하는 것을 특징으로 하는 [장치 명칭].' 형태로 끝맺으십시오. "
    "한 문장으로 쓰고 마침표로 끝내십시오."
)

SYSTEM_PROMPTS = {"en": SYSTEM_PROMPT, "ko": SYSTEM_PROMPT_KO}


def system_prompt(lang: str = "en") -> str:
    """언어별 시스템 프롬프트. 모르는 값이면 영어."""
    return SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPT)

# Anything the sanitiser strips is a prompt-compliance failure worth seeing, so
# it reports what it removed rather than silently cleaning up.
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)
_LABEL = re.compile(r"(?im)^\s*(?:\*\*)?\s*claim\s*\d+\s*[:.]?\s*(?:\*\*)?[ \t]*")
# A claim opens with an article and reaches a transition phrase. Searching for
# that pair is what separates the claim from the base model's preamble, which
# opens with "Based on the provided drawings...".
_CLAIM_START = re.compile(r"(?ms)^[ \t]*(?:A|An)\s+.*?\b(?:comprising|consisting|including)\b")
_ARTICLE_LINE = re.compile(r"(?m)^[ \t]*(?:A|An)\s+\S")
# Only cut on a marker that FOLLOWS the claim - the base model also emits a
# horizontal rule before it.
_NUMERAL = re.compile(
    r"\s+\d{1,4}\b(?!\.\d)(?=\s*(?:[;,.]|and\b|the\b|to\b|such\b|wherein\b|is\b|are\b"
    r"|that\b|which\b|for\b|in\b|of\b|connected\b|disposed\b|extending\b|having\b))"
)
_NOTES = re.compile(
    r"\n\s*(?:\*{3,}|-{3,}|_{3,}|#{1,6}\s|Drafting\s+Notes|Notes\s*&|Rationale\b|Notes\s*:"
    r"|작성\s*노트|참고\s*사항|설명\s*:)",
    re.IGNORECASE,
)

_HANGUL = re.compile(r"[가-힣]")
# 줄 앞 공백만 먹습니다. \s* 로 두면 앞의 빈 줄까지 삼켜 머리말과 청구항이 한
# 문단으로 붙고, 그러면 아래 문단 단위 머리말 제거가 통째로 무력해집니다.
_LABEL_KO = re.compile(r"(?m)^[ \t]*(?:\*\*)?[ \t]*(?:청구항|청구범위)[ \t]*\d*[ \t]*[:.]?[ \t]*(?:\*\*)?[ \t]*")
# 한국어 청구항은 관사가 아니라 끝맺음으로 알아봅니다. 머리말이 붙는다면 별도
# 문단으로 오므로, 끝맺음을 담은 첫 문단부터가 청구항입니다.
_CLAIM_BLOCK_KO = re.compile(r"(?:포함하는|구성되는|특징으로\s*하는)")
# 한국어에서 도면 부호는 '하우징 10은'처럼 조사가 바로 붙습니다. 수량은 단위가
# 띄어쓰기 뒤에 오거나('100 pL') 수관형사와 붙으므로('3개') 걸리지 않습니다.
# 실측에서는 모델이 한국어로는 부호를 아예 쓰지 않았습니다 - 이건 그물입니다.
# 조사는 함께 집어삼킵니다. 앞말의 받침에 맞춰 다시 골라야 하기 때문입니다.
_NUMERAL_KO = re.compile(
    r"(?P<prev>[가-힣])\s+\d{1,4}(?![.\d])"
    r"(?:(?P<part>은|는|이|가|을|를|과|와|의|및)|(?=\s*[;,.]))"
)

# 받침이 있으면 앞쪽, 없으면 뒤쪽. 숫자를 지우면 앞말의 받침이 달라지므로
# '부재 20을' 은 '부재을' 이 아니라 '부재를' 이 되어야 합니다.
_PARTICLE_PAIRS = {
    "은": ("은", "는"), "는": ("은", "는"),
    "이": ("이", "가"), "가": ("이", "가"),
    "을": ("을", "를"), "를": ("을", "를"),
    "과": ("과", "와"), "와": ("과", "와"),
}


def _drop_numeral_ko(m: re.Match) -> str:
    prev, part = m.group("prev"), m.group("part")
    if not part:
        return prev
    pair = _PARTICLE_PAIRS.get(part)
    if pair is None:  # '의', '및' 은 형태가 하나뿐입니다.
        return prev + part
    has_batchim = (ord(prev) - 0xAC00) % 28 != 0
    return prev + (pair[0] if has_batchim else pair[1])


def detect_lang(text: str) -> str:
    """출력의 언어. 한글이 섞여 있으면 한국어로 봅니다."""
    return "ko" if _HANGUL.search(text) else "en"


def _claim_start_ko(out: str) -> int | None:
    """한국어 청구항이 시작하는 위치. 못 찾으면 None.

    청구항은 끝맺음('~를 포함하는 것을 특징으로 하는 장치.')으로 알아보고,
    그 끝맺음이 든 문단의 처음을 청구항의 시작으로 봅니다.
    """
    offset = 0
    for block in out.split("\n\n"):
        if _CLAIM_BLOCK_KO.search(block):
            return offset + (len(block) - len(block.lstrip()))
        offset += len(block) + 2
    return None


def sanitise(text: str, lang: str = "auto") -> tuple[str, list[str]]:
    """Trim the wrapper the base model adds around the claim itself.

    Order matters: the preamble is located and dropped before any trailing
    section is cut, because the model puts a horizontal rule on both sides of
    the claim and cutting on the first one would discard the claim.

    lang picks the ruleset. The English rules key off an article and
    'comprising', which no Korean claim contains, so running them on Korean
    output finds nothing and reports a failure that is not one. The default
    reads the language off the output, which keeps every existing caller
    correct without passing anything.
    """
    removed: list[str] = []
    out = text.strip()
    if lang == "auto":
        lang = detect_lang(out)

    if _FENCE.search(out):
        out = _FENCE.sub("", out).strip()
        removed.append("code fence")

    label = _LABEL_KO if lang == "ko" else _LABEL
    stripped = label.sub("", out)
    if stripped != out:
        out, _ = stripped.strip(), removed.append("claim label")

    # Locate where the claim actually begins.
    if lang == "ko":
        start = _claim_start_ko(out)
    else:
        m = _CLAIM_START.search(out) or _ARTICLE_LINE.search(out)
        start = m.start() if m else None
    if start is not None:
        if start > 0:
            removed.append("preamble")
        out = out[start:]
    elif out:
        removed.append("no claim opening found - output returned as-is")

    # The notes cut runs while the horizontal rules are still intact: stripping
    # emphasis first would turn "***" into "*" and leave it behind.
    cut = _NOTES.search(out)
    if cut:
        out = out[: cut.start()]
        removed.append("trailing notes section")

    if "**" in out or "__" in out:
        out = out.replace("**", "").replace("__", "")
        removed.append("emphasis markers")

    # The model complies with "no reference numerals in parentheses" by writing
    # them bare ("a housing 10"), so strip a bare integer only where a numeral
    # can appear: directly before punctuation or a structural word. A real
    # quantity is followed by its unit ("100 pL") and is left alone.
    if lang == "ko":
        stripped = _NUMERAL_KO.sub(_drop_numeral_ko, out)
    else:
        stripped = _NUMERAL.sub("", out)
    if stripped != out:
        out = stripped
        removed.append("reference numerals")

    return out.strip(), removed
