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
    r"\n\s*(?:\*{3,}|-{3,}|_{3,}|#{1,6}\s|Drafting\s+Notes|Notes\s*&|Rationale\b|Notes\s*:)",
    re.IGNORECASE,
)


def sanitise(text: str) -> tuple[str, list[str]]:
    """Trim the wrapper the base model adds around the claim itself.

    Order matters: the preamble is located and dropped before any trailing
    section is cut, because the model puts a horizontal rule on both sides of
    the claim and cutting on the first one would discard the claim.
    """
    removed: list[str] = []
    out = text.strip()

    if _FENCE.search(out):
        out = _FENCE.sub("", out).strip()
        removed.append("code fence")

    stripped = _LABEL.sub("", out)
    if stripped != out:
        out, _ = stripped.strip(), removed.append("claim label")

    # Locate where the claim actually begins.
    m = _CLAIM_START.search(out) or _ARTICLE_LINE.search(out)
    if m:
        if m.start() > 0:
            removed.append("preamble")
        out = out[m.start():]
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
    stripped = _NUMERAL.sub("", out)
    if stripped != out:
        out = stripped
        removed.append("reference numerals")

    return out.strip(), removed
