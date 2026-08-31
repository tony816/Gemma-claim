"""Offline checks of the claim prompts and the output sanitiser.

Pure string work: no GPU, no network, no model. The Korean cases are anchored on
output actually returned by the endpoint - see run_artifacts and serving/README.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "serving"))

from claim_prompt import detect_lang, sanitise, system_prompt  # noqa: E402

failures: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got : {got!r}")
        print(f"        want: {want!r}")
        failures.append(name)


def main() -> int:
    # --- Korean -----------------------------------------------------------
    # Returned by the endpoint for two drawings carrying reference numerals.
    # Nothing about it is a violation, so nothing may be touched.
    real = ("입구 및 출구를 갖는 하우징, 상기 하우징 내부에 서로 연결되어 배치된 "
            "제1 구성요소 및 제2 구성요소를 포함하는 것을 특징으로 하는 장치.")
    check("clean Korean output is left alone", sanitise(real), (real, []))

    wrapped = ("제공된 도면을 바탕으로 작성한 청구항입니다.\n\n"
               "청구항 1: 하우징을 포함하는 것을 특징으로 하는 장치.\n\n"
               "설명: 도면 1의 구조를 반영했습니다.")
    check("Korean preamble, label and notes are cut",
          sanitise(wrapped)[0], "하우징을 포함하는 것을 특징으로 하는 장치.")

    check("Korean numerals are stripped",
          sanitise("하우징 10과 상기 하우징 10의 내부에 배치된 부재 20을 "
                   "포함하는 것을 특징으로 하는 장치.")[0],
          "하우징과 상기 하우징의 내부에 배치된 부재를 포함하는 것을 특징으로 하는 장치.")

    # Dropping the numeral changes whether the preceding word ends in a
    # consonant, so the particle has to be re-chosen: 부재 20을 -> 부재를.
    check("particles agree with the word left behind",
          sanitise("하우징 10과 이격된 부재 20이 배치되고, 덮개 30의 아래에 "
                   "판 16을 포함하는 장치.")[0],
          "하우징과 이격된 부재가 배치되고, 덮개의 아래에 판을 포함하는 장치.")

    qty = "100 pL 의 시약과 3개의 챔버를 포함하는 것을 특징으로 하는 장치."
    check("Korean quantities survive", sanitise(qty), (qty, []))

    # --- English regression ----------------------------------------------
    check("English numerals are stripped",
          sanitise("A cartridge comprising a body 10 connected to a plunger 20, "
                   "and a foil.")[0],
          "A cartridge comprising a body connected to a plunger, and a foil.")

    en_qty = "A device comprising a chamber holding 0.1 ml and a channel of 100 pL."
    check("English quantities survive", sanitise(en_qty), (en_qty, []))

    # --- Language selection ----------------------------------------------
    check("Korean is detected", detect_lang(real), "ko")
    check("English is detected", detect_lang(en_qty), "en")
    # Forcing the wrong ruleset must report a miss rather than mangle the text.
    check("English rules on Korean report a miss",
          sanitise(real, "en")[1], ["no claim opening found - output returned as-is"])

    check("ko prompt asks for Korean", "한국어로 작성" in system_prompt("ko"), True)
    check("en prompt is the English one",
          system_prompt("en").startswith("You are a patent attorney"), True)
    check("unknown language falls back to English",
          system_prompt("fr"), system_prompt("en"))

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
