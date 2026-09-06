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

    # --- Parenthesised reference numerals ---------------------------------
    # 8.3% of the Korean training targets carry them ('플랫폼(2)',
    # '시료 전처리 장치(100)'), so the fine-tuned model writes them too and a
    # claim must not. Everything else that lives in parentheses in the approved
    # references has to survive: enumeration markers and English glosses.
    for name, lang, text, should_change in [
        ("ko glued numeral is cut",
         "ko", "펌버주입구(131a, 141a)와 펌버배기구(131b)를 포함하는 장치.", True),
        ("ko enumeration marker survives",
         "ko", "(1) 멤브레인 패드; (2) 결합체; 를 포함하는 것을 특징으로 하는 키트.", False),
        ("ko English gloss survives",
         "ko", "버퍼 용액을 저장하는 챔버(chamber)와 유로(channel)를 포함하는 장치.", False),
        ("en loose numeral is cut",
         "en", "An apparatus comprising a housing (10) and a sensor (20a, 20b).", True),
        ("en enumeration marker survives",
         "en", "An apparatus comprising: (1) a housing; and (2) a sensor.", False),
    ]:
        check(name, sanitise(text, lang)[0] != text, should_change)

    # The numeral sits between the noun and its particle, so cutting it changes
    # which allomorph is correct: 플랫폼 has a final consonant, 유로 does not.
    check("particle is re-picked after the numeral is cut",
          sanitise("플랫폼(2)를 포함하고 유로(110)을 구비하는 장치.", "ko")[0],
          "플랫폼을 포함하고 유로를 구비하는 장치.")

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
