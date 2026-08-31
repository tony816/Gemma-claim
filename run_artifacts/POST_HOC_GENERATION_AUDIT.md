# Post-hoc generation audit — free-running outputs on the test split

Written after training completed, from `test_predictions.jsonl` in the published
adapter repo. No new compute was spent to produce this; it re-reads artifacts the
run itself emitted.

## Why this audit exists

`metrics.json` reports the tuned model beating the base model on the headline
number: validation loss 3.497 -> 1.821, test loss 2.885 -> 1.621. That number is
teacher-forced. It measures how well the model predicts the reference *given the
reference prefix*. It says nothing about what the model emits when it has to
generate a claim on its own.

Reading the free-running generations tells a different story, and the aggregate
metrics already carried the warning:

| metric | base | tuned | direction |
|---|---|---|---|
| test loss | 2.885 | **1.621** | improved |
| test ROUGE-L F | 0.121 | **0.162** | improved |
| test chrF | **28.85** | 21.04 | **regressed** |
| test mean words | 369.6 | 158.1 | shorter |

chrF falling while loss falls is the signature of a model that learned the
*unconditional* distribution of patent-claim English rather than the mapping from
drawings to a claim. ROUGE-L rose because generic claim boilerplate shares
function words and legal connectives with real claims.

## Per-record verdict (test split, n=12)

Judged on whether the generated claim is about the same apparatus as the
reference.

| # | record | reference subject | tuned output subject | grounded |
|---|---|---|---|---|
| 1 | EP0502691A2_claim1 | liquid control region / capillary gap at cartridge edge | "biological fluid collection device" | no |
| 2 | EP3360976A1_claim1 | PCR by thermal convection; brackets, temp sensor, light source | "biological fluid separation apparatus", pore sizes | no |
| 3 | KR20160128480A_claim1 | 표적물질 검출카트리지 분석장치 | correct domain, correct language, 28 words | partial |
| 4 | US20130165643A1_claim1 | nucleic acid extraction kit; wells, filter cartridge | "blood sampling device", puncturing element | no |
| 5 | US20140120585A1_claim1 | tube with seven ordered oil/liquid plugs | "digital PCR system", PCR chip | no |
| 6 | US20140120585A1_claim12 | same, plus magnetic force device | "digital PCR system", oil droplets | no |
| 7 | US20140370540A1_claim9 | ionic-liquid immersion + drying for charged particle beam | "sample prep for FIB milling", layer stack | partial |
| 8 | US20180172676A1_claim1 | automated microscope; heat sink, microprocessor | "biological fluid analysis system" | no |
| 9 | US20180172676A1_claim18 | XYZ stage, imaging, autofocus, neutrophil count | "biological fluid analysis system", three identical valves | no |
| 10 | US20180172676A1_claim7 | automated microscope; vibration isolators, XYZ stage | "biological fluid analysis system", three identical sensors | no |
| 11 | US20190151844A1_claim29 | amplification/reagent/detection modules, capture probes | "biological fluid processing system", three identical pumps | no |
| 12 | US20190307383A1_claim1 | small-volume mixing-enhanced microfluidic container | "blood sampling device", lancet and spring | no |

**Grounded 0/12. Partial 2/12. Wrong subject 10/12.**

## Two failure modes

**Mode collapse onto memorised templates.** The generations concentrate on a
handful of openings: "A biological fluid {collection,separation,analysis,
processing} system comprising: a housing having a first opening and a second
opening", "A blood sampling device comprising: ... puncturing element ...
spring", "A digital PCR system comprising: a PCR chip". Records 8, 9 and 10 are
three different claims of the same patent — a heat sink and microprocessor, a
vibration-isolated stage, and neutrophil counting respectively — and all three
received substantially the same answer. The model is emitting a prior, not
reading its inputs.

**Degenerate enumeration.** Where the template runs out, the model pads by
repeating a component with an incremented ordinal: first/second/third valve
(rec. 9, repetition 0.282), first/second/third pump (rec. 11, 0.221). Record 1
degenerates completely — 728 words of nested "first portion / second portion /
first barrier / second barrier", repetition 0.408, never terminating inside the
756-token budget.

## Image grounding was lost, not gained

The base model's generations cite figure numerals from the drawings (4010, 4600,
4070/4080 in rec. 11; 112/212/210/118/182 in rec. 12) and infer context that is
only visible in the images — rec. 8 and 10 identify bovine milk/colostrum
diagnostics from UI screenshots in the figures. The tuned model cites no
numerals and describes no drawing-specific structure in any of the 12 records.

The vision tower was frozen, so the encoder is unchanged. What regressed is the
language tower's use of the visual evidence: 410 LoRA modules across all 60
layers, r=16, trained on 91 records, had ample capacity to fit the surface form
of claim English while discarding conditioning on the image tokens.

## Cause

91 training records against a 31B model, with LoRA applied to every attention and
MLP projection in all 60 layers. The adaptation was far too strong for the
evidence available. The formatting objective — bare claim text, no markdown, no
drafting commentary, single sentence opening with an article — was learned
completely and is the one clear success. The content objective was not learned.

## What this changes

Serving this adapter would produce well-formatted, confident, wrong claims. For
patent drafting that is worse than an obviously-unhelpful output, because the
failure is not visible in the form. The planned INT4 quantisation and endpoint
swap were therefore not carried out; the money is better spent elsewhere.

The dataset is a frozen reference release and was not touched by this audit.
