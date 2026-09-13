# Case-grounded continuation of Gemma claim v2

The first reviewed **reinforcement-learning material package is frozen** at
`data/case_rl/releases/rl-pilot-27fc35d39288134b`. See [the Korean result guide](<C:/Users/VIEW LIFW/Projects/Gemma-claim/rl_materials/RELEASE_RESULT.md>)
and `MATERIALS_SPEC.md` / `REWARD_PROTOCOL.md` for its scope and contracts.
It contains 128 episodes and 176 preference comparisons, 40 case records, 48 claim sets,
and 16 case annotations. Independent source/content review, code review, QA,
24/24 blind comparison agreement and 128/128 CPU encoding checks passed.

This is a selected pilot, not full-corpus semantic review. No GPU or model training
was run. Reference answers are reward supervision/calibration material. The
working `data/case_rl/release_candidate/` directory is not the immutable release.

User decision, 2026-09-07: finish training-material preparation before starting
any GPU. Continue the existing v2 adapter; do not substitute a new base-only
fine-tune. Extend output to one independent claim and up to three dependent
claims, with separately inspectable case annotations. Three is an initial
preparation limit, not a requirement to invent three supported limitations.

## Model and training sequence

- Base: `google/gemma-4-31B-it` at the previously validated training revision.
- Initial trainable adapter: `Mepeng22/gemma-4-31b-claim-lora-v2`, revision
  `cc04579366bbb88663029d53b2aafcc84159589e`.
- Preserve v2. Publish any successful continuation under a new version only.
- Historical training records say base revision `main`; resolve and pin the
  actual base revision before a new run. A cached processor revision alone is
  not proof of the historical weight revision.
- Prepare claim-set episodes, source annotations, reward criteria and reviewed
  preference comparisons. Existing independent-claim targets are not
  dependent-claim gold. Any future format warm-up is a separate training choice.
- Later perform actual reward-based continuation (e.g. GRPO) on the adapted
  model. DPO is an optional offline preference method, not a claim that online
  reinforcement learning has already happened. Select the trainer only after
  confirming Gemma 4, multiple images, PEFT, masking, memory and reward support.
- If using a KL reference, freeze the actual pre-RL adapted policy as the
  reference. Merely disabling the v2 adapter would select the base model and
  would not implement that reference policy.
- The frozen v2 train/validation/test split is preserved. The 75 test records
  are not downloaded or opened by preparation. Training-source prompts and
  validation-source prompts are exported separately, never pooled.
- The old 75-record test measures the existing independent-claim task only.
  Dependent claims and annotations require a separately frozen claim-set test
  with new reviewed targets and family-disjoint examples; do not present the
  old single-claim reference metric as proof of these new capabilities.

## Attribution contract

Training weights do not identify which case caused a particular generation.
An annotation means either `provided_during_drafting` or `posthoc_review`.
It never means verified causal attribution to training examples.

At inference a retriever selects approved, general drafting-principle cards.
Each card includes jurisdiction, docket, court, decision date, source hash,
PDF page, excerpt and scope limitations. The model may select only supplied
card IDs. The application resolves court/docket/date/page from the card store;
free-form model-generated case metadata must not become authoritative.
For the first version annotations refer to whole claims, not character spans.

Claim text stays separate from annotations. Citation counts are not rewarded.
US and KR law must not be treated as interchangeable. Comparative materials
must be explicitly marked comparative. A case's later procedural history and
legal applicability need review; file names and precedential labels alone are
not legal validation. Raw case PDFs/Markdown never become generator prompts.

## Inputs and outputs

Input: approved drawing images in their original order, requested jurisdiction,
output language, desired dependent-claim ceiling, and approved general case
cards when available. This is an intentional new input contract for v3; retain
the existing single-claim v2 serving mode until the new mode is trained and
evaluated. Never retrieve the target patent's original claims/specification,
hidden source-oracle pages, or its litigation records as input evidence.

Output JSON:

```json
{
  "claims": [
    {"number": 1, "kind": "independent", "depends_on": [], "text": "..."},
    {"number": 2, "kind": "dependent", "depends_on": [1], "text": "..."}
  ],
  "annotations": [
    {"claim_number": 2, "card_id": "KR-2005HEO7354-STRUCTURE",
     "mode": "provided_during_drafting", "application": "..."}
  ],
  "abstentions": []
}
```

Dependent claims refer to an earlier claim and add a supported limitation;
they do not replace or negate the parent's limitations. Initial examples use
single dependencies. Claim numbers are allowed; drawing reference numerals
inside claim text remain excluded. Fewer supported dependent claims are better
than fabricated ones. Explicitly record insufficient drawing evidence.

## Material layers and completion gate

1. Source inventory: include every file; flag procedural-only, Rule 36,
   corrections, non-patent and uncertain documents. Inventory is not approval.
2. Page-addressable evidence: preserve source hashes and original-page numbers;
   extracted text may need OCR. Text search results are candidate evidence.
3. Reviewed principle cards: separate holdings, party arguments and commentary.
   Review jurisdiction, scope and subsequent history; do not infer legal
   validity from success/failure in the litigation.
4. Drawing annotations: components, relations and visible optional limitations
   with image-level references; leave invisible material/performance undecided.
5. Claim-set reward references, preference pairs and reward calibration examples:
   written from allowed inputs, independently checked and split by patent
   family/case lineage. Artificial corruption pairs are unit tests, not gold.
6. Offline validation: output graph, source resolution, unsupported structure,
   meaningful added limitations, semantic citation support, abstention,
   repetition, length and blind comparison against v2.

The CPU validator deliberately does not score legal correctness, drawing
grounding or semantic citation support from a regex. These require reviewed
labels or a calibrated judge. No passing JSON check can authorize training.
The report fails closed while such material is missing. Scripts in this
directory do not launch a GPU, call an inference endpoint, or modify serving.

## Proposed pilot acceptance criteria

- At least 40 reviewed drawing-grounded claim sets (32 train / 8 validation,
  respecting the source splits), with dependent claims where support exists.
- At least 100 reviewed preference pairs from train sources; no padding with
  trivial template variations. Hold out at least 20 independent calibration
  comparisons from validation sources.
- All cited cards source-verified and reviewed for scope/applicability. Explicit
  source-patent and case-family overlap checks; unresolved overlap blocks use.
- Judge pairwise agreement >= 85% on held-out calibration, with separate
  unsupported-feature, wrong-citation and invalid-dependency checks. A small
  pilot is not a population-level accuracy estimate.
- Material hashes fixed; real tokenizer/image/masking rehearsal passes with
  no skipped mandatory checks. Runtime and cost budget estimated before GPU.
- During the eventual run, inspect free generations at step 0 and each
  checkpoint. Stop if two of three fixed samples collapse or introduce
  unsupported dependent limitations; preserve v2 as rollback.

These are provisional engineering gates, not measured achievements.

## CPU commands

Run from the repository root. Fetch uses `huggingface_hub`; preparation uses
`pypdf`. The fetcher reads the existing HF token privately in-process and pins
the dataset revision. It fetches no model weights and never opens test JSONL.

```
python tools/fetch_case_rl_sources.py --all-images
python tools/prepare_case_rl.py --source "G:/내 드라이브/수/특허 판례"
python tools/audit_case_rl.py
python tools/probe_case_rl_processor.py
python -m unittest discover -s tests -p test_case_rl.py
```

Generated private data lives under ignored `data/case_rl/`, including the
inventory, SQLite text index, draft cards, prompt tasks and readiness report.
Only explicitly approved cards may be passed to the model; drafts are empty
retrieval context by default. The source Drive folder is read-only.
The processor probe needs the existing local Gemma 4 processor cache and the
project's torch/transformers environment. It loads no weights and validates
only four draft SFT encodings; it does not validate the GRPO training path.

`visual_seeds.json` contains four assistant-authored claim-set drafts grounded
in eight inspected training drawing pages, with twelve dependent claims.
They are not independently reviewed gold. Preparation exports eight controlled
corruption preference candidates as review examples, not as a measured reward
calibration set. The author saw allowed independent-target excerpts before
visual drafting; these examples must not be described as a blind evaluation.

References: https://huggingface.co/docs/trl/main/grpo_trainer and
https://huggingface.co/docs/trl/main/dpo_trainer (consulted 2026-09-07).
