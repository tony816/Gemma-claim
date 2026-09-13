# Final reinforcement-learning material release

User objective (2026-09-08): finish **reinforcement-learning materials** for
continuing the existing fine-tuned Gemma v2. No GPU, model training or RunPod
inference calls. This specification clarifies the older SFT-oriented draft:
reference answers are reward supervision/calibration material. Producing an SFT
training run is not the objective. Runtime/trainer readiness is a separate
future GPU gate and cannot substitute for, or prevent honest reporting of,
material completion.

## Coverage and minimum material requirements

1. Reviewed, page-addressable KR and US case evidence, aiming at 20 distinct
   reasoned case families per jurisdiction. Rule 36, procedural orders and
   corrections are inventoried but cannot supply invented merits reasoning.
   Full-corpus ingestion is not a claim of full-corpus relevance or review.
2. At least 40 reviewed drawing-grounded claim-set reward references: 32 from
   frozen train and 8 from frozen validation, with independent and supported
   dependent claims. Additionally freeze 8 new claim-set evaluation examples
   outside the old train/validation/test patent families. Preserve the old 75
   test records and all frozen release bytes.
3. Case-grounded advisory RL tasks covering issues, application to given facts,
   opposing arguments, drafting/review options, missing facts and citations.
   At least two distinct substantive advisory tasks per curated case. Partition
   case lineages before release: 14 train / 3 calibration / 3 test per country.
4. At least 100 substantive train preference comparisons and 20 held-out
   calibration comparisons, covering claim support, dependent limitations,
   citation support and advisory application. Controlled corruption unit-test
   fixtures do not count toward these minima.
5. Independent content review with named agent/reviewer identity and hashes of
   the reviewed inputs. Author-produced labels alone are not independent review.
   Blind judge calibration must meet >=85% pairwise agreement on the held-out
   calibration comparisons, with category-level errors recorded. This measures
   the specified judge protocol on this set, not all future models or matters.
6. A frozen release with prompts/images/retrieved evidence separated from reward
   references and evaluation labels, documented reward rubric and abstention
   rules, source/lineage manifest, integrity/leakage checks and processor probes.
   Material readiness must be computed from these checks, not hard-coded true.
7. Case-annotation comparisons on at least 8 train, 4 calibration and 4 new-test
   drawing episodes. Supply only approved, unrelated TRAIN case cards matching
   the requested jurisdiction. Compare substantive application of real supplied
   cards to the same claim text. Review extensions independently with bindings
   to both current drawing records and case records. This pilot teaches supplied
   evidence attribution; it does not establish attribution inside model weights.

## Authored curation interchange

Authors write UTF-8 JSON arrays or JSONL in `rl_materials/curation/`. Source
extractions and bulky private material go under `data/case_rl/curation_<lane>/`.
Do not edit the original Drive corpus or frozen source release. Author outputs
must remain `author_complete_pending_review` until another reviewer verifies
the relevant evidence. Do not fabricate source links, page numbers, patents,
review identities, current legal status, claims or observation evidence.

### Case records (`kr_cases.jsonl`, `us_cases.jsonl`)

Required: `case_id`, `jurisdiction`, `case_name`, `docket`, `court`,
`decision_date`, `pdf_path`, `source_sha256`, `case_patent_ids`, `lineage_notes`,
`history_scope`, `evidence_spans`, `principles`, `advisory_episodes`,
`author`, `review_status`.

- `history_scope`: object with `mode` (`historical_as_of_decision` or
  `current_sources_checked`), `as_of_date`, `sources`, and `limits`. Historical
  status is explicit and cannot support a current-law conclusion. Lack of later
  history is recorded, not silently represented as good law.
- `evidence_spans`: `{id, pdf_page, quote, speaker, section}`. Quotes are exact
  text after whitespace normalization, from the full relevant PDF page actually
  read. Distinguish court holding, party submission and quoted other authority.
- `principles`: `{id, statement, evidence_ids, scope, non_rules}`. Avoid blanket
  word bans and equating case outcome with the quality of any proposed claim.
- `advisory_episodes`: at least two `{episode_id, question, scenario_facts,
  scenario_origin, temporal_scope, reference_answer, required_points,
  forbidden_claims, evidence_ids, inferior_answer, preference_reason}` records.
  Use realistic authored hypotheticals, clearly labeled. The reference answer
  must explicitly connect facts, case reasoning and limitations, and include
  counterarguments/missing facts when material. Inferior answers must contain
  a plausible, substantive application error rather than broken formatting or
  an obviously invented citation. Korean answers are preferred, including US
  case discussion with US jurisdiction made explicit.

### Drawing records (`claimsets.jsonl`)

Required: `record_id`, `source_split`, `images` (paths in original order),
`observations` (`image_index`, `figure`, `visible`), `claims` (strings),
`parents` (arrays), `claim_evidence` (observation indices for each claim),
`abstentions`, `reward_required_points`, `reward_forbidden_claims`,
`alternatives` (two substantive alternatives for train, at least one for
validation, each with full `claims`, `parents`, `defect`, `preference_reason`),
`author`, `review_status`. One independent claim and up to three dependent
claims; fewer when evidence does not support more. No drawing reference
numerals in claim text. Do not read original frozen targets to author these
references. The four earlier seed drafts had target exposure and need separate
re-review rather than being called blind.

## Invariants

- Start policy remains adapter v2 at
  `cc04579366bbb88663029d53b2aafcc84159589e`; no adapter/model change in this task.
- Drawing tasks never receive original target claims/specifications or case
  discussion of the target patent. Only unrelated approved general principles.
- Advisory tasks may receive the explicitly provided hypothetical facts and
  selected case excerpts. The drawing-only boundary is not misapplied to text
  advisory questions. Neither task receives its own reward answer or label.
- Case annotations identify actually provided evidence or a separate post-hoc
  review. They do not assert causal attribution inside trained weights.
- No passing syntax/hash check counts as a semantic reward or legal judgment.
- Gold/evaluation references stay out of policy prompts. A future judge receives
  them through a separate reward channel.
- The readiness report distinguishes material completion, model capability
  (unmeasured until training/evaluation), and GPU authorization (false here).
