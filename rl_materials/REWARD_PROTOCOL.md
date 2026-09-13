# Reinforcement-learning reward material protocol v1

The starting policy is the existing Gemma claim v2 adapter. These files prepare
episodes, source evidence, reward references, preference judgments and held-out
calibration. They do not run supervised training, reinforcement learning,
inference or a GPU. Choice and integration of a future online RL trainer remain
separate. Offline preference pairs may also support preference optimization;
their existence is not evidence that online RL occurred.

## Boundary between policy inputs and reward supervision

The policy receives the task, ordered images when relevant, requested language
and jurisdiction, and only the explicitly supplied source excerpts/cards. It
does not receive its reference response, required-answer points, inferior
response, preference reason, label, reviewer findings or case/patent family
audit corpus. Images are referenced by fixed path and content hash.

Reward records join to policy episodes by episode_id through a separate channel.
They contain an acceptable example, its image/source evidence, required and
forbidden inferences, and evaluated alternatives. A reference is one supported
answer, not a mandatory wording or a claim of novelty, validity or filing
readiness. Score a different supported construction fairly. Word overlap with
the reference and response length are not quality measures.

For drafting, evidence of a feature comes from the supplied images, including
legible labels/flowchart text inside them. Bibliographic titles, original source
claims, specification passages and the target patent's litigation records are
excluded. Case principles explain review constraints; they cannot establish an
otherwise invisible material, dimension, function, connection or performance.

For advisory episodes, the supplied hypothetical facts and selected case
excerpts are allowed text inputs. Distinguish assumed facts, court reasoning,
party positions and drafting suggestions. Historical-as-of tasks assess the
selected decision in its stated time frame. They cannot support current-good-law
claims or procedures that did not exist then. Current advice would require
updated retrieval and a separately verified task scope.

## Semantic scoring rubric

Each dimension is scored from0 to its maximum, with a brief reason pointing to
an image feature or evidence ID. Intermediate scores reflect material omissions
or uncertainty, not verbosity. The sum divided by10 is the proposed semantic
reward before caps. This rubric must be calibrated for the actual future judge;
no regular expression or schema validator implements these judgments.

| Task | Dimension | Max | Full-credit behavior |
| --- | --- | ---: | --- |
| Claim set | Image support | 4 | Components, connections and limiting features are supported by supplied images; embodiments are not silently mixed. |
| Claim set | Dependency and useful scope | 3 | One independent claim; each dependent adds a supported limitation while retaining its parent; fewer dependents when support is insufficient. |
| Claim set | Source annotation | 2 | Uses only supplied relevant evidence, accurately states its application and jurisdiction, and does not imply a case proves drawing features or causal influence in the weights. |
| Claim set | Uncertainty and usability | 1 | Specific limits where needed, no invented reference numerals, circular phrasing or needless repetitions. |
| Advisory | Application to facts | 4 | Connects the particular issue, stated facts and case reasoning; addresses the strongest relevant counterargument. |
| Advisory | Evidence fidelity | 3 | Actual supplied reasoning supports the conclusion; quotations, party arguments and judicial findings remain distinct. |
| Advisory | Jurisdiction and temporal scope | 2 | Correct jurisdiction and decision-era framing; no invented current status, remedy or unavailable procedure. |
| Advisory | Decision usefulness | 1 | Gives conditional options, consequences and missing facts that could change the answer; does not invent facts or definitive outcomes. |

Where a claim task supplies no case card, source annotation earns full credit
for an empty annotation array and no fabricated attribution. Where cards are
supplied, a justified decision that none supports a particular feature is valid.
Do not award points for citation count. Citation metadata is resolved from IDs
by the application, rather than trusted from free-form model-generated names.

Severe unsupported technical features, fabricated/misattributed citations,
contradictory dependency, or a materially wrong jurisdiction/era cap the reward
at0.25. Malformed or unparseable mandatory output can be rejected before judge
scoring, but a format pass never creates a positive semantic reward. The future
trainer must log the individual dimensions, cap reason, raw judge response,
judge configuration and source hash rather than just a scalar.

## Annotation meaning

An annotation is `provided_during_drafting` when the case evidence was in the
policy context, or `posthoc_review` when a later review attaches it. It identifies
the claim being discussed, supplied card/evidence ID and a bounded explanation.
It does not identify which training example caused a generation. Preserve this
distinction in training examples, reward criteria and eventual UI wording.

## Pairwise judge calibration

1. Freeze the calibration records and their independent semantic acceptance
   hashes before preparing the judge packet. Source-family/case-lineage splits
   must precede evaluation. Controlled syntax/citation corruption fixtures do
   not count toward the20 substantive comparisons.
2. Place the task, supplied evidence/images, neutral rubric and candidate A/B
   in a blind packet. Randomize side placement with a recorded seed. Gold side,
   author preference rationale, reference labels and review findings stay in
   a separate label file that the judge may not read. Do not identify one
   candidate as the reference or disclose its author.
3. An independent judge process reads only that packet and the allowed evidence.
   It returns A, B or tie/uncertain, with source-linked reasons and dimension
   scores. Abstentions/ties count as disagreements for the acceptance rate.
4. Compare the submitted answers to the frozen labels only after judging.
   Require>=85% agreement; report total and per-category denominators/errors.
   Record judge process identity and actual exposed configuration. Native agent
   calibration applies only to that documented protocol, not an unspecified
   future GPU/API judge backend or real-model output distribution.
5. If the rubric/labels are repaired after examining disagreement, retain the
   first result and disclose the calibration-set exposure. Use fresh held-out
   comparisons for an unbiased new acceptance claim. Do not repeatedly tune on
   the same20 and call the last result a blind test.

## Release and future use

The material release must include policy episodes, separate reward supervision,
train preferences, held-out calibration/test packets, reviewed source cards,
source/family/split manifests, integrity hashes, review decisions and a material
readiness report. Rejected or changed records do not inherit an earlier review.
No calibration/test source family enters training. The old75 test examples
remain untouched. The new8 external drawing examples are a small pilot with a
laboratory-device emphasis, not broad patent-domain accuracy evidence.

CPU processor probes verify the actual serialized messages/images can be
encoded without truncating required input. They neither validate a GRPO runtime
nor measure Gemma's resulting quality. Keep `gpu_allowed=false` and
`model_quality=unmeasured` for this material-preparation goal, including when
`materials_complete=true` eventually becomes justified by evidence.
