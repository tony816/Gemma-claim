# The weightless rehearsal

A script that runs the training pipeline with no GPU, no weights, and no
billing. Its job is to move every failure that does not require a GPU off the
GPU, where each one costs a restart cycle at $1.50–2.00/hr.

## Why it can work at all

Three facts make almost the whole pipeline checkable for free:

- `AutoProcessor.from_pretrained` downloads tokenizer and processor config only
  — tens of MB, no weights. Everything about templating and tokenisation is
  therefore reproducible on a laptop.
- `accelerate.init_empty_weights()` around `AutoModelForX.from_config(cfg)`
  builds the complete module tree on the meta device: every module name, every
  `in_features`, zero memory, zero weight download. Anything that inspects
  structure — adapter targeting, layer freezing, parameter counting — works.
- Library-compatibility questions are `inspect.signature` calls.

## Gates worth building

**Versions.** Report the installed versions and fail on anything missing. Cheap,
and it makes every later failure interpretable.

**Trainer arguments.** Compare the argument names the run depends on against the
installed `TrainingArguments.__init__` signature, honouring known renames. A
library release that drops an argument would otherwise silently change training.
Keep the list of required names at module scope in the training code and import
it here, rather than copying it — a copy drifts, and a drifted copy passes.

**Template renders a strict prefix.** Render the generation prompt and the full
conversation, and assert the prompt is a literal prefix of the full sequence.
Assistant-only loss masking depends on that property. Chat templates break it in
non-obvious ways: Gemma 4's generation prompt opens an empty thought channel
that the full rendering does not contain, so the naive `full[len(prompt):]` is
wrong by a few tokens and the mask silently slides.

**Loss lands where you think.** Encode one record and inspect the labels:
supervision must be one contiguous span, must not start at token 0, and must not
be empty. Each of those is a different real bug — prompt tokens being trained
on, a mask that never engaged, a record whose target is blank.

**Adapter targets stay inside the language tower.** Run the real target-discovery
function and assert no chosen name contains `vision`, `audio`, `embed_tokens`, or
`lm_head`, and that every name is fully qualified. PEFT matches a *bare suffix*
everywhere it occurs, so `q_proj` also selects the vision tower — which may be
wrapped in a class PEFT cannot adapt, turning into a crash minutes into a run.

**Dataset contracts.** Every record has a non-empty target; every referenced
image exists; image order is preserved. Then report the split sizes, and warn
when the training split is small — a few hundred records is where format
transfers and task does not.

## Skips are not passes

Some gates need network or credentials. When one cannot run, report it as
skipped, count skips separately from passes, and say in the summary that the run
is only as verified as the gates that executed. A fallback — a fake processor, a
toy module tree — can still check the *logic*, and should say exactly that
rather than reporting a pass it did not earn.

This matters more than it looks. The purpose of the harness is to convert
unknowns into knowns. A skip silently rendered as a pass converts an unknown
into a false known, which is worse than leaving it unknown, because it removes
the reason to look.

## Prove each gate can fail

A gate that cannot fail is decoration. After building them, inject a fault per
gate and confirm it is caught: an adapter target inside the vision tower, a bare
suffix, a required trainer argument that does not exist, a prompt that is not a
prefix, an empty target. Monkeypatching the underlying function in a short test
script is enough. Run this once when the harness is written and again whenever a
gate is edited.
