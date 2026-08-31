---
name: runpod-ops
description: Operating discipline for renting GPUs to train or serve models — on RunPod, Lambda, Vast, or any pay-by-the-hour provider. Use this skill whenever the user is about to fine-tune, train, or deploy a model on rented GPUs; is starting, stopping, or debugging a GPU pod or a serverless inference endpoint; mentions RunPod, worker-vllm, vLLM, a network volume, cold starts, or workers that are UNHEALTHY or crash-looping; asks what a training run will cost or why their GPU bill is high; or is deciding whether a fine-tune is worth running or whether one that finished is any good. Trigger it even when the user only says "let's train this model" or "my endpoint is broken" without naming a provider, because the failures this prevents — spending the whole budget before ever looking at a generation, and leaving a pod billing overnight — do not announce themselves.
---

# Renting GPUs without paying the same tuition twice

This skill exists because a real project spent about $21 to produce a
fine-tuned model that was **worse at its task than the base model it started
from**, and did not find out until every dollar was spent. Nothing here is
hypothetical. Each rule below is attached to what it actually cost.

## The rule everything else follows from

**A rented GPU is not a development environment.**

In that project, every pipeline bug was discovered for the first time on a
$1.50–2.00/hr machine, one restart cycle at a time, and the pod stayed alive
between attempts. That is where $10.33 of the $21 went — a pod nobody turned
off, because the session that would have stopped it ended first.

The bugs themselves needed no GPU at all:

| Bug | Where it was found | Where it could have been found |
|---|---|---|
| Chat template emitted a token the trainer did not expect | GPU restart cycle | `AutoProcessor.from_pretrained` — tokenizer files only |
| LoRA target suffixes also matched the vision tower | GPU restart cycle | `init_empty_weights` + `from_config` — meta device, no VRAM |
| `TrainingArguments` argument removed by a library release | GPU restart cycle | `inspect.signature` — pure Python |

A model skeleton on the meta device has every module, every name, and every
shape, and downloads no weights. That is enough to catch the whole class.

## Before starting a GPU, in this order

**1. Rehearse the pipeline weightlessly.** Build a script that runs the real
data loading, template rendering, loss masking, adapter targeting, and trainer
construction against the real tokenizer and a meta-device model. It should exit
non-zero on any failure. Design detail that matters more than it sounds: when a
check cannot run, report it as *skipped* and count it separately from passed —
a harness that quietly passes what it never checked is worse than no harness,
because it converts an unknown into false confidence. See
`references/rehearsal.md` for the gates worth building and how to prove each
one can actually fail.

**2. Score the baseline you have to beat.** Run the *prompted* base model over
the eval split and score it with the same metrics the fine-tune will be scored
with. In that project nobody did this, so there was never a moment where not
training was the obvious call — and the base model turned out to read the
inputs correctly while the fine-tune did not. This costs cents. Skipping it is
how you spend a budget discovering something a prompt would have told you.

**3. Arm a guard that can stop the pod without you.** `scripts/pod_guard.sh`
stops a pod on GPU idle or a wall-clock deadline. Idle is the trigger that
matters: a forgotten pod runs at ~0% utilisation and bills at full rate. Start
it from the pod's boot script, not by hand, because the run you forget to guard
is the one that costs you.

## While it runs, and after

**Training loss is not evidence that the model works.** It is teacher-forced —
computed with the reference already in the context. A model can drive it down
by learning what the target text *looks like* while losing the ability to
condition on the input at all. In that project validation loss fell 3.50 → 1.82
and the model produced a correct-subject claim in **0 of 12** test cases.

The signal that catches this is a **reference-overlap metric falling while loss
falls** — chrF there, 28.9 → 21.0. Loss down and chrF down together means the
model is learning the output distribution, not the mapping. Track both, and
treat their divergence as a stop condition rather than a curiosity.

**Read three free-running generations at step 0 and again early.** Mode collapse
is obvious to the eye — repeated openings, incremented ordinals padding out
length, the same answer for different inputs — long before any aggregate metric
looks wrong. This costs a minute and is the highest-yield check in the run.

**Small datasets usually do not need a fine-tune.** That project had 91 training
records against an adapter on every attention and MLP projection of a 31B model.
It learned the output format completely and lost everything else. Under a few
hundred examples, expect the format to transfer and the task not to, and put the
effort into the prompt first.

## Reading the bill

Guessing is how the same project reported the wrong balance twice. Query the
provider's billing API by hour and read it; do not reconstruct it by arithmetic.

The pricing shapes differ in ways that decide architecture:

- **Pods** bill per hour of existence, working or not. This is where money
  actually leaks.
- **Serverless** bills only while a container is up. An endpoint at min-workers 0
  costs nothing idle — measured across hours: zero, disk included.
- **Network volumes** bill monthly regardless of use, and deleting one can break
  things that silently depend on its mount path. See `references/serving.md`.

For occasional inference, serverless at min-workers 0 beats a pod by a wide
margin even though every cold start reloads the weights. For a debugging
session, a pod with a guard beats serverless. Choose per workload.

## When a serverless endpoint misbehaves

Workers going UNHEALTHY with **no container output at all** is a startup failure,
not a model or GPU failure, and the cause is usually a path or a credential
rather than anything interesting. `references/serving.md` has the specific traps
— a deleted volume still named in an env var, request shapes that silently drop
half the parameters, and how to tell a real crash-loop from a scheduling wait.
Read it before forming a theory; several hours went into three confident wrong
diagnoses that it would have short-circuited.

## References

- `references/rehearsal.md` — the weightless gates, and how to verify a gate can fail
- `references/serving.md` — serverless endpoints, worker-vllm, crash-loop diagnosis
- `references/billing.md` — pricing shapes, measured numbers, how to check spend
- `scripts/pod_guard.sh` — idle and deadline shutdown, ready to drop into a boot script
