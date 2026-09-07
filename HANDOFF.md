# Handoff — state as of 2026-09-06, end of the v2 run

Read `CLAUDE.md` first; it is the durable doctrine and it was rewritten today
with what this run taught. This file is the live state: what exists, what was
decided, what is still open, and what to do next.

Everything below is verified in this session unless it says otherwise.

---

## 1. Goal and done criteria

Draft an independent physical apparatus claim, in the language of the source
patent, from that patent's drawings alone.

**Done** means a model that beats prompting on the held-out test split, scored
once, without degenerate output. Concretely:

- beats `tools/baseline.py`'s prompted base model on the 75 test records, on
  chrF read at β=1 with mean length reported beside it, and
- shows no repetition collapse above a few percent, and
- writes no drawing reference numerals.

**Not achieved.** The v2 adapter ties prompting on content and loses on
well-formedness. The test split is still unspent, because nothing yet deserves
to spend it.

---

## 2. Current state

### The v2 run finished

| | |
|---|---|
| adapter | `Mepeng22/gemma-4-31b-claim-lora-v2` — private, 498 MB, verified by re-download |
| trained from | `Mepeng22/gemma-4-31b-claim-lora` (v1), `is_trainable=True` |
| data | `finetune_multilingual_approved_20260902`, 554 train records |
| config | 3 epochs / 417 steps, LR 5e-5, r=16, 410 modules, bf16 |
| best checkpoint | **epoch 2**, eval_loss 1.3662 (epoch 3 rose to 1.3842) |
| peak VRAM | 77.04 GiB, H200 |
| cost | ~$12 measured; pod terminated, no pods, no volumes, serverless idle at $0 |

### The verdict

On the 59 Korean validation records, chrF read across β:

| system | β=0.5 | **β=1** | β=2 | mean words |
|---|---|---|---|---|
| base, no prompt | 7.14 | **9.56** | 14.82 | 344.8 |
| base, cut to reference length | 11.25 | **11.52** | 11.87 | 90.9 |
| **tuned, no prompt** | 14.16 | **11.85** | 10.81 | **105.3** |
| base + claim prompt | 15.47 | **11.78** | 9.86 | 41.8 |

References average 90.9 words. At balanced β=1 the three length-comparable
systems are within noise of each other.

| Korean, n=59 | tuned | base + prompt |
|---|---|---|
| reference numerals | 2/59 | **0/59** |
| repetition collapse | **9/59 (15%)** | 0/59 |
| well-formed | 34/59 | **58/59** |

**The fine-tune bought length calibration and nothing measurable in content, and
it pays with a 15% degeneration rate.** Teacher-forced loss fell 2.54 → 1.29
while content overlap did not move: the model learned the target's form and
length, not the mapping from drawings to claim. Same failure as the 91-record
run, much milder.

### Serving

Endpoint `fdiltabt78bogm` (`gemma4-31b-claim`) serves the **base** model —
`cyankiwi/gemma-4-31B-it-qat-AWQ-INT4`, A40 48 GB pool, min workers 0, $0 idle.
It does not serve the adapter. `serving/README.md` explains why the v1 adapter
was never served; that reasoning now needs updating, because v2 is not
mode-collapsed the way v1 was — it is merely not better than prompting.

**In flight:** the user asked to test v2 on serverless. See §5.

---

## 3. Decisions that are settled, and why

| Decision | Why | Do not re-litigate |
|---|---|---|
| Continue the v1 adapter rather than start a fresh LoRA | user's explicit call, twice | ✔ |
| Push to a **new** repo (`-v2`), never overwrite v1 | `push_hub.py`'s default target is the repo being resumed from; v1 is the fallback | ✔ |
| `EVAL_SPLITS=validation` first, test spent once at the end | evaluation costs more than training | ✔ |
| `MAX_NEW_TOKENS=1024` | measured: longest validation reference 513 tokens, longest test 894. `token_audit`'s 2401 comes from the longest *train* target, which is never generated | ✔ |
| `PUSH_MERGED=0` | a merged 62.5 GB upload is an hour of GPU time for a copy of the base | ✔ |
| Reference numerals excluded from **future** dataset releases; frozen release untouched | 97 of 694 targets carry them, so the model learned to. Editing a frozen release breaks its hashes and the handoff's byte-for-byte requirement | ✔ |
| `gemma-4-31b-claim-merged` deleted (58 GiB) | reproducible from base + adapter; came from the run that measured worse than its base. User approved | ✔ |
| **No input-shape diversification for now** | the product is drawings→claim. A fixed input shape is an advantage for a specialised tool, and the numeral problem is solved by cleaning targets, not by teaching instruction-following | ✔ |

### The principle behind the last one, because it will come up again

The training prompt is one fixed sentence per language (Korean for Korean
targets, English for English) and no system turn at all. A prompt that never
varies carries no information, so the model cannot learn to read it — it becomes
a trigger, not an instruction. That is why a serving system prompt is an
instruction the fine-tune has never seen, and why "just tell it not to write
numerals" is a weak lever against something learned from the targets.

Measured, on the tuned model, 12 records (`tools/prompt_probe.py`):

| condition | chrF | numerals | mean words | looping |
|---|---|---|---|---|
| training prompt (no system turn) | 10.24 | 2/12 | 71.2 | 1/12 |
| + serving system prompt | 10.40 | **0/12** | 98.9 | **3/12** |

So the prompt is **not** ignored — numerals go to zero and length improves — but
collapse gets worse. Both effects are real and small-sample.

---

## 4. User preferences and prohibitions

**Spending.** Name the cost and what it buys, ask once, then honour the
conditional grant. Write stop criteria *before* the run reaches the checkpoint
("at epoch N, stop if 2 of 3 samples degenerate or the mean falls below base").
The user is not slow to spend; they want each spend tied to a measurement.

**Explanations.** They interject mid-run and want the mechanism, not just the
number. A principle they understand is what they act on.

**Language.** Conversation in Korean; repository documents in English.

**Hard prohibitions.**

- Never modify, re-split, or re-hash the frozen dataset release.
- Never put `metadata/source_oracle_pages/`, `canonical/`, `evidence/`,
  original claim pages, or 발명의 설명 into a prompt, into training data, or
  into any retrieval context.
- Never write a Hugging Face token, RunPod key, or Drive credential into a file,
  a log, a commit, **or a tool call that echoes it back**. This blocked a step
  today — see §7.
- Do not open a pull request unless asked. Work on the task branch.

---

## 5. Next actions, concrete

**A. Finish the serverless LoRA test (in flight, blocked on the user).**

The user asked to test v2 on serverless and chose to reuse the existing INT4
endpoint rather than stand up a bf16 one. Setting the endpoint's `env` requires
putting `HF_TOKEN` through a tool call that echoes it back, which is prohibited,
and no curl route exists (§7). So the user was asked to add these in the RunPod
console, keeping the existing 8 variables:

```
ENABLE_LORA     true
LORA_MODULES    [{"name": "claim-v2", "path": "Mepeng22/gemma-4-31b-claim-lora-v2", "base_model_name": "gemma4-31b"}]
MAX_LORA_RANK   16
MAX_LORAS       1
HF_TOKEN        <their token>
```

`base_model_name` is set because the adapter's config names
`google/gemma-4-31B-it` while the endpoint serves the INT4 build. Once saved:
check worker logs for LoRA load, then send the same record twice —
`model: "claim-v2"` and `model: "gemma4-31b"` — and compare, sanitised and raw.

This is an **approximation**: a bf16-trained adapter on an INT4 QAT base. Useful
for a practitioner's feel, not quotable as a benchmark.

**B. Attack the repetition collapse — free first.**

15% of tuned generations degenerate. This is the one defect that clearly loses to
prompting, and it is not a data-cleanliness problem. Cheapest first:

1. A decode-time repetition penalty, measured offline against
   `run_artifacts/v2_20260906/validation_predictions.jsonl` — no GPU needed to
   see whether the failing 9 records would have been saved.
2. Fewer epochs (epoch 2 already won) or a lower LoRA rank.
3. More records.

**C. Fix the repetition metric before trusting it.**

`repetition_10gram` scores `제1 유로 … 제15 유로` as 0.00 because the numeral
makes every n-gram unique. Normalise digits to a placeholder before counting.
`evaluate.py`'s `excessive_repetition_gt_0_3` has this blind spot today, so the
run's own 9/59 count is a floor, not a total.

**D. Rotate the Hugging Face token.**

A `tail -3 pod.env` printed it into the session transcript on 2026-09-06. It has
`repo.write`. Nothing depends on it now — the pod is gone and the push is done.
If it is used for the endpoint in step A, rotate afterwards instead.

**E. When something is worth it, spend the test split — once.**

---

## 6. Open questions and hypotheses

| Question | Status |
|---|---|
| Does this worker-vllm build accept `LORA_MODULES`, and does vLLM support AWQ + LoRA here? | **Unknown.** RunPod documents the variable; issues [#162](https://github.com/runpod-workers/worker-vllm/issues/162) and [#119](https://github.com/runpod-workers/worker-vllm/issues/119) report it being dropped. Fallback: `VLLM_EXTRA_ARGS`, which the worker README documents as a verbatim escape hatch |
| Would a repetition penalty remove the 15% collapse without hurting content? | Untestable claim so far; measurable offline on the saved predictions |
| Is the tuned model's length advantage (105 vs 42 words) worth anything to a practitioner? | Subjective; that is what step A is for |
| Does the collapse come from over-adaptation (554 records, 410 modules) or from the targets? | Hypothesis: over-adaptation. Epoch 3 samples had *less* collapse than epoch 2 while eval_loss was worse, which does not fit a pure data explanation |
| Is chrF the right metric at all for this task? | Doubtful. Every system scores 7–15 and the ranking flips with β. Length must always be reported beside it |

---

## 7. Tried and failed — do not repeat

**RunPod SSH proxy rejects the pod key.** `ssh.runpod.io` authenticates against
keys registered on the *account*, not the pod's `PUBLIC_KEY`. Use the direct
address from `runtime.ports`; it also supports scp, which the proxy does not.

**No API route to set a serverless endpoint's `env` outside the MCP tool.**
Checked today: REST v1 `PATCH /endpoints/{id}` has no `env` field (scaling only);
`api.runpod.io/v2/endpoints/{id}` returns 404; `rest.runpod.io/v2/endpoints/...`
301-redirects to the docs site; the endpoint's template is not returned by
`list-templates` even with `includeEndpointBoundTemplates` (v2 endpoints carry
env directly). So env changes go through the MCP tool — which echoes secrets —
or through the console by hand.

**`pkill -f <pattern>` over SSH kills the SSH session** when the pattern appears
in your own command line. Cost three dropped connections. Use a bracketed
pattern that does not match itself: `pkill -f "tools/pod_gua[r]d\.sh"`.

**Reading a billing bucket while it is still settling understates it.** The same
06:00–07:00 hour read $1.20, then $1.51, then $3.62. Wait, or say "still
settling". This is the third time cost was got wrong in this project.

**chrF's default β=2 favours the longer output.** The base model's apparent
19.26 vs 13.12 win was entirely length: truncating its own output to the
reference length dropped it to 11.87 without changing a word.

**The four pod bring-up bugs** are in §9 — CRLF, PEP 668, `RUNPOD_POD_ID`, and
the guard's fractional deadline.

---

## 8. Where everything lives

| What | Where |
|---|---|
| Approved release (frozen, read-only) | `C:\Users\VIEW LIFW\Documents\Codex\patent-dataset-factory\outputs\releases\finetune_multilingual_approved_20260902` |
| Same, on Google Drive | `G:\내 드라이브\Patent Dataset Factory\finetune_multilingual_approved_20260902` — `canonical/all_pairs.jsonl` hashes to `ac0ce425…` |
| Converted package (what the pipeline reads) | `C:\Users\VIEW LIFW\Projects\Gemma-claim-data\v2_approved_20260902` |
| Same, private HF dataset repo the pod pulls from | `Mepeng22/gemma-claim-v2-approved-20260902` — 3,898 files, split hashes verified after a Hub round trip |
| v2 adapter | `Mepeng22/gemma-4-31b-claim-lora-v2` (private) |
| v1 adapter (fallback) | `Mepeng22/gemma-4-31b-claim-lora` (private) |
| This run's outputs | `run_artifacts/v2_20260906/` — `RESULTS.md`, metrics, predictions, samples, probe, filtered pod log |
| Previous run's audit | `run_artifacts/POST_HOC_GENERATION_AUDIT.md` |
| Dataset agent's rules | `patent-dataset-factory/AGENTS.md` §Target content rules, `governance/DECISION_LOG.md` 2026-09-06 entry |
| Local venv | `Projects\Gemma-claim\.venv` — outside OneDrive, torch imports fine |
| Pod SSH key | `~/.ssh/runpod_gemma_ed25519` |

Split ledger hashes, which prove the pod trained on the right bytes:

```
train.jsonl       69c780015eb37da06257cb33795963d77db5d8b60fd0602620ab18b792b561d3
validation.jsonl  69c28ae4b5205a9a13ed5cec2081231936e991a123100f8264c6b54e110d2db0
test.jsonl        db5533a8c06cd97fff8b953b15c2f5b9da36cff631947479d69182c0f366fe3c
```

### What the release contains

694 records, splits 554 / 65 / 75, ids disjoint. 612 Korean / 82 non-Korean
(81 US, 1 EP). 3,893 unique images, every one hash-matching the manifest. Targets
80–3,475 characters; 284 records sit at the 8-image cap. Zero leakage across
splits on family, publication, target, PDF and image.

**The trap:** each row carries 66 fields and
`canonical_source_claim_transcription` is byte-identical to the target.
`tools/convert_release.py` projects to `{id, images, messages}` by whitelist;
`pipeline/preflight.py` re-checks and dies `ORACLE_FIELD_IN_MODEL_INPUT`.
Do not weaken either check.

**5 records are `claim_boundary_status: UNRESOLVED`** (4 train, 1 validation).
ACCEPTED in the release, so they stay. Worth a look if metrics come out strange.

**The non-Korean split is too small to measure** — 82 records, only 4 in test.
Any non-Korean number is noise; say so rather than reporting it as a result.

---

## 9. Running it again

### Free gates, on the laptop, before any GPU

```bash
export DATA_ROOT="C:/Users/VIEW LIFW/Projects/Gemma-claim-data/v2_approved_20260902"
export OUT_DIR=<somewhere outside the dataset>
export RESUME_ADAPTER=Mepeng22/gemma-4-31b-claim-lora-v2   # or omit for a fresh LoRA
export PYTHONIOENCODING=utf-8            # cp949 kills rehearse.py on its own banner
PYTHONPATH=pipeline python pipeline/preflight.py
PYTHONPATH=pipeline python pipeline/token_audit.py
PYTHONPATH=pipeline python tools/rehearse.py     # 8 gates, all must pass, a SKIP is not a pass
python tests/test_pipeline.py && python tests/test_claim_prompt.py
```

All of these ran clean on 2026-09-06. `WORKSPACE` defaults to `/workspace`,
which on Windows is `C:\workspace` — set `OUT_DIR` rather than finding out later.

### Bringing the pod up by hand

Created with no start command and no secrets in `env`, then driven over SSH.
RunPod stores and returns pod `env` in plaintext, so a token passed at create
time is echoed back by every later API call.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/runpod_gemma_ed25519 -N ""
# create-pod: NVIDIA H200, COMMUNITY, runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404,
# containerDisk 30, volume 200 GB at /workspace, ports 22/tcp, sshPublicKey=<the .pub>
# then use the DIRECT address from runtime.ports, not the proxy

tar --exclude=./.venv --exclude=./.git --exclude=./.env -czf code.tgz .
ssh -i KEY -p PORT root@IP 'cat > /workspace/code.tgz' < code.tgz
ssh -i KEY -p PORT root@IP 'cd /workspace/code && tar --no-same-owner -xzf ../code.tgz'
python make_pod_env.py | ssh -i KEY -p PORT root@IP 'umask 077 && cat > /workspace/pod.env'
ssh -i KEY -p PORT root@IP "sed -i 's/\r$//' /workspace/pod.env"
ssh -i KEY -p PORT root@IP 'cd /workspace && set -a && . pod.env && set +a && \
    nohup setsid bash code/boot.sh > outputs/boot_nohup.log 2>&1 < /dev/null &'
```

Then **do not `cat` or `tail` `pod.env`.** That is how the token leaked.

### The four things that bit, in order

**CRLF.** `core.autocrlf` gives a Windows checkout CRLF and bash reads the `\r`:
`boot.sh` died on ``syntax error near unexpected token $'do\r'``. `.gitattributes`
now pins `*.sh text eol=lf`. It reached `pod.env` a second time because
`Path.write_text` translates `\n` to `\r\n` on Windows — `CODE_DIR` became
`/workspace/code\r` and `mkdir -p "$OUT_DIR"` silently created a directory named
`outputs\r`. Strip it on arrival.

**PEP 668.** The image's python is Debian-managed and refuses `pip install`.
torch is installed system-wide, so make a venv that can still see it and put the
PATH in `pod.env`:

```bash
python3 -m venv --system-site-packages /workspace/venv
# pod.env: PATH=/workspace/venv/bin:$PATH
```

**`RUNPOD_POD_ID` is only injected into the container's own start command.**
Starting `boot.sh` over SSH leaves it unset and `pod_guard.sh` degrades to
warning in a log nobody reads — the failure that cost $10.33. Put the pod id in
`pod.env` and check the guard's first log line.

**The guard's deadline is computed once.** `GUARD_DEADLINE_HOURS=4.5` with no
`bc` on the image resolved the deadline to "now" and the guard tried to
terminate the pod on its first poll. Fixed to use awk and to refuse loudly
rather than terminate silently — but still read `pod_guard.log` after every
start.

### Cost model, measured

417 optimiser steps over 554 records ≈ 50 minutes. Scoring 65 validation records
for base and tuned — 130 free-running generations — took longer than the
training. The base model runs to `MAX_NEW_TOKENS` because nothing stops it.
Whole run ≈ $12 on an H200 at $3.59/hr.

---

## 10. Sources and uncertainty

- **Metrics** come from `run_artifacts/v2_20260906/metrics.json` and the
  predictions beside it, recomputed locally for the β sweep. The β=1 comparison
  is my own analysis, not something `evaluate.py` reports.
- **The 9/59 repetition count is a floor**, not a total: the metric misses
  numeral-incremented loops (§5C).
- **Cost** is from `get-billing` hourly buckets, which settle late. ~$12 is
  read after settling for 04:00–07:00Z plus an estimate for the final 40
  minutes; re-read to confirm.
- **The prompt-probe result is 12 records.** Directionally clear on numerals
  (2→0), weak on collapse (1→3).
- **`serving/README.md` is now stale** where it explains why no adapter is
  served: that text describes v1's mode collapse, which v2 does not reproduce.
- **Not verified:** whether AWQ + LoRA works on this worker build; whether a
  repetition penalty helps; anything about non-Korean performance.

## The instruction the dataset agent asked to be passed on

> Read `FINETUNING_HANDOFF.md` first and follow it exactly. Do not search, merge,
> relabel, deduplicate or resplit the data; fine-tune and evaluate on the
> provided train/validation/test split only. Preserve the Korean
> `target_claim_clean` byte-for-byte — do not translate or normalise it. Write
> every training artefact outside the dataset directory.

`tools/convert_release.py` satisfies all of it.
