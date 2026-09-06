# Handoff — continuing the claim generator on the 694-record release

Written 2026-09-02 at the end of a local session, for whoever picks this up in a
cloud session. Read `CLAUDE.md` first; it is the record of what the previous run
cost and why. This file is only the delta since then.

## What the user decided

1. **A new dataset is ready** and replaces `v1.1.2-independent-oracle-clean`:
   694 records, 612 of them Korean. It has been verified (below) and converted.
2. **Continue training the existing adapter**, not a fresh LoRA. The user was
   told the previous adapter is mode-collapsed (0/12 apparatus match, chrF
   28.85 -> 21.04, mean length 369.6 -> 158.1 words) and that starting from the
   base is the safer call. They reaffirmed continuing from the adapter. Do not
   re-litigate it; `RESUME_ADAPTER` exists for exactly this.
3. **No pod today.** The pod runs tomorrow, after the free gates pass.

## Where things stand

| Step | Status |
|---|---|
| 1. Whitelist conversion adapter | done — `tools/convert_release.py`, ran clean |
| 2. preflight counts + oracle-field gate | done — passes, and fails correctly on a poisoned package |
| Resume-from-adapter path in `train.py` | done — `RESUME_ADAPTER` |
| Language-aware baseline scoring | done — was scoring 612 Korean records with the English prompt |
| 3. `token_audit.py` | done 2026-09-06 — ran locally, PASS |
| 4. `rehearse.py` clean | done 2026-09-06 — 7/7 after fixing the gate itself |
| 5. `baseline.py` | **not started** — the only gate left, and the first that costs money |

## The blocker that moved this to the cloud — gone (2026-09-06)

It was OneDrive. The venv now lives at `Projects\Gemma-claim\.venv`, outside
OneDrive, and `import torch` succeeds (2.14.0+cpu); the `WinError 4551`
application-control failure on `shm.dll` does not reproduce. `torchvision` was
missing and was installed from the CPU index — `transformers` 5.16.1 imports it
inside `Gemma4Processor`, so the processor cannot load without it.

Everything that does not need a GPU therefore runs on the laptop. Two things
about running it there:

- The console is cp949. Export `PYTHONIOENCODING=utf-8` or `rehearse.py` dies
  on the em dash in its own banner before running a single gate.
- `WORKSPACE` defaults to `/workspace`, which on Windows resolves to
  `C:\workspace`. That is where `outputs/token_length_audit.json` and
  `outputs/dataset_preflight.json` landed. Outside the dataset tree, so the
  policy holds, but set `OUT_DIR` explicitly rather than relying on it.

## The data — this is the part that does not travel with the repo

The dataset is **not in this repository and not in the cloud**. Two artefacts
exist on the user's laptop and one of them has to be transferred:

- **Approved release** (read-only, frozen, 5,482 files):
  `C:\Users\VIEW LIFW\Documents\Codex\patent-dataset-factory\outputs\releases\finetune_multilingual_approved_20260902`
- **Converted package** (what the pipeline actually reads, 1.1 GB):
  `C:\Users\VIEW LIFW\Projects\Gemma-claim-data\v2_approved_20260902`
  — moved out of OneDrive; all three ledger hashes below re-verified there.
- **Approved release, Google Drive copy** (uploaded 2026-09-02):
  `G:\내 드라이브\Patent Dataset Factory\finetune_multilingual_approved_20260902`
  — `canonical/all_pairs.jsonl` on Drive hashes to the same `ac0ce425…`.

Transferring the converted package is enough. Ask the user how they want to move
it; do not upload patent material anywhere without their explicit go-ahead.

If you re-run the conversion instead of transferring it, the output must match:

```
train.jsonl       69c780015eb37da06257cb33795963d77db5d8b60fd0602620ab18b792b561d3
validation.jsonl  69c28ae4b5205a9a13ed5cec2081231936e991a123100f8264c6b54e110d2db0
test.jsonl        db5533a8c06cd97fff8b953b15c2f5b9da36cff631947479d69182c0f366fe3c
```

```bash
python tools/convert_release.py --release <release dir> --out <package dir>
```

### What the release actually contains

Independently verified this session, all of it matching the dataset-building
agent's report:

- 694 records; splits 554 / 65 / 75, ids disjoint, `split` field agrees with the file
- canonical SHA-256 `ac0ce42589d3c06d689ad8ff00075f2ed9018d12bf21246c6e3133cb796401c2`
- zero leakage across splits on family, publication, target, PDF **and image**
- 3,893 unique images, every one present and hash-matching `manifests/file_manifest.jsonl`
- Korean targets are clean UTF-8 NFC; 612 Korean / 82 non-Korean (81 US, 1 EP)
- targets 80–3,475 characters; 284 records (41%) sit at the 8-image cap

### The trap in the release

Each row carries 66 fields, and `canonical_source_claim_transcription` is
**byte-identical to the target**. `semantic_evidence`, `evidence_pages`,
`source_oracle_*` and `source_boundary_next_claim_clean` are oracle material
too. Passing a row through unfiltered puts the answer in the prompt.

`tools/convert_release.py` projects each row down to `{id, images, messages}` by
whitelist. `pipeline/preflight.py` re-checks that projection at the point of use
and dies `ORACLE_FIELD_IN_MODEL_INPUT` on anything else — verified by feeding it
a deliberately poisoned package. Do not weaken either check.

## The adapter being continued

`Mepeng22/gemma-4-31b-claim-lora` (private, 489 MB), base `google/gemma-4-31B-it`,
r=16, alpha=32, dropout 0.05, fully-qualified target module names, peft 0.20.0.

**Its `adapter_config.json` has `inference_mode: true`.** Loading it without
`is_trainable=True` leaves every LoRA parameter frozen, and the run completes
with a flat loss curve and no error anywhere. `pipeline/train.py` passes the flag
and then dies `NO_TRAINABLE_PARAMETERS` if the trainable count is still zero.
`tools/rehearse.py` has a gate for this that runs when `RESUME_ADAPTER` is set.

Enable the resume path with:

```bash
export RESUME_ADAPTER=Mepeng22/gemma-4-31b-claim-lora
```

When set, the adapter's own rank and target modules win; `LORA_R`, `LORA_ALPHA`
and `LORA_DROPOUT` are ignored. `train.py` hard-fails if the adapter's base model
differs from `BASE_MODEL` or if any of its target modules is not an adaptable
language-tower Linear here.

## Run these, in this order, before any GPU

```bash
export DATA_ROOT=<converted package dir>
export OUT_DIR=<somewhere outside the dataset>
export RESUME_ADAPTER=Mepeng22/gemma-4-31b-claim-lora
```

```bash
python -m pip install torch && python -m pip install -r requirements.txt
```

```bash
PYTHONPATH=pipeline python pipeline/preflight.py
```

```bash
PYTHONPATH=pipeline python pipeline/token_audit.py
```

```bash
python tools/rehearse.py
```

```bash
python tools/baseline.py --split validation
```

`preflight.py`, `token_audit.py` and `rehearse.py` all pass on the converted
package as of 2026-09-06, with no skips. `baseline.py` is the decision point: it
scores the prompted base model through the serving endpoint, and the fine-tune is
worth its GPU time only if it beats those numbers on free-running generation.

### What token_audit measured (2026-09-06, real processor, all 694 records)

`ZERO_TARGET_TRUNCATION_PASS=true` — nothing is truncated and nothing comes
close to the 262,144-token context.

| Split | n | mean total | max total | max target |
|---|---|---|---|---|
| train | 554 | 1,811.8 | 4,075 | 1,870 |
| validation | 65 | 1,891.3 | 2,706 | 513 |
| test | 75 | 1,787.3 | 3,099 | 894 |

Whole corpus: p50 1,954, p95 2,692, max 4,075 total tokens. Max prompt is 2,205
in every split — that is the 8-image cap, and 284 records sit on it. The earlier
~1,900-mean / ~5,000-worst estimate was close; the worst case is lower.

**`recommended_max_new_tokens` is 2,401 and should not be used.** It is derived
from the longest *train* target (1,870 tokens), which is never generated.
Generation happens only over validation and test, and there the longest
reference is 894 tokens:

- validation targets: p50 204, p95 487, max 513
- test targets: p50 217, p95 641, max 894
- records over 1,024 target tokens: **0 in validation, 0 in test** (4 in train)

So `MAX_NEW_TOKENS=1024` covers every reference the evaluator will ever be asked
to reproduce, with 130 tokens of headroom on the longest one. The 11-hour
scenario this file warned about is avoidable at no cost in fidelity.

### What baseline is expected to reveal

`baseline.py` now picks the system prompt per record from the reference language
and reports metrics split by language, because 88% of the data is Korean and the
tool was previously sending the English prompt to every record. It also reports
`language_drift` — records answered in the wrong language.

Note the endpoint `fdiltabt78bogm` serves `cyankiwi/gemma-4-31B-it-qat-AWQ-INT4`,
an **INT4** build, while training is bf16. The baseline is therefore slightly
pessimistic. That is the safe direction, but say so when reporting the number.

### What the first real run of rehearse.py found

`rehearse.py` had never executed before 2026-09-06. Its first run failed one
gate — **and the bug was in the gate, not in the pipeline**:

```
[ FAIL ] loss is masked to the assistant span
         RuntimeError: supervision starts at token 0 — the prompt is being trained on
```

`encode_record` returns `labels` batched as `(1, seq)`. The gate called
`.tolist()` and enumerated it, so it walked the batch axis, saw a single element
that was a list rather than `-100`, and concluded token 0 was supervised. The
mask itself is correct: on `train[0]` it supervises 104 of 954 tokens,
contiguous from 850, which is exactly the prompt length `token_audit` reports.
The gate now takes the row before enumerating. After that, 7 passed / 0 failed /
0 skipped.

Worth stating plainly, because this is the failure mode CLAUDE.md exists to
prevent: a gate that reports a false alarm is only one sign flip away from a
gate that reports a false pass, and neither had ever been executed.

The offline suite (`tests/test_pipeline.py`) was also failing, for a different
stale-fixture reason: `tests/make_synthetic.py` still built a 91/11/12 package
carrying a `metadata` key, so it tripped both the frozen-count gate and the
oracle-field whitelist that commit 196cf23 added. The fixture now mirrors the
real release — 554/65/75, `{id, images, messages}` only. 9/9 sections pass.

## Cost and time, from the previous run's measured figures

Ground truth: 91 records, 115 optimiser steps, 610.3 s wall clock on an H200,
peak VRAM 70.03 GiB. Billing history shows the previous run cost $19.91 in pods,
$10.33 of which was a single pod nobody terminated.

Current H200 SXM pricing: community $3.59/hr, secure $4.59/hr, serverless
$5.93/hr. Availability was **LOW** in both pools when checked.

Scaled to 554 records at 6 epochs = 831 optimiser steps:

| Stage | Time | Cost @ $3.59/hr |
|---|---|---|
| Boot + 62.5 GB weight download | 0.3–0.5 h | $1–2 |
| LoRA training | 1.6–2.0 h | $6–7 |
| Evaluation, 280 generations, `MAX_NEW_TOKENS=1024` | 2.5–3.5 h | $9–13 |
| Adapter push | 0.2 h | $1 |
| **Total** | **4.6–6.2 h** | **$17–23** |

**Evaluation costs more than training.** `evaluate.py` generates over every
record of every eval split for both base and tuned: (65 + 75) x 2 = 280
generations, against 23 last time. Set `MAX_NEW_TOKENS` explicitly — left to the
token audit it may resolve to ~3,000, and 280 generations that run to that limit
without hitting EOS is 11+ hours.

Cheapest sane configuration — `EVAL_SPLITS=validation` only, `MAX_NEW_TOKENS=1024`,
adapter push without the merged model — is about 3.0–3.5 h and $11–14. Spend the
test split once, after the config and checkpoint are frozen.

`tools/baseline.py` is separate and serverless: roughly $2–5 for 65 records. It
is the only thing standing between the user and another $20 spent on a run that
loses to prompting.

## Bringing the pod up by hand (2026-09-06)

The pod was created with no start command and no secrets in its `env`, then
driven over SSH. That is deliberate: RunPod stores and returns pod `env` in
plaintext, so a token passed at create time is echoed back by every later API
call. Nothing here puts a credential on a command line either.

```bash
# 1. a dedicated key, then create the pod with it (no env, no command)
ssh-keygen -t ed25519 -f ~/.ssh/runpod_gemma_ed25519 -N ""
#    create-pod: NVIDIA H200, COMMUNITY, runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404,
#    containerDisk 30, volume 200 GB at /workspace, ports 22/tcp, sshPublicKey=<the .pub>

# 2. the account-level SSH proxy (ssh.runpod.io) rejects this key -- it
#    authenticates against keys registered on the RunPod account, not the pod's
#    PUBLIC_KEY. Use the direct address from the pod's runtime.ports instead;
#    it also supports scp, which the proxy does not.

# 3. ship the tree and the env over stdin, so neither ever appears in a command
tar --exclude=./.venv --exclude=./.git --exclude=./.env -czf code.tgz .
ssh -i KEY -p PORT root@IP 'cat > /workspace/code.tgz' < code.tgz
ssh -i KEY -p PORT root@IP 'cd /workspace/code && tar --no-same-owner -xzf ../code.tgz'
python make_pod_env.py | ssh -i KEY -p PORT root@IP 'umask 077 && cat > /workspace/pod.env'

# 4. start it detached, with the env sourced
ssh -i KEY -p PORT root@IP 'cd /workspace && set -a && . pod.env && set +a && \
    nohup setsid bash code/boot.sh > outputs/boot_nohup.log 2>&1 < /dev/null &'
```

### Four things that bit, in order

**CRLF.** `core.autocrlf` gives a Windows checkout CRLF, the tarball carries it,
and bash reads the `\r`: `boot.sh` died on ``syntax error near unexpected token
$'do\r'``. `.gitattributes` now pins `*.sh text eol=lf`. The same bug reached
`pod.env` a second time, because `Path.write_text`/`sys.stdout` translate `\n`
to `\r\n` on Windows -- `CODE_DIR` came out as `/workspace/code\r`, python
could not open the file, and `mkdir -p "$OUT_DIR"` had silently created a
directory literally named `outputs\r`. Strip it on arrival:
`sed -i 's/\r$//' /workspace/pod.env`.

**PEP 668.** The image's python is Debian-managed and refuses `pip install`
(`externally-managed-environment`). torch is installed system-wide, so the fix
is a venv that can still see it, and a PATH line in `pod.env`:

```bash
python3 -m venv --system-site-packages /workspace/venv
# pod.env: PATH=/workspace/venv/bin:$PATH
```

**`RUNPOD_POD_ID` is only injected into the container's own start command.**
Starting `boot.sh` over SSH leaves it unset, and `pod_guard.sh` then degrades to
warning in a log that nobody is reading -- exactly the failure that cost $10.33.
Put the pod id in `pod.env` explicitly and check the guard's first log line says
nothing about being unable to self-stop.

**The guard's deadline is computed once, from its environment.** The first boot
ran with `GUARD_DEADLINE_HOURS=5\r`; the arithmetic failed, the deadline
resolved to "now", and the guard tried to terminate the pod 7 minutes in. It
could not, only because the pod id was missing. Two bugs cancelling out is not a
safety margin -- read `pod_guard.log` after every start.

## Credentials

Two are needed; neither is in this repository and neither should ever be written
into a file, a log, or a commit.

- **Hugging Face**, with access to gated `google/gemma-4-31B-it` and to the
  private adapter repo. `hf auth login` stores it where every process finds it.
- **`RUNPOD_API_KEY`**, for `baseline.py` and for pod management. Account-wide:
  it can create and delete pods, not just call an endpoint.

The user set both on the local machine; they will need setting again in the cloud
environment.

## Reference numerals in the targets (decided 2026-09-06)

The fine-tuned model writes drawing reference numerals into its claims
(`펌버주입구(131a, 141a)`), which must never appear in a claim. It is not
inventing the habit — it is copying the data:

| split | Korean targets | carrying reference numerals |
|---|---|---|
| train | 482 | 40 (8.3%) |
| validation | 59 | 4 (6.8%) |
| test | 71 | 8 (11.3%) |

e.g. `시료 전처리 장치(100)에 있어서, 시료가 유입되는 유입관(110);`. Counting
these is fiddly: `(1) 시료첨가용 멤브레인 패드` is an enumeration marker and
belongs in a claim, and `챔버(chamber)` is a gloss. What separates a numeral is
that it is glued to the noun and contains only digits.

**The user's decision:** future dataset releases exclude reference numerals from
the targets; this release is frozen and stays as it is; until then the output is
constrained at serving time. Do not strip numerals from the frozen targets --
that breaks the release hashes and the handoff's byte-for-byte requirement.

Two things were done about it here:

1. `serving/claim_prompt.py` now strips parenthesised numerals in both
   languages. It never did: the rule was written for the bare form
   (`하우징 10은`) and its own comment said the net had never fired, because the
   *base* model does not use numerals in Korean. The fine-tuned model does.
   Korean needs the particle re-picked after the cut (`플랫폼(2)를` ->
   `플랫폼을`), and the English rule must not eat `(2)` in
   `comprising: (1) a housing; and (2) a sensor` -- which is why it keys on what
   follows the parenthesis rather than what precedes it: the commonest component
   noun in this corpus ends in the letters that a "not after `and`" guard would
   have to exclude (`a sensor (20)`). Seven cases are in
   `tests/test_claim_prompt.py`; `space/claim_prompt.py` is in sync.

2. `tools/prompt_probe.py` measures whether the serving system prompt actually
   suppresses numerals **on the tuned model**. This is not obvious: the training
   prompt carries no system turn at all — it is one English line plus the
   drawings — so the Korean system prompt, numeral ban included, is an
   instruction the fine-tune has never seen. It may be ignored, or it may cost
   the gains. Run it while a pod with the adapter is still alive.

`evaluate.py` scores raw generations, not sanitised ones, so numerals are in the
run's chrF as written.

## Still open

- **`pipeline/fetch_dataset.py:17` still pins the old ZIP's SHA-256.** No ZIP has
  been produced for the new package. Either produce one
  (`tools/convert_release.py --zip` prints the hash) and update the constant, or
  bypass `fetch_dataset.py` and point `DATA_ROOT` at the transferred package.
  `run_all.sh` calls `fetch_dataset.py`, so decide before using it.
- **`evaluate.py` has no per-language breakdown.** The dataset handoff requires
  Korean and non-Korean reported separately. `baseline.py` does this now;
  `evaluate.py` does not, and the two should agree before the results are
  compared.
- **5 records have `claim_boundary_status: UNRESOLVED`** (4 train, 1 validation).
  They are ACCEPTED in the release and the release is frozen, so they stay. Worth
  a look if the metrics come out strange.
- **The non-Korean split is too small to measure.** 82 records total, only 4 in
  test. Any non-Korean number will be noise; say so rather than reporting it as
  a result.

## The instruction the dataset agent asked to be passed on

> Read `FINETUNING_HANDOFF.md` first and follow it exactly. Do not search, merge,
> relabel, deduplicate or resplit the data; fine-tune and evaluate on the
> provided train/validation/test split only. Preserve the Korean
> `target_claim_clean` byte-for-byte — do not translate or normalise it. Write
> every training artefact outside the dataset directory.

`tools/convert_release.py` already satisfies all of it: it reads the release
read-only, refuses to write inside it, preserves image order, and copies the
Korean targets verbatim.
