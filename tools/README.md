# tools

Three gates that stand between a new dataset and a repeat of the last run.
Run them in this order.

## 1. `rehearse.py` — before any GPU

```bash
python tools/rehearse.py
```

Runs the real pipeline against the real processor and a weightless model
skeleton. No GPU, no weights, no billing; a config and a tokenizer are the only
downloads. Catches the three bugs that each cost a restart cycle on a
$1.50-2.00/hr machine last time:

- the chat-template prefix violation (Gemma 4's generation prompt opens an empty
  thought channel the full rendering does not contain),
- LoRA targets re-matching the vision tower through bare module suffixes,
- `TrainingArguments` arguments disappearing across a transformers release.

It also checks that loss is masked to one contiguous assistant span, and that
every dataset record has a non-empty target and images that exist.

Exit code 0 means every gate that *could* run passed. **A skip is not a pass** —
the summary lists skipped gates separately and says where to re-run them. A gate
that cannot reach the Hub falls back to a fake processor or a toy module tree,
which verifies the selection logic but not the real model, and says so.

Verified: five injected faults — a vision-tower target, bare suffixes, a missing
essential training argument, a prompt that is not a prefix, and an empty
assistant target — are all caught.

`REHEARSE_TRACE=1` prints tracebacks.

## 2. `baseline.py` — before deciding to train

```bash
export RUNPOD_API_KEY=...
python tools/baseline.py --split validation
```

Scores the prompted base model through the serving endpoint with the same
metrics `pipeline/evaluate.py` applies to a fine-tune, and writes predictions
plus a metrics file. That number is the bar: a fine-tune is worth its GPU time
only if it beats it on free-running generation.

The last run never established this bar, so there was no moment where stopping
was the obvious call — and the fine-tune it produced was worse than this
baseline would have been. Cost is one cold start plus seconds per record.

Defaults to `validation`. The test split is spent once, after the config and
checkpoint are frozen; passing `--split test` prints a warning.

Watch **chrF**, not loss. Loss fell all the way through the last run while chrF
fell too, which is the signature of a model learning the target distribution
instead of the input-to-output mapping. The tool warns if chrF cannot be
computed.

## 3. `pod_guard.sh` — while the GPU runs

Started automatically by `boot.sh`. To run it by hand:

```bash
nohup bash tools/pod_guard.sh >> /workspace/outputs/pod_guard.log 2>&1 &
```

Stops the pod on either of two triggers:

- **idle** — GPU at or below `GUARD_IDLE_PERCENT` (default 5%) for
  `GUARD_IDLE_MINUTES` (default 20). This is the one that catches an abandoned
  pod, which is what cost $10.33.
- **deadline** — `GUARD_DEADLINE_HOURS` (default 8) of wall clock, for a run
  that is busy but going nowhere.

| variable | default | |
|---|---|---|
| `GUARD_DEADLINE_HOURS` | 8 | hard cap |
| `GUARD_IDLE_MINUTES` | 20 | idle before stopping |
| `GUARD_IDLE_PERCENT` | 5 | GPU% at or below counts as idle |
| `GUARD_POLL_SECONDS` | 60 | polling interval |
| `GUARD_DRY_RUN` | 0 | 1 = log the decision, do not stop |
| `GUARD_DISABLE` | 0 | 1 = `boot.sh` does not start it |

Stopping a pod from inside needs `RUNPOD_API_KEY` on the pod, and a RunPod key
is account-wide — it can create and delete pods, not only call an endpoint.
Weigh that against a pod that bills all night. Without the key the guard still
runs and still says, in its log, that the pod should be stopped; it just cannot
do it. It prefers *stop* over *terminate* so artefacts survive.

Verified: the deadline fires, the idle timer fires at its budget, and a busy GPU
does not trigger it.
