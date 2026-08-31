# Description optimisation, run 1 — void

`scripts/run_loop.py`, 5 iterations, 20 queries, 3 runs each. Every iteration
scored train 6/12 and test 4/8, identically. Those are exactly the counts of
**negative** queries in each split.

Every negative passed. Every positive failed, at a 0.00 trigger rate — including
one that names RunPod outright.

## Cause

The loop's default `--timeout` is 30 s. A `claude -p` invocation in this
environment takes 44–59 s wall clock. Runs were killed before the trigger could
be observed, and an unobserved trigger is recorded as "did not trigger" — which
scores as a pass for a negative and a failure for a positive. Two positives
scored 0.33, the signature of a timeout near the boundary letting the occasional
run through.

## Verified separately

With no timeout, a command file carrying this description **is** listed among
the available commands and **is** invoked. Measured on two of the harder
positives:

| query | wall clock | triggered |
|---|---|---|
| "RunPod에서 gemma 파인튜닝을 다시 돌리려고 하는데 뭐부터 준비하면 될까?" | 59.5 s | yes, stream line 3 |
| "my serverless endpoint has 5 workers and all show UNHEALTHY with no container logs" | 44.1 s | yes, stream line 4 |

So the mechanism works and the description triggers. What is unmeasured is the
rate across the full set, and whether the negatives stay quiet under a timeout
long enough for them to speak.

## To re-run

Pass `--timeout 120`. Budget the quota first: one iteration is
(train + test) × runs-per-query invocations — 60 at the defaults — and each
takes about a minute. Five iterations is ~300 invocations. Consider
`--runs-per-query 2 --max-iterations 2`.
