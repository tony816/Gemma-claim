# What actually costs money

Numbers below are measured on RunPod in 2026, not quoted from a price page.
Rates change; the *shapes* are what to reason about.

## Three pricing shapes

**Pods — per hour of existence.** A pod bills whether the GPU is at 100% or 0%.
This is where money leaks, and the leak is not usually carelessness: it is that
the session which would have stopped the pod ended first. A pod therefore needs
to be able to stop itself. One abandoned pod cost $10.33 of a $21 project.

**Serverless — per second of container uptime.** At min-workers 0 an idle
endpoint costs nothing at all, disk included. Measured directly: after workers
scaled down, the hourly billing records for that endpoint stopped appearing —
not small, absent. Container disk bills only while a container runs, and in
proportion to runtime, not to the configured size (two endpoints configured at
150 GB and 100 GB billed the same per container-hour).

The cost is paid in latency instead: every cold start reloads the weights.
Measured for a 19.5 GB INT4 31B model on an A40: ~290 s total — ~20–30 s to
download, ~35–50 s to load, ~70 s of `torch.compile`, ~65 s of multimodal
warmup. Then generation in 5–7 s.

**Network volumes — per month, regardless of use.** Storage keeps billing while
you sleep. Deleting one stops that, but see `serving.md`: things silently depend
on its mount path.

## Choosing

- Occasional inference: serverless, min-workers 0. Cold starts are the price,
  and they are cheaper than an idle GPU.
- Interactive debugging: a pod with a guard, or serverless with a longer idle
  timeout so a run of requests reuses one warm worker. A 300 s idle window costs
  about $0.10 trailing and removes a 5-minute wait between questions.
- Training: a pod, guarded, with a deadline you actually expect to hit.

## Check the bill, do not derive it

Query the provider's billing API bucketed by hour and read what it says. On
RunPod that is `get-billing` with `bucketSize: hour`, optionally scoped to
pods/serverless/volumes. Reconstructing a balance by arithmetic produced a
wrong answer twice in the same project; the API is authoritative and free.

Two things worth checking whenever something looks off:

- **Are there pods at all?** List them. A forgotten pod is invisible until you
  look for it.
- **Are there volumes?** They bill monthly and belong to nothing visible.

## The guard

`scripts/pod_guard.sh` stops a pod on either trigger:

- **idle** — GPU at or below a floor (default 5%) for a window (default 20 min).
  This is the one that catches abandonment.
- **deadline** — wall-clock cap (default 8 h), for a run that is busy but wrong.

Stopping a pod from inside needs an API key on the pod, and on RunPod that key
is account-wide — it can create and delete pods, not only stop this one. That is
a real tradeoff against a pod billing all night; decide it deliberately. Without
the key the guard still runs and still writes, in its log, that the pod should be
stopped. Prefer *stop* over *terminate* so artefacts survive the shutdown.
