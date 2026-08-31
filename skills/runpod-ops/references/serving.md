# Serving, and diagnosing a serverless endpoint

## Crash-loop triage, in order

Workers UNHEALTHY with **no container output at all** is a startup failure. The
container is exiting before the server writes its first line, so the model, the
GPU and the quantisation are all still unexamined and none of them is likely to
be the cause. Check in this order:

**1. Does a path in the environment still exist?** The single most expensive
misdiagnosis in the project this skill comes from: worker-vllm downloads the
model into `BASE_PATH`, default `/runpod-volume`. The network volume backing
that path had been deleted as a cleanup step. Every container then exited in
about 16 seconds. It looked like an adapter problem, then a GPU-model problem,
then a CUDA problem, and hours went into all three before anyone checked whether
the download directory existed. **If containers die in seconds, suspect a path or
a credential before anything technical.** Point the cache at container-local
storage (`HF_HOME=/tmp/hf`) when no volume is attached.

**2. How fast do they die?** Seconds means argument parsing or filesystem —
loading a multi-billion-parameter model takes minutes, so anything faster never
reached the model. Minutes means OOM or a genuine load failure.

**3. Are the workers even on the current config version?** A worker on a stale
version serves the old setup and will happily answer with the old model. One test
came back COMPLETED and proved nothing because a stale worker served it. Check
the version and staleness fields before trusting any result.

**4. Is the config internally contradictory?** Mutually exclusive fields can be
stored together even though the API rejects setting both, leaving an endpoint
that cannot schedule. When an endpoint has been edited many times, a fresh one
with a clean minimal config is often faster than archaeology — creating an
endpoint is free, and it isolates config rot from everything else in one step.

## worker-vllm request shapes

Three input shapes are accepted and **only one honours top-level parameters**:

```json
{"input": {"openai_route": "/v1/chat/completions",
           "openai_input": {"model": "...", "messages": [...],
                            "max_tokens": 500, "temperature": 0}}}
```

The shorthand `{"messages": ..., "sampling_params": ...}` silently ignores
`model` and `max_tokens`. A round of endpoint tests using it proved nothing: a
request naming one model came back served by another, and a 96-token cap
produced 687 tokens. The conclusion drawn from those tests was confident and
wrong. If a response contradicts the request, suspect the shape before the
engine.

`{"openai_route": "/v1/models"}` is a near-free probe that answers "what is this
endpoint actually serving" without generating anything.

The same handler is reachable over HTTP at
`https://api.runpod.ai/v2/{endpoint}/openai/v1`, which any OpenAI client can use
as a `base_url` — usually the cleanest integration path.

## Adapters on a quantised base

Before serving a LoRA adapter on a quantised checkpoint, check what the
quantised repo was quantised *from*. A QAT checkpoint is a different set of
weights than the original instruct model, so an adapter trained on the latter is
being applied to a base it never saw, and whatever was measured at training time
no longer holds. This is a correctness problem, not a compatibility one: it can
load cleanly and still be a model nobody has evaluated.

Also check whether the quantisation `ignore` list leaves the layers the adapter
targets quantised, and whether the adapter's module names match the server's
naming — inference servers commonly fuse q/k/v and gate/up projections, so names
from a training framework may not resolve.

Merging the adapter and quantising the merged model sidesteps all of it, at the
cost of one quantisation job.

## Endpoint settings worth setting deliberately

- **min workers 0** — the whole reason serverless is cheap.
- **idle timeout** — the tradeoff between paying for idle and waiting out cold
  starts. Short for one-off calls; a few minutes for interactive sessions.
- **max workers 1** while testing — nothing multiplies a mistake faster.
- **credentials** — an endpoint's environment is stored and returned in
  plaintext by the API. Keep tokens out of it when the model is public.
