# KV-Base Project Instructions

## 1. Purpose and Scope

KV-Base is a research-oriented KV Cache framework derived from NVIDIA
KVPress. Its purpose is rapid prototyping of KV Cache research ideas while
reusing common model loading, generation, cache access, Prefill/Decode
lifecycle, tracing, recording, and evaluation infrastructure.

Typical downstream topics include:

- KV distribution and redundancy;
- attention, head, and token analysis;
- Prefill compression and Decode eviction;
- quantization, sparsity, merging, and reconstruction;
- temporal behavior and access-pattern traces;
- later system or accelerator studies.

This is an algorithm and phenomenon research base, not a production serving
framework, universal policy language, or hardware simulator. Downstream work
may modify attention, cache internals, kernels, or model-specific code when
necessary. Prefer reusing roughly 80% of the common infrastructure while
allowing the remaining 20% to be specialized.

## 2. Working Principles

- Keep abstractions small and research-useful.
- Make observation independent from intervention.
- Preserve existing KVPress Press behavior where practical.
- Treat Prefill and Decode as explicit execution phases.
- Do not materialize Q or full attention weights unless an experiment needs
  them.
- Keep logical KV reduction separate from physical bytes, traffic, latency,
  and throughput claims.
- Do not force every downstream idea into one rigid API.

The current architecture and extension points are documented in:

```text
docs/base/architecture.md
docs/base/extension_guide.md
docs/base/experiment_conventions.md
```

## 3. Before Modifying Code

For a fresh clone or an upstream version change:

1. Read `AGENTS.md`, `BASE_STATUS.md`, and `docs/PROJECT_GOALS.md`.
2. Read `docs/base/architecture.md`, `extension_guide.md`, and
   `experiment_conventions.md`.
3. Inspect the actual checked-out source, especially `pipeline.py`,
   `utils.py`, `context.py`, `observer.py`, `recorder.py`, and the relevant
   Press classes.
4. Verify the current baseline before making architectural changes.
5. Refresh documentation when the source and documentation disagree.

Do not assume that an older KVPress audit or a previous branch still matches
the current checkout. Preserve any additional upstream `AGENTS.md` files
unless they conflict with this fork's explicit goals.

Before changing generation or cache behavior, determine from the current code:

- how models and generation are invoked;
- how attention modules and Hugging Face Cache objects are reached;
- where Q/K/V and attention weights are available;
- how Prefill and Decode are distinguished;
- when Press hooks execute and how cache writes are applied;
- how Decode cache growth and QuantizedCache paths work;
- which model families and model-specific paths are supported;
- what existing tests and metadata cover the affected path.

Record implementation facts in the base documentation. Do not fill gaps with
assumptions.

## 4. Change Discipline

Prefer small, reviewable changes. Do not mix a broad architecture refactor
with a new paper algorithm.

For each meaningful change, document:

```text
What changed?
Why?
What was the previous behavior?
What is the new behavior?
Which code path is affected?
Does it affect Prefill, Decode, or both?
How was it verified?
```

Add a focused regression test for shared infrastructure. Keep method-specific
scoring rules, one-off visualizations, model workarounds, and specialized
kernels in downstream research code unless multiple projects need them.

When removing code, first scan imports, public exports, pipeline references,
tests, documentation, and dependencies. Do not leave broken public imports or
stale dependency declarations.

## 5. Documentation Rules

Documentation is part of the framework. Keep these files current:

```text
README.md
AGENTS.md
BASE_STATUS.md
docs/PROJECT_GOALS.md
docs/base/architecture.md
docs/base/extension_guide.md
docs/base/experiment_conventions.md
```

`AGENTS.md` contains operating rules. Detailed runtime behavior belongs in
`docs/base/architecture.md`; research workflow belongs in
`docs/base/extension_guide.md`; indexing and result metadata belong in
`docs/base/experiment_conventions.md`.

Do not maintain documentation as a file-by-file changelog. Explain design
intent and current behavior.

## 6. v0.1 Acceptance Checks

Before treating the base as usable, verify these cases:

### Observation only

- inference runs;
- an Observer records at least one layer/head/token statistic;
- Prefill/Decode context metadata is correct;
- output remains baseline-equivalent within expected numerical behavior;
- the Observer does not mutate the cache.

### Prefill intervention

- a simple deterministic or random retention policy modifies Prefill K/V;
- cache length/state is sensible;
- generation continues after the modification.

### Decode intervention

- a controlled operation runs at a known Decode interval;
- `decode_step` tracking is correct and monotonic;
- cache updates remain valid;
- generation continues after the modification.

The focused test suite is:

```bash
.venv/bin/python -m pytest \
  tests/test_context.py \
  tests/test_observer.py \
  tests/test_recorder.py \
  tests/test_infrastructure.py \
  tests/test_attention_patch.py -q
```

Also run, when the environment permits:

```bash
.venv/bin/python -m compileall -q kvpress tests
UV_CACHE_DIR=/tmp/kvbase-uv-cache uv pip check --python .venv/bin/python
```

For a real model, smoke-test ordinary generation, Prefill Press generation,
and Decode Press generation. Record model, revision, commit, device, dtype,
and cache configuration. Model downloads and GPU availability are environment
conditions, not reasons to claim unverified results.

## 7. v0.1 Boundaries

Do not make the base depend on:

- custom CUDA kernels;
- vLLM or serving internals;
- a database or heavy experiment manager;
- a physical KV memory manager;
- a universal policy DSL or large plugin registry;
- a full accelerator simulator.

Downstream projects may later integrate vLLM, FlashInfer, LMCache, custom
CUDA, simulators, or accelerator models for physical-system validation.
