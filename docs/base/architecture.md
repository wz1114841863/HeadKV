# KV-Base Architecture

This document describes the currently implemented research infrastructure.

## Design Boundary

KV-Base is the reusable algorithm and phenomenon-exploration layer. It should
make observation and intervention easy without requiring every research idea
to use the same internal implementation. A downstream project may change
attention, cache layout, or model-specific code when necessary.

The two supported conceptual modes are:

```text
Observe:
  model execution -> KVObserver -> KVRecorder / analysis

Intervene:
  model execution -> Press / policy -> modified KV state -> evaluation
```

The modes may be combined. The base does not claim production serving,
physical sparse execution, or hardware savings from logical KV reduction alone.

## Runtime Path

```text
model.generate() / pipeline
        |
        v
Transformers attention and Cache update
        |
        v
KVPress attention-layer hook
        |
        +--> KVContext + KVObserver    (read-only)
        |
        +--> BasePress / DecodingPress  (intervention)
        |
        v
KVRecorder / evaluation
```

Model and cache creation remain owned by Transformers. KV-Base observes or
intervenes through the existing KVPress hook path.

## Main Modules

| Module | Responsibility |
| --- | --- |
| `kvpress/pipeline.py` | Custom text-generation pipeline. |
| `kvpress/presses/base_press.py` | Press context manager, attention hook, and cache intervention. |
| `kvpress/presses/decoding_press.py` | Decode-time Press wrapper and periodic intervention. |
| `kvpress/context.py` | `KVPhase`, `KVContext`, and shared lifecycle metadata. |
| `kvpress/observer.py` | Read-only attention-layer observation. |
| `kvpress/recorder.py` | Explicit file-backed experiment recording. |
| `kvpress/attention_patch.py` | Attention-function patch used by supported head-wise Presses. |
| `kvpress/presses/` | Existing KVPress methods and wrappers. |

## KVContext

`KVContext` is created for each attention-layer hook event. It contains the
available execution metadata and tensor references:

```text
phase              PREFILL or DECODE
layer_id
forward_id
layer_event_index
decode_step
prefill_length
total_length
stored_length
cache_position
hidden_states
keys / values
query              optional; currently None in the generic hook
attention          optional
cache
recorder
```

`k` and `v` are aliases for `keys` and `values`. Tensor data is not copied by
default. Consumers that retain tensors must clone or move them explicitly.

## Lifecycle

`KVExecutionLifecycle` is shared by Press and Observer paths.

- `layer_id == 0` starts a new model forward.
- All layer events in that forward share `forward_id`.
- Decode forwards advance a monotonic `decode_step`.
- Phase detection preserves the existing KVPress `is_prefilling` predicate.
- Lifecycle state resets when a Press or Observer context ends.

This unifies metadata only. Cache creation, cache growth, attention execution,
and cache write-back remain in the Transformers/KVPress path.

## Observer

`KVObserver` installs a read-only post-attention hook. For every event it:

1. extracts the available K/V tensors;
2. creates a `KVContext`;
3. calls `on_layer(context)`;
4. calls `on_prefill(context)` or `on_decode(context)`;
5. returns the original attention output.

The observer does not write to the cache and does not materialize Q or full
attention weights automatically.

## Intervention

Existing Presses remain the intervention interface. `BasePress` preserves the
existing `compress(...)` contract, while its hook supplies `kv_context` to
custom code. Decode-specific wrappers perform controlled cache updates during
generation.

Do not rewrite all Presses to fit a new abstraction. A downstream method may
use the common context or directly modify model-specific code when necessary.

## Recorder

`KVRecorder` uses an inspectable directory layout:

```text
<root>/<run_name>/
  run_metadata.json
  samples/<sample_id>/
    sample_metadata.json
    scalars.jsonl
    stats.jsonl
    events.jsonl
    tensors.jsonl
    tensors/*.pt
```

Records are explicit. Full tensors are written only through `record_tensor`.
The recorder is available as `context.recorder`, but Observer callbacks decide
what to record.

## Boundaries

- Q is not exposed by the generic post-attention hook.
- Attention weights are available only when the selected attention path returns
  them or a method recomputes them.
- GQA/MQA distinguishes query heads from stored KV heads.
- Gathered cache indices do not automatically preserve original token IDs.
- Quantized-cache support uses the current Transformers/KVPress cache path and
  may depend on private cache fields.

## Downstream Workflow

Start a research branch from the frozen base version. First classify the work
as observation, Prefill intervention, Decode intervention, or a combination.
Reuse `KVContext`, `KVObserver`, `KVRecorder`, `pipeline.py`, `utils.py`, and
the common Press interfaces before modifying shared code.

Move a change back into KV-Base only when multiple research directions need
the same capability, such as lifecycle metadata, a generic observer hook, a
recording improvement, or a reusable cache utility. Keep paper-specific
scoring rules, one-off analysis, model workarounds, and specialized kernels in
the downstream project.
