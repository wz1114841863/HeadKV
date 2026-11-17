"""Dense KV storage model: exact small reference and equivalent scan counters.

No Attention approximation, tensor collection, or hardware timing measurement.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import time

import numpy as np

from experiments.head_analysis.analysis.pca_reconstruction import condition_weights
from experiments.head_analysis.common.io import atomic_file, read_json, write_json, write_npz


POLICIES = ("no_retention", "global_lru", "equal_head_lru", "equal_pin", "profile_pin", "shifted_profile_pin")
CAPACITIES = tuple(m * 1024**2 for m in (16, 32, 64))
COUNTERS = ("requests", "hits", "misses", "forward_only", "read_bytes", "hit_bytes",
            "write_bytes", "read_commands", "write_commands", "evictions_caused",
            "lookups", "lru_operations", "quota_checks", "inserts", "write_updates")
# Lookups/quota checks include append writes. Evictions are charged to the
# requesting Head, not the victim Head; these are control events, not cycles.
CI = {name: i for i, name in enumerate(COUNTERS)}
SEGMENTS = ("first", "after_first")


@dataclass(frozen=True)
class Geometry:
    layers: int = 32
    heads: int = 8
    dimension: int = 128
    page_tokens: int = 16
    query_group_size: int = 4

    @property
    def row_bytes(self):
        return 2 * self.dimension * 2  # K + V, BF16

    @property
    def page_bytes(self):
        return self.page_tokens * self.row_bytes

    @property
    def head_count(self):
        return self.layers * self.heads


def budget(capacity, policy, g=Geometry()):
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    common = g.page_bytes + g.heads * g.row_bytes + 256
    extra = (12*g.head_count if policy == "equal_head_lru" else
             4*g.head_count if policy in POLICIES[4:] else 0)
    tag = 24 if policy in POLICIES[1:3] else 16
    if capacity < common + extra:
        raise ValueError("KV budget does not cover the common buffers and policy state")
    slots = 0 if policy == "no_retention" else (capacity-common-extra)//(g.page_bytes+tag)
    used = common + extra + slots*(g.page_bytes+tag)
    return {"capacity_bytes": int(capacity), "slots": int(slots),
            "page_bytes": g.page_bytes, "data_bytes": int(slots*g.page_bytes),
            "tag_bytes": int(slots*tag), "policy_bytes": extra,
            "stream_buffer_bytes": g.page_bytes, "new_row_buffer_bytes": g.heads*g.row_bytes,
            "control_bytes": 256, "reserved_bytes": int(used),
            "unused_bytes": int(capacity-used), "tag_reset_operations": int(slots),
            "head_state_initialization_words": 3*g.head_count if policy == "equal_head_lru" else 0,
            "control_reset_operations": 1}


def divide(total, count):
    result = np.full(count, total//count, dtype=np.int64)
    result[:total % count] += 1
    return result


def quotas(slots, policy, score, g=Geometry()):
    layer_slots = divide(slots, g.layers)
    if policy not in POLICIES[4:]:
        return np.stack([divide(int(n), g.heads) for n in layer_slots])
    score = np.asarray(score)
    if score.shape != (g.layers, g.heads) or not np.isfinite(score).all():
        raise ValueError("finite, per-Layer/Head Calib scores required")
    result = np.zeros(score.shape, dtype=np.int64)
    denominator = g.heads*(g.heads+1)//2
    for layer, n in enumerate(layer_slots):
        order = np.lexsort((np.arange(g.heads), -score[layer]))
        weights = np.empty(g.heads, dtype=np.int64)
        weights[order] = np.arange(g.heads, 0, -1)
        result[layer] = n*weights//denominator
        remainder = n*weights % denominator
        winners = np.lexsort((np.arange(g.heads), -remainder))
        result[layer, winners[:int(n-result[layer].sum())]] += 1
    return np.roll(result, -1, axis=1) if policy == "shifted_profile_pin" else result


def _result(counts, ledger, peak, engine, **extra):
    return {"counts": counts, "budget": ledger, "peak_resident_pages": int(peak),
            "peak_resident_payload_bytes": int(peak*ledger["page_bytes"]),
            "engine": engine, **extra}


def simulate_reference(length, steps, capacity, policy, score=None, g=Geometry()):
    """Explicit causal requests, including write-through updates before each Layer."""
    ledger = budget(capacity, policy, g)
    allocation = quotas(ledger["slots"], policy, score, g)
    per_head = policy == "equal_head_lru"
    pin = policy in POLICIES[3:]
    caches = [OrderedDict() for _ in range(g.head_count if per_head else 1)]
    counts = np.zeros((2, g.layers, g.heads, len(COUNTERS)), dtype=np.int64)
    resident = peak = 0
    for step in range(steps):
        segment = int(step > 0)
        position = length+step
        last_page, offset = divmod(position, g.page_tokens)
        for layer in range(g.layers):
            # All new Head rows of this Layer exist before its Page reads.
            for head in range(g.heads):
                cache = caches[layer*g.heads+head if per_head else 0]
                row = counts[segment, layer, head]
                row[CI["write_bytes"]] += g.row_bytes
                row[CI["write_commands"]] += 1
                key = layer, head, last_page
                if pin:
                    row[CI["quota_checks"]] += 1
                eligible = policy != "no_retention" and (not pin or last_page < allocation[layer, head])
                if eligible:
                    row[CI["lookups"]] += 1
                if eligible and key in cache:
                    cache[key] = offset+1  # preserves OrderedDict order
                    row[CI["write_updates"]] += 1
            for head in range(g.heads):
                cache = caches[layer*g.heads+head if per_head else 0]
                limit = int(allocation[layer, head]) if per_head else ledger["slots"]
                stat = dict.fromkeys(COUNTERS, 0)
                for page in range(last_page+1):
                    key = layer, head, page
                    historical = min(g.page_tokens, position-page*g.page_tokens)
                    visible = min(g.page_tokens, position+1-page*g.page_tokens)
                    stat["requests"] += 1
                    if pin:
                        stat["quota_checks"] += 1
                    eligible = policy != "no_retention" and (not pin or page < allocation[layer, head])
                    if eligible:
                        stat["lookups"] += 1
                    hit = eligible and key in cache
                    if hit and cache[key] != visible:
                        raise ValueError("resident KV row validity is stale")
                    if historical == 0:
                        stat["forward_only"] += 1
                    elif hit:
                        stat["hits"] += 1
                        stat["hit_bytes"] += historical*g.row_bytes
                    else:
                        stat["misses"] += 1
                        stat["read_bytes"] += historical*g.row_bytes
                        stat["read_commands"] += 1
                    if eligible and limit:
                        if not hit:
                            if not pin and len(cache) == limit:
                                cache.popitem(last=False)
                                resident -= 1
                                stat["evictions_caused"] += 1
                            cache[key] = visible
                            resident += 1
                            stat["inserts"] += 1
                        if not pin:
                            cache.move_to_end(key)
                            stat["lru_operations"] += 1
                        peak = max(peak, resident)
                counts[segment, layer, head] += [stat[name] for name in COUNTERS]
    return _result(counts, ledger, peak, "reference", final_valid_rows={k: v for c in caches for k, v in c.items()})


def simulate_fast(length, steps, capacity, policy, score=None, g=Geometry()):
    """Exact counts for full scans with zero-hit LRU or full-Prompt prefix pins.

    Conditions are checked explicitly. Other dataflows use the small reference;
    this routine never silently approximates or truncates a request stream.
    """
    ledger = budget(capacity, policy, g)
    slots = ledger["slots"]
    allocation = quotas(slots, policy, score, g)
    full_prompt = length//g.page_tokens
    if length < 1 or steps < 0:
        raise ValueError("positive Prompt length and nonnegative Decode length required")
    if policy == "global_lru" and slots >= g.head_count*full_prompt:
        raise ValueError("global LRU fast path requires capacity below the scan reuse distance")
    if policy == "equal_head_lru" and int(allocation.max()) >= full_prompt:
        raise ValueError("per-Head LRU fast path requires smaller Head pools")
    if policy in POLICIES[3:] and int(allocation.max()) > full_prompt:
        raise ValueError("pin fast path requires fully populated Prompt pages")
    counts = np.zeros((2, g.layers, g.heads, len(COUNTERS)), dtype=np.int64)
    position = length + np.arange(steps, dtype=np.int64)
    first_pages = length//g.page_tokens+1
    for segment, q in enumerate((position[:1], position[1:])):
        if not len(q):
            continue
        n = len(q)
        requests = int((q//g.page_tokens+1).sum())
        misses = int(((q+g.page_tokens-1)//g.page_tokens).sum())
        row = counts[segment]
        for name, value in {"requests": requests, "misses": misses,
                            "forward_only": int((q % g.page_tokens == 0).sum()),
                            "read_bytes": int(q.sum())*g.row_bytes,
                            "write_bytes": n*g.row_bytes, "read_commands": misses,
                            "write_commands": n}.items():
            row[..., CI[name]] = value
        if policy == "no_retention":
            continue
        if policy in POLICIES[3:]:
            hits = allocation*n if segment else np.zeros_like(allocation)
            row[..., CI["hits"]] = hits
            row[..., CI["misses"]] -= hits
            row[..., CI["read_commands"]] -= hits
            row[..., CI["hit_bytes"]] = hits*g.page_bytes
            row[..., CI["read_bytes"]] -= hits*g.page_bytes
            row[..., CI["lookups"]] = allocation*n
            row[..., CI["quota_checks"]] = requests+n  # Page reads and append writes
            row[..., CI["inserts"]] = allocation if not segment else 0
        else:
            enabled = allocation > 0 if policy == "equal_head_lru" else slots > 0
            row[..., CI["lookups"]] = requests+n
            row[..., CI["inserts"]] = enabled*requests
            row[..., CI["lru_operations"]] = enabled*requests
            if segment:
                evictions = enabled*requests
                # Writes update existing tail rows without promoting them in LRU.
                updates = np.zeros((g.layers, g.heads), dtype=np.int64)
                for p in q[q % g.page_tokens != 0]:
                    if policy == "equal_head_lru":
                        updates += allocation > 0
                    else:
                        pages = int(p//g.page_tokens+1)
                        updates += (np.arange(1, g.heads+1)*pages-1 >= g.head_count*pages-slots)
                row[..., CI["write_updates"]] = updates
            else:
                if policy == "equal_head_lru":
                    warm = np.minimum(allocation, first_pages)
                else:
                    offset = np.arange(g.head_count).reshape(g.layers, g.heads)*first_pages
                    warm = np.clip(slots-offset, 0, first_pages)
                evictions = enabled*requests-warm
            row[..., CI["evictions_caused"]] = evictions
    return _result(counts, ledger, slots if steps else 0, "exact_scan")


def fit_profile(bundle, records):
    """Use only original Calib long inputs and the unchanged B1 RCO definition."""
    selected = [r for r in records if r["split"] == "calib" and r["dataset"] != "ruler"]
    with np.load(Path(bundle)/"head_profiles.npz") as z:
        ids = list(z["sample_ids"])
        scope = list(z["page_scopes"]).index("all_visible")
        k = list(z["rco_k_values"]).index(32)
        delta = list(z["rco_deltas"]).index(32)
        metric = list(z["rco_metrics"]).index("rco_raw_mean")
        indices = [ids.index(r["sample_id"]) for r in selected]
        values = z["rco_sample_head_values"][indices, :, :, scope, k, delta, metric].astype(np.float64)
        valid = z["rco_sample_head_valid_pair_count"][indices, :, :, scope, k, delta]
    if not np.isfinite(values).all() or not (valid > 0).all():
        raise ValueError("Calib long-input RCO values must have valid step pairs")
    weights = condition_weights([r["condition"] for r in selected], [r["source_group"] for r in selected])
    return np.einsum("s,slh->lh", weights, values), selected, weights


def load_inputs(run_set, source_root, bundle, b1_bundle, codec_root):
    from experiments.head_analysis.storage.b1_reader import load_completed_run_set
    from experiments.head_analysis.storage.representation_inputs import load_codec_cohorts
    index = load_completed_run_set(run_set, source_root=source_root, expected_sample_count=100,
                                   expected_batch_count=10, verify_file_hashes=False, verify_trace_arrays=False)
    cohorts = load_codec_cohorts(bundle, codec_root)
    batch_by_id = {sid: b for b in index.batches for sid in b.sample_ids}
    for rows in cohorts:
        for row in rows:
            batch = batch_by_id[row["sample_id"]]
            directory = batch.run_dir/"samples"/row["sample_id"]
            meta = read_json(directory/"sample_metadata.json")
            if row["prompt_length"] != meta["prompt_length"]:
                raise ValueError("B1 and frozen V3 Prompt lengths differ")
            row.update(decode_steps=meta["decode_trace_length"], b1_sample_dir=str(directory))
    first = cohorts[0][0]
    dimensions = batch_by_id[first["sample_id"]].schema["dimensions"]
    g = Geometry(first["num_layers"], first["num_kv_heads"], first["head_dim"],
                 batch_by_id[first["sample_id"]].schema["page_size"], dimensions["Hq"]//dimensions["Hkv"])
    score, profile_records, weights = fit_profile(b1_bundle, read_json(Path(bundle)/"split.json")["samples"])
    if len(profile_records) != 36 or len({r["source_group"] for r in profile_records}) != 30:
        raise ValueError("expected the approved 36 long Calib inputs / 30 sources")
    return cohorts, g, score, profile_records, weights


def check_positions(row, g):
    expected = 0
    for path in sorted(Path(row["b1_sample_dir"]).glob("decode_stats_*.npz")):
        with np.load(path) as z:
            s = z["decode_steps"].astype(np.int64)
            q = z["query_positions"].astype(np.int64)
            visible = z["visible_token_count"].astype(np.int64)
            if not (np.array_equal(s, np.arange(expected, expected+len(s)))
                    and np.array_equal(q, row["prompt_length"]+s)
                    and np.array_equal(visible, q+1)
                    and np.array_equal(z["page_candidate_count_all_visible"], (visible+g.page_tokens-1)//g.page_tokens)):
                raise ValueError(f"B1 continuous Dense positions differ: {row['sample_id']}")
            expected += len(s)
    if expected != row["decode_steps"]:
        raise ValueError("Decode position files do not cover the recorded length")


def check_counts(length, steps, result, g):
    counts = result["counts"].sum(axis=0)
    required = g.row_bytes*(steps*length+steps*(steps-1)//2)
    np.testing.assert_array_equal(counts[..., CI["requests"]],
        counts[..., CI["hits"]]+counts[..., CI["misses"]]+counts[..., CI["forward_only"]])
    np.testing.assert_array_equal(counts[..., CI["read_bytes"]]+counts[..., CI["hit_bytes"]],
                                  np.full((g.layers, g.heads), required))
    np.testing.assert_array_equal(counts[..., CI["write_bytes"]],
                                  np.full((g.layers, g.heads), steps*g.row_bytes))
    if (counts < 0).any() or result["peak_resident_pages"] > result["budget"]["slots"]:
        raise ValueError("invalid Dense counters or capacity overflow")


def result_row(record, capacity, policy, result, g):
    total = result["counts"].sum((0, 1, 2))
    row = {"sample_id": record["sample_id"], "condition": record["condition"],
           "prompt_length": record["prompt_length"], "decode_steps": record["decode_steps"],
           "capacity_bytes": capacity, "policy": policy, "engine": result["engine"],
           **dict(zip(COUNTERS, map(int, total))), **result["budget"],
           "peak_resident_pages": result["peak_resident_pages"],
           "peak_resident_payload_bytes": result["peak_resident_payload_bytes"]}
    for segment, name in enumerate(SEGMENTS):
        row.update({f"{name}_{key}": int(value) for key, value in
                    zip(COUNTERS, result["counts"][segment].sum((0, 1)))})
    length = record["prompt_length"]
    row.update(prefill_write_bytes=g.head_count*g.row_bytes*length,
               prefill_write_commands=g.head_count*((length+g.page_tokens-1)//g.page_tokens),
               quota_load_bytes=4*g.head_count if policy in POLICIES[4:] else 0,
               quota_load_commands=int(policy in POLICIES[4:]))
    row["decode_total_bytes"] = row["read_bytes"]+row["write_bytes"]
    row["full_total_bytes"] = row["decode_total_bytes"]+row["prefill_write_bytes"]+row["quota_load_bytes"]
    denom = row["hits"]+row["misses"]
    row["page_hit_rate"] = row["hits"]/denom if denom else None
    denom = row["read_bytes"]+row["hit_bytes"]
    row["byte_hit_rate"] = row["hit_bytes"]/denom if denom else None
    for bandwidth in (16, 32, 64):
        for overhead in (0, 100):
            row[f"decode_transfer_s_b{bandwidth}_n{overhead}"] = (
                row["decode_total_bytes"]/(bandwidth*1e9)+(row["read_commands"]+row["write_commands"])*overhead*1e-9)
            row[f"full_transfer_s_b{bandwidth}_n{overhead}"] = (
                row["full_total_bytes"]/(bandwidth*1e9)
                +(row["read_commands"]+row["write_commands"]+row["prefill_write_commands"]+row["quota_load_commands"])*overhead*1e-9)
    return row


def _write_csv(path, rows):
    with atomic_file(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(rows):
    """Input-equal within conditions; differences retain their actual byte costs."""
    result = []
    for capacity in sorted({r["capacity_bytes"] for r in rows}):
        for condition in sorted({r["condition"] for r in rows}):
            selected = [r for r in rows if r["capacity_bytes"] == capacity and r["condition"] == condition]
            for policy in POLICIES:
                candidates = [r for r in selected if r["policy"] == policy]
                output = {"capacity_bytes": capacity, "condition": condition, "policy": policy,
                          "sample_count": len(candidates)}
                for key in ("read_bytes", "write_bytes", "decode_total_bytes", "full_total_bytes",
                            "page_hit_rate", "byte_hit_rate", "read_commands", "write_commands"):
                    output[key] = float(np.mean([r[key] for r in candidates]))
                for reference in ("no_retention", "equal_pin", "shifted_profile_pin"):
                    base = [r for r in selected if r["policy"] == reference]
                    for key in ("decode_total_bytes", "full_total_bytes"):
                        difference = sum(r[key] for r in base)-sum(r[key] for r in candidates)
                        output[f"{key}_saved_vs_{reference}"] = difference/len(candidates)
                result.append(output)
    return result


def run_analysis(run_set, source_root, bundle, b1_bundle, codec_root, output, *, capacities=CAPACITIES):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cohorts, g, score, profile_records, weights = load_inputs(run_set, source_root, bundle, b1_bundle, codec_root)
    all_counts, all_rows = [], []
    reference_report = []
    for phase, records in zip(("calib", "validation"), cohorts):
        for i, record in enumerate(records):
            check_positions(record, g)
            sample_counts = []
            for capacity in capacities:
                methods = []
                for policy in POLICIES:
                    result = simulate_fast(record["prompt_length"], record["decode_steps"], capacity, policy, score, g)
                    check_counts(record["prompt_length"], record["decode_steps"], result, g)
                    if phase == "calib" and i == 0 and capacity == capacities[len(capacities)//2]:
                        n = min(2, record["decode_steps"])
                        explicit = simulate_reference(record["prompt_length"], n, capacity, policy, score, g)
                        fast = simulate_fast(record["prompt_length"], n, capacity, policy, score, g)
                        np.testing.assert_array_equal(explicit["counts"], fast["counts"])
                        if explicit["peak_resident_pages"] != fast["peak_resident_pages"]:
                            raise ValueError("reference/fast peak occupancy differs")
                        reference_report.append({"sample_id": record["sample_id"], "steps": n,
                                                 "capacity_bytes": capacity, "policy": policy, "matched": True})
                    methods.append(result["counts"])
                    all_rows.append(result_row(record, capacity, policy, result, g))
                # Same quota multiset and metadata: Head identity must not change traffic.
                for key in ("read_bytes", "write_bytes", "read_commands", "write_commands", "hits", "misses"):
                    a = methods[4][..., CI[key]].sum(); b = methods[5][..., CI[key]].sum()
                    if a != b:
                        raise ValueError(f"Profile/Shifted identity equivalence failed: {key}")
                sample_counts.append(np.stack(methods))
            all_counts.append(np.stack(sample_counts))
            print(f"Dense memory {phase}: {record['sample_id']}", flush=True)
        phase_rows = [r for r in all_rows if r["sample_id"] in {s["sample_id"] for s in records}]
        _write_csv(output/f"{phase}.csv", phase_rows)
        if phase == "calib":
            write_json(output/"config.json", {"status": "frozen_before_validation", "geometry": asdict(g),
                "capacities_bytes": list(capacities), "policies": list(POLICIES),
                "sources": {"run_set": str(run_set), "source_root": str(source_root), "bundle": str(bundle),
                            "b1_bundle": str(b1_bundle), "codec_root": str(codec_root)},
                "calib_ids": [r["sample_id"] for r in cohorts[0]], "validation_ids": [r["sample_id"] for r in cohorts[1]],
                "profile_calib_ids": [r["sample_id"] for r in profile_records],
                "dataflow": "layer/head/page ascending; grouped Query reuse; write-through/no-write-allocate; cold start",
                "bandwidth_GB_s": [16, 32, 64], "command_overhead_ns": [0, 100],
                "exploratory": True, "old_test_used": False})
            write_npz(output/"profile.npz", {"score": score, "fit_weights": weights,
                "fit_sample_ids": np.array([r["sample_id"] for r in profile_records]),
                "capacities_bytes": np.array(capacities), "policies": np.array(POLICIES),
                "quotas": np.array([[quotas(budget(c,p,g)["slots"],p,score,g) for p in POLICIES] for c in capacities], dtype=np.uint32)})
    write_npz(output/"head_counters.npz", {"counts": np.stack(all_counts), "sample_ids": np.array([r["sample_id"] for rs in cohorts for r in rs]),
        "capacities_bytes": np.array(capacities), "policies": np.array(POLICIES),
        "segments": np.array(SEGMENTS), "counters": np.array(COUNTERS),
        "axes": np.array(["sample", "capacity", "policy", "segment", "layer", "kv_head", "counter"])})
    report = {"status": "completed", "reference_check": reference_report,
              "calib": summarize([r for r in all_rows if r["sample_id"] in {s["sample_id"] for s in cohorts[0]}]),
              "validation": summarize([r for r in all_rows if r["sample_id"] in {s["sample_id"] for s in cohorts[1]}]),
              "identity_equivalence_passed": True, "exploratory": True,
              "full_scan_conservation_passed": True,
              "hardware_or_quality_measured": False, "elapsed_seconds": time.perf_counter()-started}
    write_json(output/"summary.json", report)
    print(f"Dense memory analysis completed: {output}", flush=True)
    return report
