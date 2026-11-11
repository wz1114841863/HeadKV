"""Explore whether Prefill CV adds held-out Head stability information."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.head_analysis.analysis.prefill_stability import CV_ID, METRICS, TARGETS, analyze, load_inputs
from experiments.head_analysis.common.io import write_json, write_npz


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="existing B2 completed numeric bundle")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; existing outputs are preserved")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261111)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    inputs = load_inputs(args.bundle)
    def progress(label):
        print(f"completed: {label}", flush=True)
    rows, summaries, boot = analyze(inputs, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed,
                                     batch_size=args.batch_size, progress=progress)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "sample_metrics.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_npz(args.output_dir / "bootstrap.npz", boot)
    write_json(args.output_dir / "summary.json", {
        "analysis": "prefill_cv_decode_stability_exploratory_v1",
        "source_bundle": str(args.bundle.resolve()), "source_run_set_sha256": inputs.source_run_set,
        "source_numeric_file": "stages/stage4/stage4_joint_profiles.npz", "sample_index": "sample_index.json",
        "cv_metric": CV_ID, "targets": TARGETS, "metrics": METRICS,
        "split": "within each condition, leave one sample group out; PG19 group is pair_id",
        "history": "paired-finite training mean RCO per Head",
        "history_cv": "history + pooled within-Head training OLS slope * (test CV - training mean CV)",
        "correlation": "pairwise-valid, per-layer centered Head profiles; Spearman over flattened Heads",
        "mse": "per-sample equally weighted mean squared error across common valid Heads",
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed,
                      "unit": "sample; PG19 pair", "method": "paired resampling with full refit; exclude all held-out group copies",
                      "interval": "2.5 and 97.5 percentile of finite resample means", "batch_size": args.batch_size},
        "limits": ["exploratory split of already-studied inputs; not an independent new-data validation",
                   "one model; no causal, compression, accuracy or system-efficiency inference",
                   "RULER has only six Decode steps; long delta is unavailable",
                   "sample index does not establish source-document independence"],
        "sample_metrics": "sample_metrics.jsonl", "bootstrap_arrays": "bootstrap.npz",
        "conditions": summaries,
    })
    print(f"results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
