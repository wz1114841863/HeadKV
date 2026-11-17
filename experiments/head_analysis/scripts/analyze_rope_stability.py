"""Analyze V2 RoPE rank changes across the Pilot and extension inputs (CPU only)."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.head_analysis.analysis.rope_stability import run_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="extension V2 spectrum/CKA package")
    parser.add_argument("--reference-bundle", type=Path, required=True, help="six Pilot spectrum/CKA package")
    parser.add_argument("--sample-index", type=Path, required=True, help="existing B2 sample_index.json with target lengths and Pair IDs")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; preserve existing results")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)
    run_analysis(args.bundle, args.reference_bundle, args.output_dir, sample_index=args.sample_index,
                 bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    print(f"results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
