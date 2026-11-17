"""Compare V2 cross-Head CKA relations using already published numerical values."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.head_analysis.analysis.cka_stability import run_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, required=True, help="existing B2 sample_index.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; preserve existing packages")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)
    run_analysis(args.bundle, args.reference_bundle, args.output_dir, sample_index=args.sample_index,
                 top_k=args.top_k, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    print(f"results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
