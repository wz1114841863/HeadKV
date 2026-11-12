"""Analyze completed V2 KV samples with aligned B1 Attention profiles."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.head_analysis.analysis.kv_representation import run_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True,
                        help="completed V2 collection; repeat to combine runs")
    parser.add_argument("--b1-bundle", type=Path, required=True,
                        help="completed B1 numeric bundle containing head_profiles.npz")
    parser.add_argument("--output-dir", type=Path, required=True, help="new analysis directory")
    parser.add_argument("--permutation-replicates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args(argv)
    run_analysis(args.run_dir, args.output_dir, b1_bundle=args.b1_bundle,
                 permutation_replicates=args.permutation_replicates, seed=args.seed)
    print(f"results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
